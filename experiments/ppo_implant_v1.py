"""직접 구현한 PPO (수업 9장 구조) + 행동 마스킹 + 시간 기준 할인. implant 교체 결정용.

9장과 같은 부분
- 1. 수집: N개 환경에서 T step씩 현재 정책으로 모은다.
- 2. GAE(λ)로 advantage, 가치 목표를 계산한다.
- 3. K epoch × 미니배치로 clipped surrogate − 가치 손실 + 엔트로피 보너스를 최적화한다.
- 4. 배치를 버리고 다시 수집한다.

이 문제 때문에 바꾼 부분
- 마스킹: 대기 lot이 없는 가스로는 바꿀 수 없다. 막힌 행동의 logit을 매우 작은 값으로 둔다.
- 시간 기준 할인: 결정 간격이 몇 초 ~ 몇 시간으로 들쭉날쭉하다. 한 step의 할인을 γ_h^(흐른 시간)으로 둔다(SMDP).
- 중단(truncated): 14일 구간 끝은 우리가 자른 것이라 가치로 이어 붙인다(bootstrap). 1장의 episodic/continuing 구분.

실행: python3 experiments/ppo_implant.py --iters 150 [--no_upstream] [--seed 0]
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import implant_env as E  # noqa: E402
import implant_gym as G  # noqa: E402

OUT = ROOT / "results" / "rl_ppo"
TEST = [(f, rec, ss) for f in E.FAMILIES for rec in (2, 3) for ss in (1, 2)]
R_SCALE = 0.1


class ActorCritic(nn.Module):
    def __init__(self, n_obs, n_act=3, h=64):
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(n_obs, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, n_act))
        self.v = nn.Sequential(nn.Linear(n_obs, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))

    def dist(self, obs, mask):
        logits = self.pi(obs).masked_fill(~mask, -1e9)
        return torch.distributions.Categorical(logits=logits)

    def value(self, obs):
        return self.v(obs).squeeze(-1)


def collect(envs, state, net, T):
    """envs마다 T step을 모은다. state: 환경별 (obs, mask). 반환: 환경별 궤적 목록."""
    trajs = []
    for k, env in enumerate(envs):
        obs, mask = state[k]
        tr = {key: [] for key in ("obs", "mask", "a", "logp", "v", "r", "g", "end", "boot")}
        for _ in range(T):
            o = torch.as_tensor(obs).unsqueeze(0)
            m = torch.as_tensor(mask).unsqueeze(0)
            with torch.no_grad():
                d = net.dist(o, m)
                a = d.sample()
                lp, v = d.log_prob(a).item(), net.value(o).item()
            obs2, r, term, trunc, info = env.step(a.item())
            tr["obs"].append(obs); tr["mask"].append(mask); tr["a"].append(a.item())
            tr["logp"].append(lp); tr["v"].append(v); tr["r"].append(r * R_SCALE)
            tr["g"].append(info["dt_h"])
            if term or trunc:
                with torch.no_grad():  # 구간 끝: 마지막 관측의 가치로 이어 붙인다
                    tr["boot"].append(0.0 if term else net.value(torch.as_tensor(obs2).unsqueeze(0)).item())
                tr["end"].append(True)
                obs, _ = env.reset()
                mask = env.action_masks().copy()
            else:
                tr["boot"].append(0.0)
                tr["end"].append(False)
                obs, mask = obs2, env.action_masks().copy()
        with torch.no_grad():
            last_v = net.value(torch.as_tensor(obs).unsqueeze(0)).item()
        state[k] = (obs, mask)
        trajs.append((tr, last_v))
    return trajs


def gae(trajs, gamma_h, lam, lam_time=False):
    obs, mask, a, logp, adv, ret = [], [], [], [], [], []
    for tr, last_v in trajs:
        n = len(tr["r"])
        A = np.zeros(n)
        nxt_adv, nxt_v = 0.0, last_v
        for t in reversed(range(n)):
            g = gamma_h ** tr["g"][t]
            if tr["end"][t]:
                nxt_v, nxt_adv = tr["boot"][t], 0.0
            delta = tr["r"][t] + g * nxt_v - tr["v"][t]
            lt = lam ** tr["g"][t] if lam_time else lam  # lam_time: λ도 흐른 시간 기준 (짧은 step이 많아도 보는 범위가 줄지 않게)
            A[t] = delta + g * lt * nxt_adv
            nxt_adv, nxt_v = A[t], tr["v"][t]
        obs += tr["obs"]; mask += tr["mask"]; a += tr["a"]; logp += tr["logp"]
        adv += list(A); ret += list(A + np.array(tr["v"]))
    f = lambda x, dt: torch.as_tensor(np.array(x), dtype=dt)  # noqa: E731
    return f(obs, torch.float32), f(mask, torch.bool), f(a, torch.int64), f(logp, torch.float32), \
        f(adv, torch.float32), f(ret, torch.float32)


def update(net, opt, batch, K, mb, clip, c_v, c_ent):
    obs, mask, a, logp_old, adv, ret = batch
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    n = len(a)
    for _ in range(K):
        for idx in torch.randperm(n).split(mb):
            d = net.dist(obs[idx], mask[idx])
            ratio = torch.exp(d.log_prob(a[idx]) - logp_old[idx])
            l_clip = torch.min(ratio * adv[idx], torch.clamp(ratio, 1 - clip, 1 + clip) * adv[idx]).mean()
            l_v = ((net.value(obs[idx]) - ret[idx]) ** 2).mean()
            loss = -(l_clip - c_v * l_v + c_ent * d.entropy().mean())
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()


def eval_job(job):
    path, use_up, fam, rec, ss = job
    torch.set_num_threads(1)
    net = ActorCritic(G.N_OBS)
    net.load_state_dict(torch.load(path))

    def act(o, m):
        with torch.no_grad():
            logits = net.pi(torch.as_tensor(o)).masked_fill(~torch.as_tensor(m), -1e9)
        return int(logits.argmax())
    return (fam, rec, ss), G.run_policy(fam, rec, ss, act, use_up)


def rule_job(job):
    fam, rec, ss, kind = job
    if kind == "lssu":
        pol = E.lssu
    else:
        q, t = G.TUNED_COV[fam]
        pol = E.thr(q, t * 60, cov=True)
    return job, E.summarize(G.make_sim(fam, rec, ss, pol).run())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=150)
    p.add_argument("--envs", type=int, default=8)
    p.add_argument("--T", type=int, default=256, help="환경당 수집 step")
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--mb", type=int, default=256)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--gamma_h", type=float, default=0.9, help="1시간당 할인")
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lam_time", action="store_true", help="GAE λ를 1시간당 값으로 보고 λ^(흐른 시간)을 쓴다")
    p.add_argument("--ent", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_upstream", action="store_true")
    a = p.parse_args()
    use_up = not a.no_upstream
    tag = f"{'up' if use_up else 'noup'}{'_lt' if a.lam_time else ''}_s{a.seed}"
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(2)

    envs = [G.ImplantEnv(seeds=(1,), window_days=14, use_upstream=use_up, rng_seed=a.seed * 100 + i)
            for i in range(a.envs)]
    state = []
    for env in envs:
        o, _ = env.reset()
        state.append((o, env.action_masks().copy()))
    net = ActorCritic(G.N_OBS)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    log, t0 = [], time.time()
    for it in range(a.iters):
        trajs = collect(envs, state, net, a.T)
        batch = gae(trajs, a.gamma_h, a.lam, a.lam_time)
        update(net, opt, batch, a.K, a.mb, a.clip, 0.5, a.ent)
        mean_r = float(np.mean([np.mean(tr["r"]) for tr, _ in trajs])) / R_SCALE
        wait_share = float((batch[2] == 0).float().mean())
        log.append(dict(it=it, mean_r=mean_r, wait_share=wait_share, sec=time.time() - t0))
        if it % 10 == 0 or it == a.iters - 1:
            print(f"[{tag}] iter {it:4d}  step당 보상 {mean_r:8.3f}  기다리기 비율 {wait_share:.2f}  "
                  f"{(time.time() - t0) / 60:5.1f}분", flush=True)
    path = OUT / f"ppo_{tag}.pt"
    torch.save(net.state_dict(), path)

    res = {}
    with ProcessPoolExecutor(8) as ex:
        for (fam, rec, ss), s in ex.map(eval_job, [(str(path), use_up, *j) for j in TEST]):
            res.setdefault(fam, {}).setdefault("ppo", []).append(s)
        for (fam, rec, ss, kind), s in ex.map(rule_job, [(*j, k) for j in TEST for k in ("lssu", "cov")]):
            res.setdefault(fam, {}).setdefault(kind, []).append(s)
    print(f"\n[{tag}] 시험(기록 시드 2·3 × 고장 시드 1·2), lot당 지연 D(시간)과 교체 시간 비율")
    for fam, r in res.items():
        m = {k: np.mean([x["delay_h"] for x in v]) for k, v in r.items()}
        su = {k: np.mean([x["setup_ratio"] for x in v]) for k, v in r.items()}
        print(f"{fam:12s} lssu {m['lssu']:.2f}  cov {m['cov']:.2f}  PPO {m['ppo']:.2f} "
              f"(cov 대비 {100 * (m['ppo'] / m['cov'] - 1):+.1f}%)  교체 lssu/cov/PPO "
              f"{100 * su['lssu']:.1f}/{100 * su['cov']:.1f}/{100 * su['ppo']:.1f}%")
    (OUT / f"result_{tag}.json").write_text(json.dumps(dict(args=vars(a), log=log, eval=res), indent=1))


if __name__ == "__main__":
    main()
