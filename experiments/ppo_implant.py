"""직접 구현한 PPO (수업 9장 구조, 2026-11-13 검증 체크포인트 추가. 이전 판은 ppo_implant_v1.py) + 행동 마스킹 + 시간 기준 할인. implant 교체 결정용.

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
sys.path.insert(0, str(ROOT / "experiments"))
import implant_env as E  # noqa: E402,F401
import implant_gym as G  # noqa: E402
import rl_common as C  # noqa: E402
from rl_common import ActorCritic, R_SCALE  # noqa: E402

OUT = ROOT / "results" / "rl_ppo"


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
    p.add_argument("--val_every", type=int, default=10, help="몇 iter마다 검증해서 가장 좋은 모델을 저장할지")
    p.add_argument("--lr_decay", action="store_true", help="학습률을 끝에서 0이 되도록 선형으로 줄인다")
    p.add_argument("--suffix", default="", help="결과 파일 이름 뒤에 붙일 표시")
    p.add_argument("--wait_min", type=float, default=0, help="0보다 크면 '기다리기' = 이 분 동안 다시 묻지 않기 (팀 결정: 고정 시간)")
    p.add_argument("--eval_sample", action="store_true",
                   help="검증·시험에서 argmax 대신 확률 정책에서 뽑는다. 결정이 촘촘해 '결정마다 작은 확률로 교체'가 학습되므로 argmax는 그 규칙을 지운다")
    p.add_argument("--wait_init", type=float, default=0.0,
                   help="0보다 크면 처음 정책의 기다리기 확률을 이 값으로 맞춘다(출력 층 편향). 결정이 촘촘해서 고르게 뽑으면 교체를 남발한다")
    p.add_argument("--wait_init_rate_h", type=float, default=None,
                   help="처음 정책의 교체 빈도를 시간 기준으로: 쉬는 장비 1시간당 교체 횟수. wait_init을 덮어쓴다 "
                        "(0.3 = '다음 사건까지'에서 wait_init 0.99와 같음)")
    p.add_argument("--ent_end", type=float, default=None, help="엔트로피 계수를 ent에서 이 값까지 선형으로 줄인다")
    p.add_argument("--no_upstream", action="store_true")
    a = p.parse_args()
    if a.wait_init_rate_h is not None:
        a.wait_init = 1 - G.per_decision_prob(a.wait_init_rate_h, a.wait_min)
        print(f"wait_init_rate_h {a.wait_init_rate_h}/h, wait_min {a.wait_min} → 처음 기다리기 확률 {a.wait_init:.4f}", flush=True)
    use_up = not a.no_upstream
    tag = f"{'up' if use_up else 'noup'}{'_lt' if a.lam_time else ''}{a.suffix}_s{a.seed}"
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(2)

    envs = [G.ImplantEnv(seeds=(1,), window_days=14, use_upstream=use_up, rng_seed=a.seed * 100 + i, wait_min=a.wait_min)
            for i in range(a.envs)]
    state = []
    for env in envs:
        o, _ = env.reset()
        state.append((o, env.action_masks().copy()))
    net = ActorCritic(G.N_OBS)
    if a.wait_init > 0:
        last = net.pi[-1]
        with torch.no_grad():
            last.weight.mul_(0.01)  # 처음엔 상태와 거의 상관없이 편향이 행동 확률을 정한다
            sw = (1 - a.wait_init) / 2
            last.bias.copy_(torch.log(torch.tensor([a.wait_init, sw, sw])))
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    ex = ProcessPoolExecutor(8)
    log, t0 = [], time.time()
    sim_h = 0.0  # 모든 환경이 경험한 시뮬레이션 시간 합
    best, best_it, path = float("inf"), -1, OUT / f"ppo_{tag}.pt"
    for it in range(a.iters):
        if a.lr_decay:
            for pg in opt.param_groups:
                pg["lr"] = a.lr * (1 - it / a.iters)
        trajs = collect(envs, state, net, a.T)
        sim_h += sum(sum(tr["g"]) for tr, _ in trajs)
        batch = gae(trajs, a.gamma_h, a.lam, a.lam_time)
        ent = a.ent if a.ent_end is None else a.ent + (a.ent_end - a.ent) * it / max(a.iters - 1, 1)
        update(net, opt, batch, a.K, a.mb, a.clip, 0.5, ent)
        mean_r = float(np.mean([np.mean(tr["r"]) for tr, _ in trajs])) / R_SCALE
        wait_share = float((batch[2] == 0).float().mean())
        row = dict(it=it, sim_days=sim_h / 24, mean_r=mean_r, wait_share=wait_share, sec=time.time() - t0)
        if (it + 1) % a.val_every == 0 or it == a.iters - 1:
            v, vf = C.validate(ex, "ppo", net, use_up, a.eval_sample, a.wait_min)
            row.update(val=v, val_fam=vf)
            if v < best:
                best, best_it = v, it
                torch.save(net.state_dict(), path)
            print(f"[{tag}] iter {it:4d}  step당 보상 {mean_r:8.3f}  기다리기 {wait_share:.2f}  "
                  f"검증(cov=1) {v:.3f} {' '.join(f'{x:.2f}' for x in vf.values())}  최고 {best:.3f}@{best_it}  "
                  f"{(time.time() - t0) / 60:5.1f}분", flush=True)
        log.append(row)
    torch.save(net.state_dict(), OUT / f"ppo_{tag}_last.pt")
    print(f"[{tag}] 검증 최고 iter {best_it} 점수 {best:.3f}로 시험", flush=True)
    res = C.test(ex, "ppo", path, use_up, tag, a.eval_sample, a.wait_min)
    (OUT / f"result_{tag}.json").write_text(json.dumps(dict(args=vars(a), best_it=best_it, best_val=best, log=log, eval=res), indent=1))

if __name__ == "__main__":
    main()
