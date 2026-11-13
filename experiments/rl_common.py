"""PPO와 DQN이 같이 쓰는 부분: 네트워크, 검증(체크포인트 고르기), 시험, 결과 출력 (2026-11-13).

검증과 시험을 나눈 이유
- 학습 중 보상은 매번 다른 14일 구간에서 나와 부하가 달라서, 숫자만 보고는 정책이 좋아졌는지 알 수 없다.
- 그래서 고정된 구간 18개(장비군 3 × 시작일 6, 고장 시드 고정)에서 일정 간격으로 탐욕 정책을 돌리고,
  같은 구간에서 cov가 낸 비용 대비 비율로 점수를 매긴다. 이 점수가 가장 낮은 시점의 모델을 저장한다.
- 검증은 학습(기록 시드 1)과 독립인 기록 시드 VAL_REC(=7)를 쓴다 (2026-11-13 수정).
  전에는 시드 1의 날짜 구간을 썼는데, 학습 구간(60~234일)과 도착이 겹쳐 검증 점수가 부풀었다(0.81~0.88 대 시험 cov 대비 −3%).
  최종 시험은 학습·검증에 쓰지 않은 기록 시드 2·3이다. 검증 점수는 고르기에만 쓰고 성능으로 보고하지 않는다.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import implant_env as E  # noqa: E402
import implant_gym as G  # noqa: E402

TEST = [(f, rec, ss) for f in E.FAMILIES for rec in (2, 3) for ss in (1, 2, 3, 4, 5)]  # 2026-11-13: 고장 시드 1~2 → 1~5
VAL_REC = 7  # 검증 전용 공장 기록 시드 (학습 1, 시험 2·3, 공장 검증 4~6과 겹치지 않음)
VAL_DAYS0 = (70, 95, 120, 145, 170, 195)  # 겹치지 않는 14일 구간 6개 (장비군당), 2026-11-13 3개 → 6개
VAL = [(f, d, 9000 + i) for f in E.FAMILIES for i, d in enumerate(VAL_DAYS0)]  # (장비군, 시작일, 고장 시드)
VAL_DAYS = 14
R_SCALE = 0.1


def mlp(n_in, n_out, h=64):
    return nn.Sequential(nn.Linear(n_in, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, n_out))


class ActorCritic(nn.Module):
    def __init__(self, n_obs=G.N_OBS, n_act=3, h=64):
        super().__init__()
        self.pi, self.v = mlp(n_obs, n_act, h), mlp(n_obs, 1, h)

    def dist(self, obs, mask):
        return torch.distributions.Categorical(logits=self.pi(obs).masked_fill(~mask, -1e9))

    def value(self, obs):
        return self.v(obs).squeeze(-1)

    def scores(self, obs):  # 탐욕 행동용
        return self.pi(obs)


class QNet(nn.Module):
    def __init__(self, n_obs=G.N_OBS, n_act=3, h=64):
        super().__init__()
        self.q = mlp(n_obs, n_act, h)

    def forward(self, obs):
        return self.q(obs)

    def scores(self, obs):
        return self.q(obs)


NETS = {"ppo": ActorCritic, "dqn": QNet}


def sampler(net, seed=0):
    """확률 정책 그대로 뽑기 (PPO용). 결정이 촘촘하면 '결정마다 작은 확률로 교체'가 곧 시간에 따른 교체 규칙이라,
    argmax로 바꾸면 그 규칙이 '절대 교체 안 함'으로 무너진다."""
    g = torch.Generator().manual_seed(seed)

    def act(o, m):
        with torch.no_grad():
            p = torch.softmax(net.scores(torch.as_tensor(o)).masked_fill(~torch.as_tensor(m), -1e9), -1)
        return int(torch.multinomial(p, 1, generator=g))
    return act


def greedy(net):
    def act(o, m):
        with torch.no_grad():
            s = net.scores(torch.as_tensor(o)).masked_fill(~torch.as_tensor(m), -1e9)
        return int(s.argmax())
    return act


def _load(kind, state):
    torch.set_num_threads(1)
    net = NETS[kind]()
    net.load_state_dict(state if isinstance(state, dict) else torch.load(state))
    net.eval()
    return net


# ---- 검증
def _window_cost_cov(job):
    fam, day, ss = job
    q, t = G.TUNED_COV[fam]
    sim = G.make_sim(fam, VAL_REC, ss, E.thr(q, t * 60, cov=True))
    t0 = day * 86400
    sim.run(until=t0)
    a0 = sim.area
    sim.run(until=t0 + VAL_DAYS * 86400)
    return (sim.area - a0) / 3600 / len(sim.ms)


_COV_COST = None


def cov_val_costs(ex):
    """검증 구간별 cov 비용 (장비 1대당 lot·시간). 한 번만 계산한다."""
    global _COV_COST
    if _COV_COST is None:
        _COV_COST = list(ex.map(_window_cost_cov, VAL))
    return _COV_COST


def _val_job(job):
    kind, state, use_up, (fam, day, ss), stoch, wait_min = job
    net = _load(kind, state)
    act = sampler(net, ss) if stoch else greedy(net)
    env = G.ImplantEnv(seeds=(VAL_REC,), use_upstream=use_up, wait_min=wait_min)
    o, _ = env.reset(options=dict(fam=fam, seed=VAL_REC, sim_seed=ss, t_start=day * 86400, t_end=(day + VAL_DAYS) * 86400,
                                  count_from_start=True))  # 분모(cov)와 같이 구간 시작부터 센다 (2026-11-13)
    cost = 0.0
    while True:
        o, r, _, done, _ = env.step(act(o, env.action_masks()))
        cost -= r
        if done:
            return cost


def validate(ex, kind, net, use_up, stoch=False, wait_min=0):
    """검증 점수 = 구간별 (정책 비용 / cov 비용)의 평균. 1보다 작으면 cov보다 낫다."""
    state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    base = cov_val_costs(ex)
    costs = list(ex.map(_val_job, [(kind, state, use_up, w, stoch, wait_min) for w in VAL]))
    ratios = [c / b for c, b in zip(costs, base)]
    per_fam = {f: float(np.mean([r for (ff, _, _), r in zip(VAL, ratios) if ff == f])) for f in E.FAMILIES}
    return float(np.mean(ratios)), per_fam


# ---- 시험
def _eval_job(job):
    kind, path, use_up, fam, rec, ss, stoch, wait_min = job
    net = _load(kind, path)
    act = sampler(net, 100 * rec + ss) if stoch else greedy(net)
    return (fam, rec, ss), G.run_policy(fam, rec, ss, act, use_up, wait_min)


def _rule_job(job):
    fam, rec, ss, kind = job
    if kind == "lssu":
        pol = E.lssu
    else:
        q, t = G.TUNED_COV[fam]
        pol = E.thr(q, t * 60, cov=True)
    return job, E.summarize(G.make_sim(fam, rec, ss, pol).run())


def test(ex, kind, path, use_up, tag, stoch=False, wait_min=0):
    res = {}
    for (fam, rec, ss), s in ex.map(_eval_job, [(kind, str(path), use_up, *j, stoch, wait_min) for j in TEST]):
        res.setdefault(fam, {}).setdefault(kind, []).append(s)
    for (fam, rec, ss, k), s in ex.map(_rule_job, [(*j, k) for j in TEST for k in ("lssu", "cov")]):
        res.setdefault(fam, {}).setdefault(k, []).append(s)
    print(f"\n[{tag}] 시험(기록 시드 2·3 × 고장 시드 1·2), lot당 지연 D(시간), hot lot 지연, 교체 시간 비율")
    for fam, r in res.items():
        m = {k: np.mean([x["delay_h"] for x in v]) for k, v in r.items()}  # cov 대비 % = 평균 D끼리의 비율
        h = {k: np.mean([x["hot_delay_h"] for x in v]) for k, v in r.items()}
        su = {k: np.mean([x["setup_ratio"] for x in v]) for k, v in r.items()}
        print(f"{fam:12s} lssu {m['lssu']:.2f}  cov {m['cov']:.2f}  {kind.upper()} {m[kind]:.2f} "
              f"(cov 대비 {100 * (m[kind] / m['cov'] - 1):+.1f}%)  hot cov/{kind} {h['cov']:.2f}/{h[kind]:.2f}  "
              f"교체 lssu/cov/{kind} {100 * su['lssu']:.1f}/{100 * su['cov']:.1f}/{100 * su[kind]:.1f}%", flush=True)
    return res
