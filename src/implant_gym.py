"""implant 재생 환경을 gymnasium 환경으로 감싼다 (2026-11-12, 첫 RL 실행용).

- 결정: 장비가 비었고, 현재 가스가 있고, 현재 가스 lot이 없고, hot lot이 없을 때. (hot lot이 있거나
  가스가 없으면 cov 규칙이 처리한다. docs/implant_sar_mapping.md)
- 상태: implant_state.transform(encode()) 23개 + 앞 공정 lot 수 6개 = 29개. use_upstream=False면 마지막 6개를 0으로.
- 행동 A1: 0 = 다음 사건까지 기다리기, 1 = g1로 교체, 2 = g2로 교체. 대기 lot이 없는 가스는 마스킹.
- 보상 R1: 직전 결정 이후 늘어난 (대기 lot 수 × 시간 + 교체 시간)을 장비 수 × 시간으로 나눈 값의 음수.
- 에피소드: 장비군 하나, 기록 하나에서 window_days 구간. 그 앞은 cov로 돌린다.
"""
import bisect
import random
from collections import deque

import gymnasium as gym
import numpy as np
from greenlet import greenlet

import implant_env as E
import implant_state as S

TUNED_COV = {"Implant_128": (5, 120), "Implant_132": (3, 240), "Implant_91": (7, 240)}  # 재생 환경 튜닝값 (EDA 06)
N_OBS = len(S.NAMES) + 6
EVENT_DT_H = 2.0 / 60  # '다음 사건까지' 방식의 평균 결정 간격(cov 기준, 2026-11-13 측정 약 2분)


def per_decision_prob(rate_h, wait_min):
    """시간당 비율(쉬는 장비 1시간당 횟수)을 결정 1번당 확률로 바꾼다.
    결정 간격 = wait_min분(고정 시간) 또는 EVENT_DT_H(다음 사건까지). p = 1 − exp(−rate × 간격)."""
    import math
    dt = wait_min / 60 if wait_min > 0 else EVENT_DT_H
    return 1 - math.exp(-rate_h * dt)


class _LogDeque(deque):
    def __init__(self, sink):
        super().__init__()
        self.sink = sink

    def append(self, x):
        self.sink.append(x[0])
        super().append(x)


_CHAINS = {}


def make_sim(fam, seed, sim_seed, policy, fab_downs=False):
    """fab_downs=True면 고장·PM을 sim_seed 난수가 아니라 같은 기록 시드의 공장 기록(downs.csv)으로 재생한다."""
    if (fam, seed) not in _CHAINS:  # 기록 읽기는 프로세스마다 한 번만
        _CHAINS[(fam, seed)] = E.load_chains(fam, seed, 12 * 3600)[0]
    sched = E.fab_schedule(fam, seed, sim_seed) if fab_downs else None
    if fab_downs and sched is None:
        raise FileNotFoundError(f"공장 고장 기록 없음: {E.record_path(seed, 'downs.csv')}")
    sim = E.Sim(fam, _CHAINS[(fam, seed)], policy, seed=sim_seed, sched=sched)
    arr = [[] for _ in range(sim.n_gas)]
    sim.q = [_LogDeque(arr[g]) for g in range(sim.n_gas)]
    sim.qh = [_LogDeque(arr[g]) for g in range(sim.n_gas)]
    sim._arr_log = arr
    return sim


def observe(s, m, use_upstream=True):
    """(관측 벡터, 상대 가스 순서)."""
    ends = {}
    for e in s.ev:
        if e[2] == E.DONE and e[4] == s.busy_id[e[3]]:
            ends[e[3]] = e[0]
    others = [(o.gas if o.gas >= 0 else None, not o.free, o.left is not None, ends.get(o.idx)) for o in s.ms if o is not m]
    queue = {}
    for g in range(s.n_gas):
        firsts = [x[0][0] for x in (s.q[g], s.qh[g]) if x]
        queue[g] = (len(s.q[g]), len(s.qh[g]), min(firsts) if firsts else None)
    arr1h = {g: len(a) - bisect.bisect_left(a, s.t - 3600) for g, a in enumerate(s._arr_log)}
    v = S.View(n=len(s.ms), setup_s=s.setup, now=s.t, gases=list(range(s.n_gas)), gas0=m.gas,
               waited_s=s.t - m.hold_since, pm_pieces=min(m.pieces), others=others, queue=queue, arr1h=arr1h)
    gs = S.order(v)
    if use_upstream:
        up = s.upstream()
        u = [up[0][g] / v.n for g in gs] + [up[1][g] / v.n for g in gs]
        u = [float(np.log1p(x)) for x in u]  # 최대 11까지 커서 다른 변수(대부분 3 이하)에 맞춘다
    else:
        u = [0.0] * 6
    return np.array(S.transform(S.encode(v)) + u, dtype=np.float32), gs


def rl_turn(s, m):
    """RL이 결정할 차례인가."""
    return m.gas >= 0 and not any(s.qh)


class ImplantEnv(gym.Env):
    def __init__(self, fams=tuple(E.FAMILIES), seeds=(1,), window_days=14, use_upstream=True, rng_seed=0, wait_min=0):
        """wait_min: 0이면 '기다리기' = 다음 사건(도착, 다른 장비 끝 등)에 다시 묻기.
        0보다 크면 '기다리기' = wait_min분 동안 다시 묻지 않기(그사이 지금 가스 lot이 오면 바로 처리)."""
        self.wait_s = wait_min * 60
        self.fams, self.seeds, self.window = list(fams), list(seeds), window_days * 86400
        self.use_upstream = use_upstream
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (N_OBS,), np.float32)
        self.action_space = gym.spaces.Discrete(3)
        self.rng = random.Random(rng_seed)
        self._mask = np.ones(3, bool)

    def action_masks(self):
        return self._mask

    def _sim_main(self):
        self.sim.run(until=self.t_end)
        return None

    def _policy(self, s, m):
        if s.t < self.t_start or not rl_turn(s, m):
            return self.base(s, m)
        obs, gs = observe(s, m, self.use_upstream)
        mask = np.array([True, s.waiting(gs[1]) > 0, s.waiting(gs[2]) > 0])
        a = self.parent.switch((obs, mask))
        if a == 0:
            if self.wait_s > 0:
                m.forced_until = s.t + self.wait_s  # 이 시각까지 교체 결정을 다시 묻지 않는다
                return None, m.forced_until
            return None, None
        return gs[a], None

    def _advance(self, action=None):
        out = self.g.switch(action) if action is not None else self.g.switch()
        area = self.sim.area
        r = -(area - self.last_area) / 3600 / len(self.sim.ms)
        self.last_area = area
        self.dt_h = (self.sim.t - self.last_t) / 3600  # 이번 결정부터 다음 결정까지 흐른 시간 (할인에 쓴다)
        self.last_t = self.sim.t
        if out is None or self.g.dead:
            return self.last_obs, r, True
        obs, self._mask = out
        self.last_obs = obs
        return obs, r, False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        o = options or {}
        fam = o.get("fam", self.rng.choice(self.fams))
        rec = o.get("seed", self.rng.choice(self.seeds))
        sim_seed = o.get("sim_seed", self.rng.randrange(10 ** 6))
        self.fam = fam
        q, tmax = TUNED_COV[fam]
        self.base = E.thr(q, tmax * 60, cov=True)
        self.t_start = o.get("t_start", E.WARMUP + self.rng.uniform(0, 160 * 86400))
        self.t_end = o.get("t_end", self.t_start + self.window)
        self.sim = make_sim(fam, rec, sim_seed, self._policy)
        self.sim.hot_breaks_wait = self.wait_s > 0
        self.parent = greenlet.getcurrent()
        self.g = greenlet(self._sim_main)
        self.last_area = 0.0
        # 시작 전 구간은 cov로 돌고, 첫 RL 결정에서 멈춘다. 시작 전 비용은 보상에 넣지 않는다.
        out = self.g.switch()
        self.last_area = self.sim.area
        self.last_t = self.sim.t
        if out is None:
            return np.zeros(N_OBS, np.float32), {}
        obs, self._mask = out
        self.last_obs = obs
        return obs, {}

    def step(self, action):
        """구간 끝은 과제가 끝난 것이 아니라 우리가 자른 것이므로 terminated가 아니라 truncated다(1장).
        truncated일 때 obs는 마지막 결정의 관측이다(다음 결정 관측이 없으므로 가치 추정의 근사로 쓴다)."""
        obs, r, done = self._advance(int(action))
        return obs, r, False, done, {"dt_h": self.dt_h}


def run_policy(fam, rec, sim_seed, act, use_upstream=True, wait_min=0):
    """act(obs, mask) -> 행동. 240일 기록 전체를 돌려 summarize 결과를 돌려준다."""
    q, tmax = TUNED_COV[fam]
    base = E.thr(q, tmax * 60, cov=True)

    cnt = [0, 0]  # [RL이 고른 결정, cov가 대신 처리한 결정] (교체가 필요한 결정만)

    def pol(s, m):
        if not rl_turn(s, m):
            cnt[1] += 1
            return base(s, m)
        cnt[0] += 1
        obs, gs = observe(s, m, use_upstream)
        mask = np.array([True, s.waiting(gs[1]) > 0, s.waiting(gs[2]) > 0])
        a = act(obs, mask)
        if a == 0 and wait_min > 0:
            m.forced_until = s.t + wait_min * 60
            return None, m.forced_until
        return (None, None) if a == 0 else (gs[a], None)
    sim = make_sim(fam, rec, sim_seed, pol)
    sim.hot_breaks_wait = wait_min > 0
    sim.run()
    out = E.summarize(sim)
    out["rl_share"] = cnt[0] / max(1, sum(cnt))  # 결정 중 RL이 고른 비율 (나머지는 hot lot·가스 없음으로 cov가 처리)
    return out
