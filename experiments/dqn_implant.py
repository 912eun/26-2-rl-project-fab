"""직접 구현한 DQN (Double DQN + 경험 재생 + 타깃 네트워크) + 행동 마스킹 + 시간 기준 할인. implant 교체 결정용 (2026-11-13).

DQN 구조 (수업 DQN 장)
- 1. ε-탐욕으로 행동하고 (s, a, r, s', 마스크', 흐른 시간, 끝) 을 재생 버퍼에 넣는다.
- 2. 버퍼에서 무작위 미니배치를 뽑아 목표 y = r + γ^Δt · Q_타깃(s', argmax_a' Q(s', a'))로 Huber 손실을 줄인다.
     Double DQN: 행동 고르기는 온라인 망, 값 매기기는 타깃 망 (최댓값 과대추정 완화).
- 3. 타깃 망은 일정 step마다 온라인 망을 복사한다.

이 문제 때문에 바꾼 부분 (PPO와 같은 처리)
- 마스킹: 다음 상태에서 고를 수 없는 행동은 argmax에서 뺀다. 탐험할 때도 고를 수 있는 행동 안에서만 뽑는다.
- 시간 기준 할인: 할인을 γ_h^(흐른 시간)으로 둔다(SMDP).
- 14일 구간 끝은 truncated라서 목표에서 다음 가치를 지우지 않는다.
- (선택, --nstep_h) 시간 기준 다단계 목표. 결정이 몇 초~몇 분 간격이라 1단계 목표는 Q 추정을 수백 번 이어 붙이고,
  그때마다 max의 과대추정이 쌓여 기다리기 가치가 부풀 수 있다. 실제 보상을 nstep_h 시간만큼 더한 뒤에만 Q로 이어 붙인다.
  주의: 그 사이 행동이 지금 정책과 다를 수 있어 off-policy 편향이 있다(Rainbow의 n-step과 같은 근사).

PPO와 비교 조건을 맞춘 부분: 같은 환경, 같은 보상 크기(R_SCALE), 같은 망 크기, 같은 검증·시험 (rl_common.py).

실행: python3 experiments/dqn_implant.py --steps 600000 [--no_upstream] [--seed 0]
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
import implant_gym as G  # noqa: E402
import rl_common as C  # noqa: E402
from rl_common import QNet, R_SCALE  # noqa: E402

OUT = ROOT / "results" / "rl_dqn"


class Replay:
    def __init__(self, cap, n_obs):
        self.cap, self.i, self.n = cap, 0, 0
        self.o = np.zeros((cap, n_obs), np.float32)
        self.o2 = np.zeros((cap, n_obs), np.float32)
        self.m2 = np.zeros((cap, 3), bool)
        self.a = np.zeros(cap, np.int64)
        self.r = np.zeros(cap, np.float32)
        self.dt = np.zeros(cap, np.float32)
        self.term = np.zeros(cap, np.float32)

    def add(self, o, a, r, o2, m2, dt, term):
        k = self.i
        self.o[k], self.a[k], self.r[k], self.o2[k], self.m2[k], self.dt[k], self.term[k] = o, a, r, o2, m2, dt, term
        self.i = (k + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, b):
        idx = np.random.randint(0, self.n, b)
        t = lambda x: torch.as_tensor(x[idx])  # noqa: E731
        return t(self.o), t(self.a), t(self.r), t(self.o2), t(self.m2), t(self.dt), t(self.term)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=600_000, help="환경 step 총수 (모든 환경 합)")
    p.add_argument("--envs", type=int, default=8)
    p.add_argument("--buffer", type=int, default=200_000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma_h", type=float, default=0.9, help="1시간당 할인")
    p.add_argument("--train_every", type=int, default=4, help="환경 step 몇 번마다 경사 1번")
    p.add_argument("--target_every", type=int, default=2000, help="경사 몇 번마다 타깃 망 복사")
    p.add_argument("--learn_start", type=int, default=10_000)
    p.add_argument("--eps_end", type=float, default=0.05)
    p.add_argument("--eps_frac", type=float, default=0.3, help="전체 step 중 ε을 1에서 eps_end로 줄이는 비율 (--eps_steps가 없을 때)")
    p.add_argument("--eps_steps", type=int, default=None,
                   help="ε을 1에서 eps_end로 줄이는 step 수를 최대 학습량과 떼어 고정한다 (예: 360000). 주면 eps_frac 무시")
    p.add_argument("--val_every", type=int, default=20_480, help="환경 step 몇 번마다 검증 (PPO 10 iter와 같은 양)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--patience", type=int, default=0,
                   help="0보다 크면 조기 종료: 가장 좋은 검증 점수가 min_delta 이상 좋아지지 않은 검증이 이 횟수만큼 연속이면 멈춘다(--steps는 최대)")
    p.add_argument("--min_delta", type=float, default=0.01)
    p.add_argument("--min_steps", type=int, default=200_000, help="조기 종료를 이 step 전에는 하지 않는다")
    p.add_argument("--no_upstream", action="store_true")
    p.add_argument("--suffix", default="")
    p.add_argument("--wait_min", type=float, default=0, help="0보다 크면 '기다리기' = 이 분 동안 다시 묻지 않기 (팀 결정: 고정 시간)")
    p.add_argument("--nstep_h", type=float, default=0.0,
                   help="0보다 크면 시간 기준 다단계 목표: 흐른 시간이 이 값(시간) 이상이 될 때까지 실제 보상을 더하고 그 뒤에서만 Q로 이어 붙인다")
    p.add_argument("--nstep_max", type=int, default=64, help="다단계 목표에 묶는 최대 결정 수")
    p.add_argument("--explore_rate_h", type=float, default=None,
                   help="ε 탐색의 교체를 시간 기준으로: 쉬는 장비 1시간당 횟수. 결정 1번당 확률로 바꿔 explore_switch를 덮어쓴다 "
                        "(0.6 = '다음 사건까지'에서 결정당 0.02와 같음)")
    p.add_argument("--explore_switch", type=float, default=None,
                   help="ε 탐색에서 교체를 고를 확률. 없으면 고를 수 있는 행동 중 고르게. 결정이 촘촘해서 고르게 뽑으면 교체를 남발한다")
    a = p.parse_args()
    if a.explore_rate_h is not None:
        a.explore_switch = G.per_decision_prob(a.explore_rate_h, a.wait_min)
        print(f"explore_rate_h {a.explore_rate_h}/h, wait_min {a.wait_min} → 결정당 교체 탐색 확률 {a.explore_switch:.4f}", flush=True)
    use_up = not a.no_upstream
    tag = f"{'up' if use_up else 'noup'}{a.suffix}_s{a.seed}"
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    torch.set_num_threads(2)

    envs = [G.ImplantEnv(seeds=(1,), window_days=14, use_upstream=use_up, rng_seed=a.seed * 100 + i, wait_min=a.wait_min)
            for i in range(a.envs)]
    pend = [[] for _ in range(a.envs)]  # 환경별 아직 목표를 못 만든 전이 (o, a, r, dt)

    def emit(k, o2, m2, term, flush):
        """pend[k] 앞에서부터, 묶은 시간이 nstep_h 이상(또는 flush)이면 다단계 전이를 만들어 버퍼에 넣는다.
        목표 = Σ γ^(앞 전이들의 누적 시간) · r_j + γ^(전체 시간) · Q(마지막 다음 상태)."""
        q_ = pend[k]
        while q_:
            tot = sum(x[3] for x in q_)
            if not flush and tot < a.nstep_h and len(q_) < a.nstep_max:
                return
            R, cum = 0.0, 0.0
            for (_, _, r_, dt_) in q_:
                R += (a.gamma_h ** cum) * r_
                cum += dt_
            o0, a0 = q_[0][0], q_[0][1]
            buf.add(o0, a0, R, o2, m2, cum, term)
            q_.pop(0)

    state = []
    for env in envs:
        o, _ = env.reset()
        state.append((o, env.action_masks().copy()))
    q, qt = QNet(), QNet()
    qt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=a.lr)
    buf = Replay(a.buffer, G.N_OBS)
    ex = ProcessPoolExecutor(8)

    log, t0 = [], time.time()
    best, best_step, path = float("inf"), -1, OUT / f"dqn_{tag}.pt"
    step, n_upd, r_sum, r_n, waits, losses = 0, 0, 0.0, 0, 0, []
    sim_h = 0.0  # 모든 환경이 경험한 시뮬레이션 시간 합 (학습량을 날수 축으로 보기 위해)
    next_val = a.val_every
    no_improve, stop_reason = 0, "max"
    while step < a.steps:
        eps_len = a.eps_steps if a.eps_steps else a.eps_frac * a.steps
        eps = max(a.eps_end, 1 - (1 - a.eps_end) * step / eps_len)
        for k, env in enumerate(envs):
            o, m = state[k]
            if rng.random() < eps:
                ok = np.flatnonzero(m)
                if a.explore_switch is not None and len(ok) > 1:
                    act = 0 if rng.random() >= a.explore_switch else int(rng.choice(ok[1:]))
                else:
                    act = int(rng.choice(ok))
            else:
                with torch.no_grad():
                    act = int(q(torch.as_tensor(o)).masked_fill(~torch.as_tensor(m), -1e9).argmax())
            o2, r, term, trunc, info = env.step(act)
            m2 = env.action_masks().copy()
            if a.nstep_h > 0:
                pend[k].append((o, act, r * R_SCALE, info["dt_h"]))
                emit(k, o2, m2, float(term), flush=term or trunc)
            else:
                buf.add(o, act, r * R_SCALE, o2, m2, info["dt_h"], float(term))
            r_sum += r; r_n += 1; waits += act == 0
            sim_h += info["dt_h"]
            if term or trunc:
                o2, _ = env.reset()
                m2 = env.action_masks().copy()
            state[k] = (o2, m2)
            step += 1
            if buf.n >= a.learn_start and step % a.train_every == 0:
                bo, ba, br, bo2, bm2, bdt, bterm = buf.sample(a.batch)
                with torch.no_grad():
                    a2 = q(bo2).masked_fill(~bm2, -1e9).argmax(1, keepdim=True)  # 고르기: 온라인 망
                    v2 = qt(bo2).gather(1, a2).squeeze(1)                         # 값: 타깃 망
                    y = br + (a.gamma_h ** bdt) * (1 - bterm) * v2
                qsa = q(bo).gather(1, ba.unsqueeze(1)).squeeze(1)
                loss = F.smooth_l1_loss(qsa, y)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 10.0)
                opt.step()
                losses.append(loss.item())
                n_upd += 1
                if n_upd % a.target_every == 0:
                    qt.load_state_dict(q.state_dict())
        if step >= next_val or step >= a.steps:
            next_val += a.val_every
            v, vf = C.validate(ex, "dqn", q, use_up, False, a.wait_min)
            row = dict(step=step, sim_days=sim_h / 24, eps=eps, mean_r=r_sum / max(r_n, 1), wait_share=waits / max(r_n, 1),
                       loss=float(np.mean(losses)) if losses else None, val=v, val_fam=vf, sec=time.time() - t0)
            log.append(row)
            no_improve = 0 if v < best - a.min_delta else no_improve + 1
            if v < best:
                best, best_step = v, step
                torch.save(q.state_dict(), path)
            print(f"[dqn_{tag}] step {step:7d}  ε {eps:.2f}  step당 보상 {row['mean_r']:8.3f}  기다리기 {row['wait_share']:.2f}  "
                  f"손실 {row['loss'] or 0:.4f}  검증(cov=1) {v:.3f} {' '.join(f'{x:.2f}' for x in vf.values())}  "
                  f"최고 {best:.3f}@{best_step}  {(time.time() - t0) / 60:5.1f}분", flush=True)
            r_sum, r_n, waits, losses = 0.0, 0, 0, []
            if a.patience > 0 and step >= a.min_steps and no_improve >= a.patience:
                stop_reason = f"patience {a.patience} at step {step}"
                print(f"[dqn_{tag}] 조기 종료: 검증 {a.patience}번 연속 {a.min_delta} 이상 개선 없음 (step {step})", flush=True)
                break
    torch.save(q.state_dict(), OUT / f"dqn_{tag}_last.pt")
    print(f"[dqn_{tag}] 검증 최고 step {best_step} 점수 {best:.3f}로 시험", flush=True)
    res = C.test(ex, "dqn", path, use_up, f"dqn_{tag}", False, a.wait_min)
    (OUT / f"result_{tag}.json").write_text(json.dumps(dict(args=vars(a), best_step=best_step, best_val=best, steps_run=step, stop_reason=stop_reason,
                                                                 sim_days=sim_h / 24, log=log, eval=res), indent=1))


if __name__ == "__main__":
    main()
