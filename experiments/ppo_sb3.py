"""PPO = SB3 MaskablePPO (sb3-contrib), 우리 환경·검증·시험과 같은 조건으로 (2026-11-15, 교수님 답: 라이브러리 사용 허용).

우리가 바꾸거나 붙인 것 (설계·기법, 보고서에 쓸 부분)
- 환경: src/implant_gym.ImplantEnv 그대로(상태 29개, 행동 마스크 action_masks(), 보상 R1, wait_min).
- 시간 기준 할인(SMDP): SB3 PPO는 step마다 같은 γ를 쓴다. 결정 간격이 들쭉날쭉하므로 rollout buffer의 GAE 계산을
  γ_h^(흐른 시간), λ_h^(흐른 시간)으로 바꾼 TimeGAEBuffer를 쓴다(직접 구현 PPO의 --lam_time과 같은 식).
  흐른 시간은 env info["dt_h"]를 콜백이 buffer에 넣는다.
- 탐색 기준점: 처음 정책의 기다리기 확률을 --wait_init_rate_h로 맞춘다(출력 층 편향, 직접 구현과 같음).
- 엔트로피 계수 ent → ent_end 선형 감소, 학습률 선형 감소.
- 검증: rl_common.validate (기록 시드 7, 18구간, 확률 정책 그대로)로 val_every rollout마다, 가장 좋은 시점 저장.
- 저장 형식: SB3 정책 가중치를 우리 ActorCritic 형식으로 옮겨 results/rl_ppo/ppo_up_lt<suffix>_s<seed>.pt 로 저장한다.
  그래서 evaluate.py(--kind ppo --stoch), select_e1.py, pick_policy.py, 공장 RL 모드를 그대로 쓴다.
  SB3 원본 모델도 *_sb3.zip으로 남긴다.
- 알려진 근사: 14일 구간 끝(truncated)에서 SB3가 다음 가치를 이어 붙일 때는 고정 γ(γ_h^(대표 간격))를 쓴다. 에피소드당 1번.

실행: python3 experiments/ppo_sb3.py --iters 300 --wait_init_rate_h 0.3 --ent 0.003 --ent_end 0 --wait_min <W> --suffix _<태그> --seed <s>
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
import implant_gym as G  # noqa: E402
import rl_common as C  # noqa: E402
from sb3_contrib import MaskablePPO  # noqa: E402
from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

OUT = ROOT / "results" / "rl_ppo"


class TimeGAEBuffer(MaskableRolloutBuffer):
    """GAE를 흐른 시간 기준 할인으로: δ_t = r_t + γ_h^Δt V(s_{t+1}) − V(s_t), A_t = δ_t + γ_h^Δt λ_h^Δt A_{t+1}."""
    gamma_h, lam_h = 0.9, 0.95

    def reset(self):
        super().reset()
        self.dts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def compute_returns_and_advantage(self, last_values, dones):
        last_values = last_values.clone().cpu().numpy().flatten()
        last_gae = 0.0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                nnt, nv = 1.0 - dones.astype(np.float32), last_values
            else:
                nnt, nv = 1.0 - self.episode_starts[step + 1], self.values[step + 1]
            g = self.gamma_h ** self.dts[step]
            lt = self.lam_h ** self.dts[step]
            delta = self.rewards[step] + g * nv * nnt - self.values[step]
            last_gae = delta + g * lt * nnt * last_gae
            self.advantages[step] = last_gae
        self.returns = self.advantages + self.values


class ScaledReward(gym.Wrapper):
    def step(self, a):
        o, r, te, tr, info = self.env.step(a)
        return o, r * C.R_SCALE, te, tr, info

    def action_masks(self):
        return self.env.action_masks()


def sb3_to_ac(policy):
    """SB3 정책(pi=[64,64], vf=[64,64], Tanh) 가중치를 rl_common.ActorCritic 형식으로."""
    sd = policy.state_dict()
    m = {"pi.0": "mlp_extractor.policy_net.0", "pi.2": "mlp_extractor.policy_net.2", "pi.4": "action_net",
         "v.0": "mlp_extractor.value_net.0", "v.2": "mlp_extractor.value_net.2", "v.4": "value_net"}
    out = {}
    for k, src in m.items():
        out[f"{k}.weight"] = sd[f"{src}.weight"].detach().cpu().clone()
        out[f"{k}.bias"] = sd[f"{src}.bias"].detach().cpu().clone()
    return out


class TrainCallback(BaseCallback):
    def __init__(self, a, tag, use_up, ex, path):
        super().__init__()
        self.a, self.tag, self.use_up, self.ex, self.path = a, tag, use_up, ex, path
        self.it, self.sim_h, self.best, self.best_it, self.log = 0, 0.0, float("inf"), -1, []
        self.t0 = time.time()
        self.r_sum = self.n = self.waits = 0

    def _on_step(self):
        buf = self.model.rollout_buffer
        infos = self.locals["infos"]
        dts = np.array([i.get("dt_h", 0.0) for i in infos], dtype=np.float32)
        buf.dts[buf.pos] = dts
        self.sim_h += float(dts.sum())
        self.r_sum += float(np.sum(self.locals["rewards"])) / C.R_SCALE
        self.n += len(infos)
        self.waits += int(np.sum(self.locals["actions"] == 0))
        return True

    def _on_rollout_end(self):
        a = self.a
        if a.ent_end is not None:
            self.model.ent_coef = a.ent + (a.ent_end - a.ent) * self.it / max(a.iters - 1, 1)
        row = dict(it=self.it, sim_days=self.sim_h / 24, mean_r=self.r_sum / max(self.n, 1),
                   wait_share=self.waits / max(self.n, 1), sec=time.time() - self.t0)
        self.r_sum = self.n = self.waits = 0
        if (self.it + 1) % a.val_every == 0 or self.it == a.iters - 1:
            net = C.ActorCritic()
            net.load_state_dict(sb3_to_ac(self.model.policy))
            v, vf = C.validate(self.ex, "ppo", net, self.use_up, True, a.wait_min)
            row.update(val=v, val_fam=vf)
            if v < self.best:
                self.best, self.best_it = v, self.it
                torch.save(net.state_dict(), self.path)
                self.model.save(str(self.path).replace(".pt", "_sb3.zip"))
            print(f"[{self.tag}] iter {self.it:4d}  step당 보상 {row['mean_r']:8.3f}  기다리기 {row['wait_share']:.2f}  "
                  f"검증 {v:.3f} {' '.join(f'{x:.2f}' for x in vf.values())}  최고 {self.best:.3f}@{self.best_it}  "
                  f"날수 {row['sim_days']:.0f}  {(time.time() - self.t0) / 60:5.1f}분", flush=True)
        self.log.append(row)
        self.it += 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=300, help="rollout 횟수 (rollout당 envs × T step)")
    p.add_argument("--envs", type=int, default=8)
    p.add_argument("--T", type=int, default=256, help="환경당 rollout step")
    p.add_argument("--K", type=int, default=4, help="epoch")
    p.add_argument("--mb", type=int, default=256)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--gamma_h", type=float, default=0.9, help="1시간당 할인")
    p.add_argument("--lam", type=float, default=0.95, help="1시간당 GAE λ")
    p.add_argument("--lr", type=float, default=3e-4, help="처음 학습률 (끝에서 0이 되게 선형 감소)")
    p.add_argument("--ent", type=float, default=0.003)
    p.add_argument("--ent_end", type=float, default=0.0)
    p.add_argument("--wait_init_rate_h", type=float, default=0.3, help="처음 정책의 교체 빈도(쉬는 장비 1시간당). 없애려면 음수")
    p.add_argument("--wait_min", type=float, default=0)
    p.add_argument("--val_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_upstream", action="store_true")
    p.add_argument("--suffix", default="")
    a = p.parse_args()
    use_up = not a.no_upstream
    tag = f"{'up' if use_up else 'noup'}_lt{a.suffix}_s{a.seed}"  # 직접 구현 PPO와 같은 이름 규칙 (select_e1·evaluate가 그대로 읽음)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"ppo_{tag}.pt"
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(2)

    venv = DummyVecEnv([lambda i=i: ScaledReward(G.ImplantEnv(seeds=(1,), window_days=14, use_upstream=use_up,
                                                              rng_seed=a.seed * 100 + i, wait_min=a.wait_min))
                        for i in range(a.envs)])
    rep_dt = (a.wait_min / 60) if a.wait_min > 0 else G.EVENT_DT_H
    model = MaskablePPO(
        "MlpPolicy", venv, learning_rate=lambda prog: a.lr * prog, n_steps=a.T, batch_size=a.mb, n_epochs=a.K,
        gamma=a.gamma_h ** rep_dt, gae_lambda=a.lam, clip_range=a.clip, ent_coef=a.ent, vf_coef=0.5, max_grad_norm=0.5,
        rollout_buffer_class=TimeGAEBuffer, seed=a.seed, device="cpu", verbose=0,
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]), activation_fn=nn.Tanh))
    model.rollout_buffer.gamma_h, model.rollout_buffer.lam_h = a.gamma_h, a.lam
    if a.wait_init_rate_h is not None and a.wait_init_rate_h >= 0:
        wait_init = 1 - G.per_decision_prob(a.wait_init_rate_h, a.wait_min)
        with torch.no_grad():
            model.policy.action_net.weight.mul_(0.01)
            sw = (1 - wait_init) / 2
            model.policy.action_net.bias.copy_(torch.log(torch.tensor([wait_init, sw, sw])))
        print(f"wait_init_rate_h {a.wait_init_rate_h}/h, wait_min {a.wait_min} → 처음 기다리기 확률 {wait_init:.4f}", flush=True)

    ex = ProcessPoolExecutor(8)
    cb = TrainCallback(a, tag, use_up, ex, path)
    model.learn(total_timesteps=a.iters * a.envs * a.T, callback=cb)
    net = C.ActorCritic()
    net.load_state_dict(sb3_to_ac(model.policy))
    torch.save(net.state_dict(), OUT / f"ppo_{tag}_last.pt")
    print(f"[{tag}] 검증 최고 iter {cb.best_it} 점수 {cb.best:.3f}로 시험", flush=True)
    res = C.test(ex, "ppo", path, use_up, tag, True, a.wait_min)
    (OUT / f"result_{tag}.json").write_text(json.dumps(dict(args=vars(a), impl="sb3_maskable_ppo", best_it=cb.best_it,
                                                            best_val=cb.best, log=cb.log, eval=res), indent=1))


if __name__ == "__main__":
    main()
