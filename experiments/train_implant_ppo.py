"""첫 RL 실행: MaskablePPO, 장비군 3개 공유 정책 (2026-11-12).

학습: 도착 기록 시드 1, 14일 구간을 무작위로 뽑는다. 시험: 기록 시드 2, 3 전체(240일), 고장 시드 1~2.
비교: lssu, cov(재생 튜닝값), PPO(앞 공정 정보 있음/없음).
지표: 주 지표 D = lot당 (대기 + 그 lot이 일으킨 셋업), 60일 이후 도착분. 낮을수록 좋다.

실행: python3 experiments/train_implant_ppo.py --steps 200000 [--no_upstream]
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import implant_env as E  # noqa: E402
import implant_gym as G  # noqa: E402

OUT = ROOT / "results" / "rl_ppo"
TEST = [(f, rec, ss) for f in E.FAMILIES for rec in (2, 3) for ss in (1, 2)]


class Scaled(G.ImplantEnv):
    def step(self, a):
        o, r, d, t, i = super().step(a)
        return o, r / 10, d, t, i


def make_env(rank, use_upstream):
    def f():
        return Scaled(seeds=(1,), window_days=14, use_upstream=use_upstream, rng_seed=rank)
    return f


def eval_rule(job):
    fam, rec, ss, kind = job
    if kind == "lssu":
        pol = E.lssu
    else:
        q, t = G.TUNED_COV[fam]
        pol = E.thr(q, t * 60, cov=True)
    s = G.make_sim(fam, rec, ss, pol).run()
    return job, E.summarize(s)["delay_h"]


def eval_model(job):
    path, use_up, fam, rec, ss = job
    import torch
    torch.set_num_threads(1)
    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(path, device="cpu")

    def act(o, m):
        a, _ = model.predict(o, action_masks=m, deterministic=True)
        return int(a)
    return (fam, rec, ss), G.run_policy(fam, rec, ss, act, use_up)["delay_h"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200000)
    p.add_argument("--no_upstream", action="store_true")
    p.add_argument("--envs", type=int, default=8)
    a = p.parse_args()
    use_up = not a.no_upstream
    tag = "up" if use_up else "noup"
    OUT.mkdir(parents=True, exist_ok=True)

    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    venv = SubprocVecEnv([make_env(i, use_up) for i in range(a.envs)])
    model = MaskablePPO("MlpPolicy", venv, n_steps=512, batch_size=512, gamma=0.99, learning_rate=3e-4,
                        ent_coef=0.01, policy_kwargs=dict(net_arch=[64, 64]), verbose=0, seed=0, device="cpu")
    t0 = time.time()
    model.learn(total_timesteps=a.steps)
    path = OUT / f"ppo_{tag}_{a.steps}.zip"
    model.save(path)
    venv.close()
    print(f"학습 {a.steps} step, {(time.time() - t0) / 60:.1f}분", flush=True)

    res = {}
    with ProcessPoolExecutor(8) as ex:
        for (fam, rec, ss), d in ex.map(eval_model, [(str(path), use_up, *j) for j in TEST]):
            res.setdefault(fam, {}).setdefault("ppo", []).append(d)
        for (fam, rec, ss, kind), d in ex.map(eval_rule, [(*j, k) for j in TEST for k in ("lssu", "cov")]):
            res.setdefault(fam, {}).setdefault(kind, []).append(d)
    print("\n장비군        lssu    cov     PPO    PPO−cov")
    for fam, r in res.items():
        m = {k: float(np.mean(v)) for k, v in r.items()}
        print(f"{fam:12s} {m['lssu']:6.2f}  {m['cov']:6.2f}  {m['ppo']:6.2f}  {100 * (m['ppo'] / m['cov'] - 1):+5.1f}%")
    (OUT / f"eval_{tag}_{a.steps}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
