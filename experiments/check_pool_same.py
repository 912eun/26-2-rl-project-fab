"""검증 일꾼 방식을 바꿔도 검증 점수가 같은지 확인 (2026-11-15~16, E1 P1).
저장된 모델을 (가) 예전 방식 ProcessPoolExecutor(8) (나) 지금 방식 rl_common.FreshPool(8)로
각각 검증해 점수가 같은지, 그리고 학습 때 기록된 best_val과 같은지 본다.
(첫 수정안 max_tasks_per_child=20도 같은 점수였지만 Python 3.12에서 가끔 멈춰 FreshPool로 바꿨다.)
실행: python3 experiments/check_pool_same.py
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
import rl_common as C  # noqa: E402

CASES = [("dqn", ROOT / "results/rl_dqn/dqn_up_e1ev_s0.pt", ROOT / "results/rl_dqn/result_up_e1ev_s0.json", False),
         ("ppo", ROOT / "results/rl_ppo/ppo_up_lt_e1ev_s2.pt", ROOT / "results/rl_ppo/result_up_lt_e1ev_s2.json", True)]

if __name__ == "__main__":
    for kind, pt, res, stoch in CASES:
        net = C._load(kind, str(pt))
        out = {}
        for name in ("예전", "지금"):
            C._COV_COST = None  # cov 기준도 각 방식으로 다시 계산
            if name == "예전":
                with ProcessPoolExecutor(8) as ex:
                    out[name] = C.validate(ex, kind, net, True, stoch, 0)
            else:
                out[name] = C.validate(C.FreshPool(8), kind, net, True, stoch, 0)
        rec = json.loads(res.read_text())["best_val"]
        a, b = out["예전"][0], out["지금"][0]
        print(f"{pt.name}: 예전 {a:.6f}  지금 {b:.6f}  학습 기록 {rec:.6f}  예전=지금 {a == b}  "
              f"장비군별 같음 {out['예전'][1] == out['지금'][1]}", flush=True)
