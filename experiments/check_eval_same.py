"""E1 P1: evaluate.py 결과(results/eval)와 학습 스크립트 끝의 시험(result_*.json의 eval)이 시나리오마다 같은지 확인 (읽기만 함).
실행: python3 experiments/check_eval_same.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
import rl_common as C  # noqa: E402

KEYS = ("delay_h", "hot_delay_h", "setup_ratio")
bad, n = 0, 0
for kind, pat in (("dqn", "dqn_up_e1{t}_s{s}"), ("ppo", "ppo_up_lt_e1{t}_s{s}")):
    for t in ("ev", "w5"):
        for s in range(5):
            stem = pat.format(t=t, s=s)
            ev = json.loads((ROOT / f"results/eval/{kind}_{stem}.json").read_text())["scenarios"]
            res_name = stem.replace(f"{kind}_", "result_", 1)
            tr = json.loads((ROOT / f"results/rl_{kind}/{res_name}.json").read_text())["eval"]
            idx = {}
            for f, rec, ss in C.TEST:  # 학습 스크립트의 eval 목록은 C.TEST 순서
                i = idx.get(f, 0)
                idx[f] = i + 1
                a, b = ev[f"{f}|{rec}|{ss}"], tr[f][kind][i]
                n += 1
                diff = [k for k in KEYS if abs(a[k] - b[k]) > 1e-9]
                if diff:
                    bad += 1
                    print(f"다름: {stem} {f} 기록{rec} 고장{ss} {diff}")
print(f"시나리오 {n}개 비교, 다른 것 {bad}개")
