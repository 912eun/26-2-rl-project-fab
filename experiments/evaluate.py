"""저장된 정책을 시험 시나리오(기록 시드 2·3 × 고장 시드 1~5)에서 평가해 시나리오별 지표를 저장한다.

같은 시나리오에서 lssu, cov도 한 번 돌려 results/eval/rules.json에 둔다(공통 난수: 같은 도착, 같은 고장 시드).
정책 결과는 results/eval/<종류>_<파일 이름>.json. 분석은 experiments/analyze.py.

예:
    python3 experiments/evaluate.py --kind dqn --models "results/rl_dqn/dqn_*_v4_s?.pt"
    python3 experiments/evaluate.py --kind ppo --stoch --models "results/rl_ppo/ppo_*_v4_s?.pt"
    python3 experiments/evaluate.py --kind dqn --wait_min 10 --models "results/rl_dqn/dqn_*_w10_s?.pt"
앞 공정 정보 사용 여부는 파일 이름에 noup이 있으면 끈다.
"""
import argparse
import glob
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
import rl_common as C  # noqa: E402

OUT = ROOT / "results" / "eval"


def pol_job(x):
    kind, path, use_up, fam, rec, ss, stoch, wait_min = x
    return x, C._eval_job((kind, path, use_up, fam, rec, ss, stoch, wait_min))[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["ppo", "dqn"], required=True)
    p.add_argument("--models", required=True, help="glob")
    p.add_argument("--stoch", action="store_true", help="확률 정책 그대로 (PPO)")
    p.add_argument("--wait_min", type=float, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--tag", default="", help="같은 모델을 다른 방식으로 평가할 때 결과 이름에 붙일 표시 (예: PPO argmax 평가 _argmax). 시드 번호 앞에 붙는다")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    paths = sorted(x for x in glob.glob(str(ROOT / a.models)) + glob.glob(a.models) if not x.endswith("_last.pt"))
    paths = sorted(set(paths))
    with ProcessPoolExecutor(a.workers) as ex:
        rules_path = OUT / "rules.json"
        rules = json.loads(rules_path.read_text()) if rules_path.exists() else {}
        need = [(f, r, ss, k) for f, r, ss in C.TEST for k in ("lssu", "cov") if f"{f}|{r}|{ss}|{k}" not in rules]
        for (f, r, ss, k), m in ex.map(C._rule_job, need):
            rules[f"{f}|{r}|{ss}|{k}"] = m
        rules_path.write_text(json.dumps(rules))
        for path in paths:
            use_up = "noup" not in Path(path).name
            jobs = [(a.kind, path, use_up, f, r, ss, a.stoch, a.wait_min) for f, r, ss in C.TEST]
            res = {f"{x[3]}|{x[4]}|{x[5]}": m for x, m in ex.map(pol_job, jobs)}
            stem = Path(path).stem
            if a.tag:
                import re
                m = re.match(r"(.*)(_s\d+)$", stem)
                stem = f"{m.group(1)}{a.tag}{m.group(2)}" if m else stem + a.tag
            out = OUT / f"{a.kind}_{stem}.json"
            out.write_text(json.dumps(dict(kind=a.kind, model=str(Path(path).relative_to(ROOT)) if str(path).startswith(str(ROOT)) else path,
                                           use_upstream=use_up, stoch=a.stoch, wait_min=a.wait_min, scenarios=res)))
            print("저장", out.name, flush=True)


if __name__ == "__main__":
    main()
