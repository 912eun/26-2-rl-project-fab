"""E4 학습률·E1b γ 판정 (안내서 5절 진행 순서, 7-1 (나) 기준). 규칙은 결과를 보기 전에 정한 것.

규칙: 기본 설정(--base)을 유지한다. 후보(--cands) 중 검증 점수 평균이 가장 좋은 것이 기본보다
      기준 이상 좋을 때만 그 후보로 바꾼다.
  - 검증 점수 = 시드별 학습 중 가장 좋았던 검증 값(낮을수록 좋음, 시드 7 기록).
    조건 뒤에 ':es'를 붙이면 1000회 곡선에 PPO 조기 종료 규칙(600회 이후, 15번 연속 0.01 이상 개선 없음)을
    적용했을 때 멈췄을 시점까지의 가장 좋은 값을 쓴다(E4 PPO 학습률 A·B, `--patience 0`으로 끝까지 돌린 실행).
  - 기준 = t(0.975, 자유도) × s_p × √(1/n_기본 + 1/n_후보). s_p = 이 비교에 든 모든 조건의 시드를 합친 공통 표준편차,
    자유도 = 시드 수 합 − 조건 수.
  - γ(E1b)는 "동점이면 0.9 유지"라서 기본 = γ 0.9 조건으로 주면 같은 규칙이다.

사용 예:
  python3 experiments/judge.py --alg ppo --base e4lrA:es --cands e4lrB:es e4lrC      # E4 PPO 학습률
  python3 experiments/judge.py --alg dqn --base e1w40 --cands e4lrdec                # E4 DQN 학습률
  python3 experiments/judge.py --alg dqn --base e1w40 --cands e1bg08 e1bg095         # E1b γ (학습률 그대로일 때)
태그는 학습 때 준 --suffix에서 앞의 '_'를 뺀 것. 파일: results/rl_dqn/result_up_<태그>_s*.json,
results/rl_ppo/result_up_lt_<태그>_s*.json
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
ES = dict(min_iters=600, patience=15, min_delta=0.01)  # 안내서 3-3 PPO 조기 종료


def es_best(log):
    """PPO 조기 종료를 1000회 곡선에 그대로 적용: (멈춘 iter, 그때까지 가장 좋은 검증 값). ppo_sb3.TrainCallback과 같은 논리."""
    best, no_imp = float("inf"), 0
    for row in log:
        if row.get("val") is None:
            continue
        v = row["val"]
        no_imp = 0 if v < best - ES["min_delta"] else no_imp + 1
        best = min(best, v)
        if row["it"] + 1 >= ES["min_iters"] and no_imp >= ES["patience"]:
            return row["it"] + 1, best
    return None, best


def load(alg, cond):
    tag, _, rule = cond.partition(":")
    d, pre = ("rl_dqn", "result_up") if alg == "dqn" else ("rl_ppo", "result_up_lt")
    rows = []
    for f in sorted((ROOT / "results" / d).glob(f"{pre}_{tag}_s*.json")):
        m = re.fullmatch(rf"{pre}_{re.escape(tag)}_s(\d+)\.json", f.name)
        if not m:
            continue
        r = json.loads(f.read_text())
        if rule == "es":
            stop, v = es_best(r["log"])
            rows.append((int(m.group(1)), v, f"멈춤 {stop or '끝까지'}"))
        else:
            rows.append((int(m.group(1)), r["best_val"], ""))
    if not rows:
        raise SystemExit(f"결과 파일 없음: results/{d}/{pre}_{tag}_s*.json")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alg", choices=["dqn", "ppo"], required=True)
    p.add_argument("--base", required=True, help="기본 설정 태그 (뒤에 :es 가능)")
    p.add_argument("--cands", nargs="+", required=True, help="후보 태그들 (뒤에 :es 가능)")
    a = p.parse_args()

    conds = [a.base] + a.cands
    data = {c: load(a.alg, c) for c in conds}
    vals = {c: np.array([v for _, v, _ in data[c]]) for c in conds}
    N, k = sum(len(v) for v in vals.values()), len(conds)
    df = N - k
    if df < 1:
        raise SystemExit("시드가 너무 적어 공통 표준편차를 낼 수 없다")
    sp = float(np.sqrt(sum(((v - v.mean()) ** 2).sum() for v in vals.values()) / df))
    t = float(stats.t.ppf(0.975, df))

    print(f"[{a.alg}] 공통 표준편차 s_p {sp:.4f}, 자유도 {df}, t {t:.3f}")
    for c in conds:
        v = vals[c]
        seeds = " ".join(f"s{s}:{x:.3f}" + (f"({note})" if note else "") for s, x, note in data[c])
        print(f"  {'기본' if c == a.base else '후보'} {c:14s} 시드 {len(v)}  평균 {v.mean():.3f} ({v.min():.3f}~{v.max():.3f})  [{seeds}]")

    b = vals[a.base]
    best_c = min(a.cands, key=lambda c: vals[c].mean())
    for c in a.cands:
        thr = t * sp * np.sqrt(1 / len(b) + 1 / len(vals[c]))
        diff = b.mean() - vals[c].mean()
        print(f"  {c}: 기본보다 {diff:+.3f} 좋음 (기준 {thr:.3f}) → {'기준 이상' if diff >= thr else '기준 미만'}")
    thr = t * sp * np.sqrt(1 / len(b) + 1 / len(vals[best_c]))
    diff = b.mean() - vals[best_c].mean()
    if diff >= thr:
        print(f"\n판정: {best_c}로 바꾼다 (후보 중 평균이 가장 좋고, 기본보다 {diff:.3f} ≥ 기준 {thr:.3f})")
    else:
        print(f"\n판정: 기본 {a.base} 유지 (가장 좋은 후보 {best_c}도 차이 {diff:+.3f} < 기준 {thr:.3f})")
    print("판정, 조건별 평균(최소~최대), 기준을 STATUS에 적는다.")


if __name__ == "__main__":
    main()
