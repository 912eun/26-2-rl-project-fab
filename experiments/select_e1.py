"""E1 기다리기 길이 고르기 (docs/EXPERIMENT_PLAN.md 7-1, 7-2절을 그대로 계산).

입력: results/rl_dqn/result_up_e1<길이>_s*.json, results/rl_ppo/result_up_lt_e1<길이>_s*.json
      (<길이> = ev, w5, w10, w20, w40 등)
출력: 길이별 검증 점수 표((a) 최고값, (b) 마지막 5번 평균, 같은 날수 지점 값), 공통 표준편차, 기준, 고른 길이와 확인 결과.
--sd_env: val_noise.py로 잰 고장 난수 잡음 sd_env (7-1 (가)). 주면 (가)(나) 중 큰 기준을 쓴다.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PAT = {"dqn": ("rl_dqn", r"result_up_e1(\w+?)_s(\d+)\.json"), "ppo": ("rl_ppo", r"result_up_lt_e1(\w+?)_s(\d+)\.json")}


def load():
    out = {}  # (alg, length) -> [run dict]
    for alg, (d, pat) in PAT.items():
        for f in sorted((ROOT / "results" / d).glob("result_*e1*_s*.json")):
            m = re.fullmatch(pat, f.name)
            if not m:
                continue
            r = json.loads(f.read_text())
            vals = [(x.get("sim_days"), x["val"]) for x in r["log"] if "val" in x]
            out.setdefault((alg, m.group(1)), []).append(dict(best=r["best_val"], last5=float(np.mean([v for _, v in vals[-5:]])), curve=vals))
    return out


def at_days(curve, D):
    """학습 곡선에서 sim_days가 D 이하인 마지막 검증 점수."""
    xs = [v for d, v in curve if d is not None and d <= D]
    return xs[-1] if xs else float("nan")


def pooled_sd(groups):
    ss = sum(((np.array(g) - np.mean(g)) ** 2).sum() for g in groups if len(g) > 1)
    df = sum(len(g) - 1 for g in groups if len(g) > 1)
    return (np.sqrt(ss / df) if df > 0 else float("nan")), df


def length_key(L):
    return (0, 0) if L == "ev" else (1, float(L.lstrip("w")))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sd_env", type=float, default=None)
    a = p.parse_args()
    runs = load()
    lengths = sorted({L for _, L in runs}, key=length_key)
    algs = [x for x in ("dqn", "ppo") if any(k[0] == x for k in runs)]
    D = min(max(d for d, _ in r["curve"] if d is not None) for rs in runs.values() for r in rs)  # 모두가 도달한 최대 날수
    print(f"길이 {lengths}, 알고리즘 {algs}, 같은 날수 비교 지점 = {D:.0f}일")
    score = {}
    se2 = 0.0
    for alg in algs:
        sp, df = pooled_sd([[r["best"] for r in runs.get((alg, L), [])] for L in lengths])
        n = np.mean([len(runs.get((alg, L), [])) for L in lengths])
        se = sp * np.sqrt(2 / n)
        se2 += se ** 2
        print(f"\n[{alg}] 공통 표준편차 s_p = {sp:.3f} (자유도 {df}), 두 길이 차이 SE = {se:.3f}")
        for L in lengths:
            rs = runs.get((alg, L), [])
            b = np.mean([r["best"] for r in rs]); l5 = np.mean([r["last5"] for r in rs]); dd = np.mean([at_days(r["curve"], D) for r in rs])
            score.setdefault(L, {})[alg] = (b, l5, dd)
            print(f"  {L:5s} 시드 {len(rs)}  (a) 최고 {b:.3f}  (b) 마지막5 {l5:.3f}  같은 날수 {dd:.3f}")
    se_comb = 0.5 * np.sqrt(se2) if len(algs) == 2 else np.sqrt(se2)
    df_all = sum(pooled_sd([[r["best"] for r in runs.get((alg, L), [])] for L in lengths])[1] for alg in algs)
    thr = stats.t.ppf(.975, max(df_all, 1)) * se_comb
    if a.sd_env is not None:
        thr = max(thr, np.sqrt(2) * a.sd_env)
    comb = {L: [np.mean([score[L][alg][i] for alg in algs if alg in score[L]]) for i in range(3)] for L in lengths}
    best = min(lengths, key=lambda L: comb[L][0])
    ties = [L for L in lengths if comb[L][0] - comb[best][0] < thr]
    fixed = [L for L in ties if L != "ev"]
    pick = min(fixed, key=lambda L: comb[L][0]) if fixed else best
    print(f"\n두 알고리즘 합친 점수 (a)/(b)/같은 날수:")
    for L in lengths:
        print(f"  {L:5s} {comb[L][0]:.3f} / {comb[L][1]:.3f} / {comb[L][2]:.3f}" + ("  ← 동점 묶음" if L in ties else ""))
    print(f"\n기준 = {thr:.3f} (t × ½√(SE_DQN²+SE_PPO²)" + (", sd_env와 비교해 큰 쪽" if a.sd_env else "") + ")")
    print(f"1등(a) = {best}, 동점 묶음 = {ties} → 고른 길이 = {pick}")
    b_best = min(lengths, key=lambda L: comb[L][1]); d_best = min(lengths, key=lambda L: comb[L][2])
    print(f"확인: (b) 1등 = {b_best} ({'동점 묶음 안' if b_best in ties else '다름 → 적기'}), "
          f"같은 날수 1등 = {d_best} ({'동점 묶음 안' if d_best in ties else '다름 → 적기'})")


if __name__ == "__main__":
    main()
