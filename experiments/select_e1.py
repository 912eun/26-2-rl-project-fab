"""E1 기다리기 길이 고르기. 규칙은 E1 설계 문서 E1-5절 그대로 (결과를 보기 전에 정한 것).

1. 알고리즘별, 길이별 검증 점수: (a) best_val(주 기준), (b) 마지막 검증 5번 평균(확인용).
2. 알고리즘별 1등 = (a) 평균이 가장 좋은(낮은) 길이.
3. 알고리즘별 동점 묶음: 1등과 차이가 기준보다 작은 길이. 기준 = t(0.975, 자유도) × s_p × √(2/n),
   s_p = 그 알고리즘의 모든 길이 시드를 합친 공통 표준편차, 자유도 = 시드 수 합 − 길이 수, n = 길이당 시드 수.
4. 두 알고리즘의 동점 묶음이 겹치면, 겹치는 길이 중 두 알고리즘 (a) 평균의 평균이 가장 좋은 길이를 둘 다 쓴다.
5. 안 겹치면 알고리즘별로: 동점 묶음 안 고정 시간 중 가장 좋은 길이(고정 시간이 없으면 1등).
6. 확장 규칙: 1등이 후보의 끝(다음 사건까지 또는 가장 긴 길이)이고 동점 묶음에 안쪽 이웃이 없으면 바깥으로 한 단계 더 돌린다.
7. 확인: (b)로 매긴 1등이 동점 묶음 안인가, 같은 시뮬레이션 날수 지점에서 1등이 동점 묶음 안인가.

입력: results/rl_dqn/result_up_e1<길이>_s*.json, results/rl_ppo/result_up_lt_e1<길이>_s*.json  (<길이> = ev 또는 w<분>)
실행: python3 experiments/select_e1.py
"""
import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PAT = {"dqn": ("rl_dqn", r"result_up_e1(ev|w\d+)_s(\d+)\.json"), "ppo": ("rl_ppo", r"result_up_lt_e1(ev|w\d+)_s(\d+)\.json")}


def load():
    out = {}
    for alg, (d, pat) in PAT.items():
        for f in sorted((ROOT / "results" / d).glob("result_*e1*_s*.json")):
            m = re.fullmatch(pat, f.name)
            if not m:
                continue
            r = json.loads(f.read_text())
            vals = [(x.get("sim_days"), x["val"]) for x in r["log"] if "val" in x]
            out.setdefault(alg, {}).setdefault(m.group(1), []).append(
                dict(best=r["best_val"], last5=float(np.mean([v for _, v in vals[-5:]])), curve=vals))
    return out


def key(L):
    return 0.0 if L == "ev" else float(L[1:])


def at_days(curve, D):
    pts = [(d, v) for d, v in curve if d is not None]
    if not pts:
        return float("nan")
    d, v = min(pts, key=lambda x: abs(x[0] - D))
    return v


def judge(runs):
    """알고리즘 하나: 길이별 점수, 1등, 동점 묶음, 기준."""
    Ls = sorted(runs, key=key)
    groups = [[r["best"] for r in runs[L]] for L in Ls]
    ss = sum(((np.array(g) - np.mean(g)) ** 2).sum() for g in groups)
    df = sum(len(g) for g in groups) - len(groups)
    sp = np.sqrt(ss / df) if df > 0 else float("nan")
    n = np.mean([len(g) for g in groups])
    thr = stats.t.ppf(.975, max(df, 1)) * sp * np.sqrt(2 / n)
    mean = {L: float(np.mean(g)) for L, g in zip(Ls, groups)}
    best = min(Ls, key=lambda L: mean[L])
    ties = [L for L in Ls if mean[L] - mean[best] < thr]
    return dict(Ls=Ls, mean=mean, best=best, ties=ties, thr=thr, sp=sp, df=df, n=n)


def main():
    runs = load()
    algs = [a for a in ("dqn", "ppo") if a in runs]
    if not algs:
        print("E1 결과 파일이 없다")
        return
    D = min(max(d for d, _ in r["curve"] if d is not None) for a in algs for rs in runs[a].values() for r in rs)
    J = {}
    for a in algs:
        j = J[a] = judge(runs[a])
        print(f"\n[{a}] 공통 표준편차 s_p {j['sp']:.3f}, 자유도 {j['df']}, 길이당 시드 {j['n']:.0f} → 기준 {j['thr']:.3f}")
        for L in j["Ls"]:
            rs = runs[a][L]
            l5 = np.mean([r["last5"] for r in rs]); dd = np.mean([at_days(r["curve"], D) for r in rs])
            b = [r["best"] for r in rs]
            print(f"  {L:5s} 시드 {len(rs)}  (a) {j['mean'][L]:.3f} ({min(b):.3f}~{max(b):.3f})  (b) {l5:.3f}  "
                  f"같은 날수({D:.0f}일) {dd:.3f}" + ("  ← 1등" if L == j["best"] else "  ← 동점" if L in j["ties"] else ""))
        # 확장 규칙
        Ls = j["Ls"]; i = Ls.index(j["best"])
        if i in (0, len(Ls) - 1):
            inner = Ls[1] if i == 0 else Ls[-2]
            if inner not in j["ties"]:
                nxt = "더 짧은 쪽 없음(다음 사건까지가 가장 짧음)" if i == 0 else f"w{int(key(Ls[-1]) * 2)}"
                print(f"  확장 규칙: 1등이 끝({j['best']})이고 안쪽 이웃({inner})이 동점이 아님 → {nxt} 추가")
        # 확인
        b_best = min(Ls, key=lambda L: np.mean([r["last5"] for r in runs[a][L]]))
        d_best = min(Ls, key=lambda L: np.mean([at_days(r["curve"], D) for r in runs[a][L]]))
        print(f"  확인: (b) 1등 {b_best} {'(동점 묶음 안)' if b_best in j['ties'] else '(다름 → 적기)'}, "
              f"같은 날수 1등 {d_best} {'(동점 묶음 안)' if d_best in j['ties'] else '(다름 → 적기)'}")
    if len(algs) == 2:
        common = [L for L in J["dqn"]["ties"] if L in J["ppo"]["ties"]]
        if common:
            pick = min(common, key=lambda L: (J["dqn"]["mean"][L] + J["ppo"]["mean"][L]) / 2)
            print(f"\n동점 묶음이 겹침 {common} → 공통 길이 = {pick} (두 알고리즘 (a) 평균의 평균이 가장 좋음)")
        else:
            print("\n동점 묶음이 안 겹침 → 알고리즘별 길이")
            for a in algs:
                fixed = [L for L in J[a]["ties"] if L != "ev"]
                pick = min(fixed, key=lambda L: J[a]["mean"][L]) if fixed else J[a]["best"]
                print(f"  {a}: {pick}")
    print("\n고른 길이, 1등, 동점 묶음, 확인 결과를 STATUS에 적고 E1b로 넘어간다.")


if __name__ == "__main__":
    main()
