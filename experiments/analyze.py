"""evaluate.py 결과로 보고서용 표를 만든다 (docs/EXPERIMENT_PLAN.md의 평가 규칙).

1. 조건(파일 이름에서 시드 번호를 뺀 것, 예: dqn_noup_v4)마다, 장비군마다
   - D = lot당 (대기 + 교체), 정책 평균 D ÷ cov 평균 D − 1 (평균끼리의 비율)
   - 시드별 짝 차이: 같은 시나리오(기록 시드 × 고장 시드)의 (정책 D − cov D)를 시나리오 평균 → 시드당 1개
     → 시드들로 t 95% 구간, cov 평균 대비 %로 표시
   - 전체: 세 장비군을 lot 수로 가중
   - 굶기기 확인: 처리 못 한 측정 대상 lot 수(unstarted) 최대, |리틀 대기 − 대기| / 대기 최대
2. --compare A B: 두 조건의 시드별 짝 차이를 Welch t로 비교 (예: 앞 공정 유무)
3. --selected 파일: 고른 정책 1개의 주간 짝 차이 (정책 − cov), 4주 블록 부트스트랩 95% 구간

예: python3 experiments/analyze.py
    python3 experiments/analyze.py --compare dqn_dqn_noup_v4 dqn_dqn_up_v4 --selected dqn_dqn_noup_v4_s0
"""
import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "results" / "eval"
FAMS = ["Implant_128", "Implant_132", "Implant_91"]


def load():
    rules = json.loads((EV / "rules.json").read_text())
    runs = {}
    for f in sorted(EV.glob("*.json")):
        if f.name == "rules.json":
            continue
        runs[f.stem] = json.loads(f.read_text())
    return rules, runs


def group_of(stem):
    return re.sub(r"_s\d+$", "", stem)


def tci(x, level=.95):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(x.mean()), float("nan"), float("nan")
    h = stats.t.ppf(.5 + level / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean()), float(x.mean() - h), float(x.mean() + h)


def scen_rows(run, rules, fam=None):
    """시나리오별 (정책 D, cov D, lssu D, lot 수, 미처리, |리틀−대기|/대기, RL 결정 비율). fam=None이면 세 장비군을 lot 가중으로 합친다."""
    by = defaultdict(list)
    for key, m in run["scenarios"].items():
        f, r, ss = key.split("|")
        if fam is not None and f != fam:
            continue
        c, l = rules[f"{f}|{r}|{ss}|cov"], rules[f"{f}|{r}|{ss}|lssu"]
        by[(r, ss)].append((m["delay_h"], c["delay_h"], l["delay_h"], m["n"], m.get("unstarted", 0),
                            abs(m["little_wait_h"] - m["wait_h"]) / m["wait_h"], m.get("rl_share", float("nan"))))
    out = []
    for xs in by.values():
        n = sum(x[3] for x in xs)
        w = lambda i: sum(x[i] * x[3] for x in xs) / n  # noqa: E731
        out.append((w(0), w(1), w(2), n, max(x[4] for x in xs), max(x[5] for x in xs), w(6)))
    return out


def per_seed(run, rules, fam=None):
    """시드 1개의 값: cov 대비 = 시나리오별 짝 비율(D_RL/D_cov − 1)의 평균. lssu 대비도 같은 방식."""
    rows = scen_rows(run, rules, fam)
    return dict(ratio_cov=np.mean([r[0] / r[1] - 1 for r in rows]), ratio_lssu=np.mean([r[0] / r[2] - 1 for r in rows]),
                D=np.mean([r[0] for r in rows]), Dcov=np.mean([r[1] for r in rows]), Dlssu=np.mean([r[2] for r in rows]),
                wins=[r[0] < r[1] for r in rows], unst=max(r[4] for r in rows), little=max(r[5] for r in rows),
                rl_share=np.mean([r[6] for r in rows]))


AUX = [("p95_h", "지연95%"), ("hot_delay_h", "hot"), ("setup_ratio", "교체비율"), ("switch_per_day", "교체/일"), ("throughput_per_day", "처리량/일")]


def aux_txt(stems, runs, rules, fam):
    """보조 지표: 정책 평균 / cov 평균 (같은 시나리오). fam=None이면 생략."""
    if fam is None:
        return ""
    out = []
    for k, name in AUX:
        P, Cv = [], []
        for s_ in stems:
            for key, m in runs[s_]["scenarios"].items():
                f, r, ss = key.split("|")
                if f == fam:
                    P.append(m[k]); Cv.append(rules[f"{f}|{r}|{ss}|cov"][k])
        fmt = (lambda x: f"{100 * x:.1f}%") if k == "setup_ratio" else (lambda x: f"{x:.2f}")
        out.append(f"{name} {fmt(np.mean(P))}/{fmt(np.mean(Cv))}")
    return "      보조(정책/cov): " + ", ".join(out)


def rl_txt(vs):
    x = [v["rl_share"] for v in vs if not np.isnan(v["rl_share"])]
    return f" | RL 결정 {100 * np.mean(x):.0f}%" if x else ""


def p_improve(seed_wins, B=4000, seed=0):
    """개선 확률 P(RL < cov): (시드 × 시나리오) 짝 중 RL이 더 짧은 비율. 구간은 시드를 다시 뽑는 부트스트랩."""
    rng = random.Random(seed)
    est = np.mean([w for ws in seed_wins for w in ws])
    bs = sorted(np.mean([w for _ in seed_wins for w in rng.choice(seed_wins)]) for _ in range(B))
    return est, bs[int(.025 * B)], bs[int(.975 * B)]


def block_boot(diffs_by_scen, cov_mean, block=4, B=4000, seed=0):
    """시나리오별 주간 차이 시계열을 4주 블록으로 다시 뽑아 평균 차이의 95% 구간 (주끼리 독립이 아니어서)."""
    rng = random.Random(seed)
    blocks = []
    for xs in diffs_by_scen:
        blocks += [xs[i:i + block] for i in range(0, len(xs), block)]
    allx = [x for b in blocks for x in b]
    est = np.mean(allx)
    ms = []
    for _ in range(B):
        s = [x for _ in blocks for x in rng.choice(blocks)]
        ms.append(np.mean(s))
    ms.sort()
    return 100 * est / cov_mean, 100 * ms[int(.025 * B)] / cov_mean, 100 * ms[int(.975 * B)] / cov_mean


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--compare", nargs=2, default=None)
    p.add_argument("--selected", default=None)
    a = p.parse_args()
    rules, runs = load()
    groups = defaultdict(list)
    for stem, run in runs.items():
        groups[group_of(stem)].append(stem)
    seed_val = {}
    print("D = lot당 (대기 + 교체) 시간. cov 대비 = 시나리오별 짝 비율의 평균(시드당 1개) → 시드들로 t 95% 구간.")
    print("P(RL<cov) = (시드 × 시나리오) 짝 중 RL이 짧은 비율, 시드 부트스트랩 구간. 굶기기: 미처리 lot 최대, |리틀 대기−대기| 최대")
    for g, stems in sorted(groups.items()):
        n_sc = len(scen_rows(runs[stems[0]], rules, FAMS[0]))
        print(f"\n[{g}] 학습 시드 {len(stems)}개, 시나리오 {n_sc}개/장비군")
        for fam in FAMS + [None]:
            vs = [per_seed(runs[s_], rules, fam) for s_ in stems]
            seed_val[(g, fam or "all")] = [v["ratio_cov"] for v in vs]
            m, lo, hi = tci([v["ratio_cov"] for v in vs])
            ml = np.mean([v["ratio_lssu"] for v in vs])
            pe, plo, phi = p_improve([v["wins"] for v in vs])
            name = fam or "전체(lot 가중)"
            print(f"  {name:14s} D {np.mean([v['D'] for v in vs]):.3f}h (cov {np.mean([v['Dcov'] for v in vs]):.3f}, "
                  f"lssu {np.mean([v['Dlssu'] for v in vs]):.3f}) | cov 대비 {100 * m:+5.1f}% [{100 * lo:+5.1f}, {100 * hi:+5.1f}]"
                  f" | lssu 대비 {100 * ml:+5.1f}% | P(RL<cov) {pe:.2f} [{plo:.2f}, {phi:.2f}]"
                  f" | 미처리 {max(v['unst'] for v in vs)} 리틀차 {100 * max(v['little'] for v in vs):.1f}%"
                  + rl_txt(vs))
            if fam is not None:
                print(aux_txt(stems, runs, rules, fam))
    if a.compare:
        A, Bg = a.compare
        print(f"\n조건 비교 {A} − {Bg} (Welch t, cov 대비 %p)")
        for fam in FAMS + ["all"]:
            x, y = np.array(seed_val[(A, fam)]), np.array(seed_val[(Bg, fam)])
            t, pv = stats.ttest_ind(x, y, equal_var=False)
            print(f"  {fam:12s} {100 * (x.mean() - y.mean()):+5.1f}%p  p={pv:.3f}")
    if a.selected:
        run = runs[a.selected]
        print(f"\n고른 정책 {a.selected}: 주간 짝 차이 (정책 − cov), 4주 블록 부트스트랩 95% 구간")
        for fam in FAMS:
            diffs, covs = [], []
            for key, m in run["scenarios"].items():
                f, r, ss = key.split("|")
                if f != fam:
                    continue
                c = rules[f"{f}|{r}|{ss}|cov"]["weekly"]
                ks = sorted(set(m["weekly"]) & set(c), key=int)
                diffs.append([(m["weekly"][k] - c[k]) / 3600 for k in ks])
                covs += [c[k] / 3600 for k in ks]
            est, lo, hi = block_boot(diffs, np.mean(covs))
            print(f"  {fam:12s} {est:+5.1f}% [{lo:+5.1f}, {hi:+5.1f}]" + ("  0 포함" if lo <= 0 <= hi else ""))


if __name__ == "__main__":
    main()
