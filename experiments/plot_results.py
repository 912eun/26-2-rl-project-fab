"""보고서용 결과 그림 (E3 시험, E6 공장 검증). 출력: results/figures/*.png

  e3_test_vs_cov.png   E3 시험 cov 대비 (장비군별·전체), DQN·PPO 시드별 점 + 평균과 t 95% 구간
  e6_implant_wait.png  E6 공장 implant 대기 cov 대비 (장비군 하나씩 / 세 장비군 동시), 공장 시드별 점 + 평균
  e6_ontime_vs_ct.png  E6 27회 실행의 공장 평균 CT와 납기 준수율 (준수율이 CT를 따라 움직임을 보임)

사용: python3 experiments/plot_results.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.argv = sys.argv[:1]
import analyze as A  # noqa: E402
import fab_report as F  # noqa: E402

OUT = ROOT / "results/figures"
OUT.mkdir(parents=True, exist_ok=True)
C = {"DQN": "#2a78d6", "PPO": "#eb6834", "cov": "#8a8984"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "font.family": "Apple SD Gothic Neo", "axes.unicode_minus": False,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold", "font.size": 9,
    "figure.facecolor": SURF, "axes.facecolor": SURF,
})
FAMS = ["Implant_128", "Implant_132", "Implant_91"]
FL = {"Implant_128": "128", "Implant_132": "132", "Implant_91": "91", None: "전체"}


def style(ax):
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color=INK2, lw=1)


def tci(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), h


def dots(ax, xc, vals, color, label=None, ci=True):
    vals = np.asarray(vals, float)
    jit = np.linspace(-0.06, 0.06, len(vals)) if len(vals) > 1 else [0]
    ax.scatter(xc + jit, vals, s=22, color=color, alpha=0.45, lw=0, zorder=2)
    m = vals.mean()
    if ci and len(vals) > 2:
        _, h = tci(vals)
        ax.errorbar(xc, m, yerr=h, fmt="none", ecolor=color, elinewidth=2, capsize=4, zorder=3)
    ax.scatter([xc], [m], s=70, marker="D", color=color, edgecolor=SURF, lw=1.5, zorder=4, label=label)
    ax.annotate(f"{m:+.1f}%", (xc + 0.13, m), va="center", fontsize=8, color=INK)


# 1. E3 시험
def fig_e3():
    rules, runs = A.load()
    groups = {"DQN": "dqn_dqn_up_e3", "PPO": "ppo_ppo_up_lt_e3"}
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for j, fam in enumerate(FAMS + [None]):
        for k, (alg, g) in enumerate(groups.items()):
            stems = sorted(s for s in runs if A.group_of(s) == g)
            v = [100 * A.per_seed(runs[s], rules, fam)["ratio_cov"] for s in stems]
            dots(ax, j + (k - 0.5) * 0.4, v, C[alg], label=f"{alg} (시드 {len(v)}개)" if j == 0 else None)
    ax.set_xticks(range(4), [f"{FL[f]}" for f in FAMS + [None]])
    ax.axvline(2.5, color=GRID, lw=1)
    style(ax)
    ax.set_ylabel("lot당 지연, cov 대비 (%)  ← 좋음")
    ax.set_title("E3 시험: 시나리오 10개/장비군, 학습 시드별 점 · 평균(◆)과 t 95% 구간", loc="left")
    ax.legend(frameon=False, loc="upper left", ncol=2)
    fig.savefig(OUT / "e3_test_vs_cov.png", dpi=160)
    plt.close(fig)


# 2. E6 공장 implant 대기
E6 = {"DQN": "rl_dqn_up_e3_s11", "PPO": "rl_ppo_up_lt_e3_s12"}
SEEDS = [4, 5, 6]


def fab(name):
    return {s: F.one(name, s) for s in SEEDS}


def fig_e6_wait():
    cov = fab("covfb")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True, constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[3, 4]))
    for k, (alg, base) in enumerate(E6.items()):
        off = (k - 0.5) * 0.4
        for j, fam in enumerate(FAMS):
            r = fab(f"{base}_only{FL[fam]}")
            v = [100 * (r[s][f"wait_{fam}"] / cov[s][f"wait_{fam}"] - 1) for s in SEEDS]
            dots(axes[0], j + off, v, C[alg], label=alg if j == 0 else None, ci=False)
        r = fab(base)
        for j, key in enumerate([f"wait_{f}" for f in FAMS] + ["wait_all"]):
            v = [100 * (r[s][key] / cov[s][key] - 1) for s in SEEDS]
            dots(axes[1], j + off, v, C[alg], ci=False)
    axes[0].set_xticks(range(3), [f"{FL[f]}에만 넣음" for f in FAMS])
    axes[1].set_xticks(range(4), ["128", "132", "91", "전체"])
    axes[1].axvline(2.5, color=GRID, lw=1)
    axes[0].set_title("주 결과: 한 장비군에만 RL → 그 장비군 대기", loc="left")
    axes[1].set_title("보조: 세 장비군 동시에 RL", loc="left")
    for ax in axes:
        style(ax)
    axes[0].set_ylabel("implant 장비 앞 대기, cov 대비 (%)  ← 좋음")
    axes[0].legend(frameon=False, loc="upper left", ncol=2)
    fig.suptitle("E6 공장 검증 (SMT2020 HVLM, 730일 중 2년째): 공장 시드 4·5·6 점, ◆ 평균. 시드 3개라 방향만 본다",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.savefig(OUT / "e6_implant_wait.png", dpi=160)
    plt.close(fig)


# 3. E6 납기 준수율 vs CT
def fig_e6_ontime():
    runs = {"cov": ["covfb"]}
    for alg, base in E6.items():
        runs[alg] = [base] + [f"{base}_only{FL[f]}" for f in FAMS]
    fig, ax = plt.subplots(figsize=(7, 4.8), constrained_layout=True)
    X, Y = [], []
    for alg, names in runs.items():
        pts = [(F.one(n, s), n, s) for n in names for s in SEEDS]
        x = [p[0]["ct_regular_d"] for p in pts]
        y = [p[0]["ontime_regular"] for p in pts]
        X += x; Y += y
        ax.scatter(x, y, s=40 if alg == "cov" else 28, color=C[alg], alpha=0.85, edgecolor=SURF, lw=1,
                   marker="s" if alg == "cov" else "o", label=f"{alg} ({len(x)}회)", zorder=3)
        for r, n, s in pts:
            if n == E6["PPO"] and s == 4:
                ax.annotate("PPO 동시 · 시드 4", (r["ct_regular_d"], r["ontime_regular"]), xytext=(-8, 0), ha="right", va="center",
                            textcoords="offset points", fontsize=8, color=INK)
    k = np.polyfit(X, Y, 1)
    xs = np.linspace(min(X), max(X), 50)
    rr = np.corrcoef(X, Y)[0, 1]
    ax.plot(xs, np.polyval(k, xs), color=INK2, lw=1, ls="--", zorder=1)
    ax.text(0.98, 0.95, f"상관 r = {rr:.2f}\nCT 1일 늘면 준수율 {k[0]:.1f}%p", transform=ax.transAxes,
            ha="right", va="top", color=INK2)
    ax.grid(True, color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("공장 평균 CT, 일반 lot (일)")
    ax.set_ylabel("납기 준수율, 일반 lot (%)")
    ax.set_title("E6: 납기 준수율은 공장 전체 CT를 따라 움직인다 (27회)", loc="left")
    ax.legend(frameon=False, loc="lower left")
    fig.savefig(OUT / "e6_ontime_vs_ct.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    fig_e3()
    fig_e6_wait()
    fig_e6_ontime()
    print("\n".join(str(p) for p in sorted(OUT.glob("*.png"))))
