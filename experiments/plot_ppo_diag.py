"""PPO 디버깅 지표 그래프(10장 24쪽 점검 항목).

ppo_sb3.py가 결과 JSON의 log에 남긴 SB3 학습 지표(entropy, approx_kl, clip_fraction,
value_loss, explained_variance)와 학습 보상(mean_r)을 반복(iteration)별로 그린다.
흐린 선 = 반복별 원값, 진한 선 = 20회 이동 평균, 점선 = 검증 최고 반복(best_it).

사용: python3 experiments/plot_ppo_diag.py [tag ...]   (기본: up_lt_e1w40diag_s0 up_lt_e1w40diag_s1)
출력: results/figures/ppo_diag.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TAGS = sys.argv[1:] or ["up_lt_e1w40diag_s0", "up_lt_e1w40diag_s1"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # 범주 고정 순서
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
W = 20

plt.rcParams.update({
    "font.family": "Apple SD Gothic Neo", "axes.unicode_minus": False, "mathtext.fontset": "dejavusans",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold", "font.size": 9,
    "figure.facecolor": SURF, "axes.facecolor": SURF,
})

PANELS = [
    ("entropy", "정책 엔트로피", "행동 3개 → 최댓값 ln3 ≈ 1.10", None),
    ("approx_kl", "근사 KL (로그 축)", "한 번 갱신에 정책이 바뀐 정도", "log"),
    ("clip_fraction", "clip fraction (%)", "비율이 0.8~1.2 밖으로 잘린 표본 비율", None),
    ("value_loss", "value 손실", "", None),
    ("explained_variance", "explained variance", "1 = value 망이 수익을 완벽히 맞힘", None),
    ("mean_r", "학습 보상 (rollout 평균)", "높을수록 좋음 (R1 = -대기 비용)", None),
]


def smooth(y):
    s = np.convolve(y, np.ones(W) / W, mode="valid")
    return np.arange(W - 1, W - 1 + len(s)), s


runs = []
for t in TAGS:
    d = json.load(open(ROOT / f"results/rl_ppo/result_{t}.json"))
    runs.append((t, d))

fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
for ax, (key, title, sub, scale) in zip(axes.flat, PANELS):
    for i, (t, d) in enumerate(runs):
        rows = [r for r in d["log"] if key in r]
        x = np.array([r["it"] for r in rows])
        y = np.array([r[key] for r in rows]) * (100 if key == "clip_fraction" else 1)
        c = COLORS[i]
        if scale == "log":
            y = np.where(y > 0, y, np.nan)
        ax.plot(x, y, color=c, lw=0.8, alpha=0.25)
        xs, ys = smooth(np.nan_to_num(y, nan=np.nanmin(y)) if scale == "log" else y)
        ax.plot(x[xs], ys, color=c, lw=2, label=t.replace("up_lt_", ""))
        ax.axvline(d["best_it"], color=c, lw=1, ls=":")
    if key == "entropy":
        ax.axhline(np.log(3), color=INK2, lw=1, ls="--")
        ax.text(5, np.log(3) - 0.04, "최댓값 ln3", color=INK2, va="top")
        ax.set_ylim(0, 1.2)
    if scale:
        ax.set_yscale(scale)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0e}".replace("e-0", "e-")))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_title(title, loc="left", pad=16 if sub else 6)
    if sub:
        ax.text(0, 1.015, sub, transform=ax.transAxes, color=INK2, fontsize=8, va="bottom")
    ax.grid(True, color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("반복 (1회 = 2,048 스텝)")
axes.flat[0].legend(frameon=False, loc="center right")
fig.suptitle("PPO 학습 진단 지표 — 최종 설정(대기 40분, 학습률 일정 A, γ 0.9). 흐린 선 원값, 진한 선 20회 이동평균, 점선 검증 최고 반복",
             x=0.01, ha="left", fontsize=10, color=INK)
out = ROOT / "results/figures/ppo_diag.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=160)
print(out)
