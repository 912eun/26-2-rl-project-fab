"""implant RL 상태 계산 (재생 환경과 공장 시뮬레이터가 같이 쓴다).

설계: docs/implant_sar_mapping.md "상태 설계". 두 환경은 각자 View를 만들고, 상태 벡터는 encode()만 만든다.
그래서 두 환경에서 같은 이름의 변수는 같은 계산으로 나온다.
"""
from dataclasses import dataclass, field

MIN_RUN = 7
PIECES = 25
NAMES = (["A_n", "A_setup_h"] + ["B_waited", "B_pm"] +
         [f"C_{g}_{k}" for g in ("g0", "g1", "g2") for k in ("lock", "run", "idle")] + ["C_nogas"] +
         [f"D_{g}_{k}" for g in ("g1", "g2") for k in ("n", "hot", "oldest")] +
         [f"E_{g}_arr1h" for g in ("g0", "g1", "g2")])
# 상태에는 넣지 않고 "빠진 정보" 확인에만 쓰는 값
EXTRA_NAMES = [f"X_{g}_busy_left" for g in ("g0", "g1", "g2")]
BUSY_CAP = 10.0  # 그 가스로 처리 중인 장비가 없으면 교체 시간의 10배로 둔다


@dataclass
class View:
    n: int                      # 장비군 장비 수
    setup_s: float              # 교체 시간(초)
    now: float
    gases: list                 # 장비군의 가스 이름 3개
    gas0: object                # 결정하는 장비의 현재 가스
    waited_s: float             # 결정하는 장비가 이미 기다린 시간
    pm_pieces: float            # 결정하는 장비의 PM까지 남은 매수 (여러 PM 중 최소)
    others: list                # 나머지 장비 [(가스 또는 None, 처리 중?, 잠금?, 끝나는 시각 또는 None)]
    queue: dict                 # 가스 -> (일반 lot 수, hot lot 수, 가장 오래된 도착 시각 또는 None)
    arr1h: dict                 # 가스 -> 지난 1시간 도착 수
    extra: dict = field(default_factory=dict)


def order(v):
    others = sorted((g for g in v.gases if g != v.gas0), key=lambda g: -(v.queue[g][0] + v.queue[g][1]))
    return [v.gas0] + others


def encode(v):
    n, st, t = v.n, v.setup_s, v.now
    gs = order(v)
    pos = {g: i for i, g in enumerate(gs)}
    c = [0.0] * 10
    for g, busy, locked, _ in v.others:
        if g is None or g not in pos:
            c[9] += 1
        elif busy:
            c[pos[g] * 3 + (0 if locked else 1)] += 1
        else:
            c[pos[g] * 3 + 2] += 1
    k = max(1, len(v.others))
    c = [x / k for x in c]
    d = []
    for g in gs[1:]:
        nn, nh, old = v.queue[g]
        d += [nn / n, float(nh), (t - old) / st if old is not None else 0.0]
    e = [v.arr1h.get(g, 0) / n for g in gs]
    return [n / 10, st / 3600, v.waited_s / st, v.pm_pieces / PIECES / MIN_RUN, *c, *d, *e]


def extras(v):
    gs = order(v)
    out = []
    for g in gs:
        left = [end - v.now for gg, busy, _, end in v.others if gg == g and busy and end is not None]
        out.append(min(left) / v.setup_s if left else BUSY_CAP)
    return out


# ---- 신경망 입력 변환 (2026-11-12 스케일 점검 결과, results/state_check/report_240d.txt)
# encode()는 해석 가능한 원래 값을 돌려주고 기록에도 이 값을 쓴다. 정책에는 transform(encode(v))를 넣는다.
# - B_pm: PM까지 lot 수 ÷ 7이 15~45로 다른 변수보다 훨씬 크다. 의미 있는 것은 "최소 연속 처리(7 lot) 안에
#   PM이 오나"이므로 3(=21 lot)에서 자른다.
# - D_*_n, D_*_oldest: 꼬리가 길다(95% 지점 1~3, 최대 12~17). log(1+x)로 줄인다.
import math as _math

_IDX = {n: i for i, n in enumerate(NAMES)}
_LOG = [_IDX[n] for n in NAMES if n.startswith("D_") and (n.endswith("_n") or n.endswith("_oldest"))]


def transform(x):
    y = list(x)
    y[_IDX["B_pm"]] = min(y[_IDX["B_pm"]], 3.0)
    for i in _LOG:
        y[i] = _math.log1p(y[i])
    return y
