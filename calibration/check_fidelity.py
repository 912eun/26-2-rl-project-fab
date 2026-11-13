"""작은 환경(기록 재생)이 공장 전체 시뮬레이터를 얼마나 재현하는지 잰다 = 오차 보정의 기준표.

비교 방법
- 같은 규칙을 공장(PySCFabSim-revised, 2년, 시드 1~3)과 작은 환경(240일 도착 기록 재생, 고장 시드 1·2 평균)에 각각 넣는다.
- 공장 값은 data/fab_reference/<규칙>_730d_s<시드>/implant_info.json (2년 차 통계)에서 읽는다.
- 규칙: lssu(기본), hold15, hold30, cov. 공장 실행 당시 cov는 재방문 연결을 고치기 전 튜닝값(FAB_COV)이라 여기서도 같은 값을 쓴다.

지표
- ① 절대값: 장비군 대기(시간) 재생 − 공장 차이(%), 교체 시간 비율 차이(%p). 기준 ±15% / ±2%p.
- ② 상대값: 기본 규칙(lssu) 대비 대기 변화가 공장과 재생에서 같은 만큼인가. 기준: 순위 동일, 차이 ±5%p.
  RL 정책을 공장으로 옮길 때 가장 직접적인 근거라 ②를 우선한다.

현재 값 (2026-11-13): ① 대기 128 −7~+5%, 132 +3~+13%(시드 3이 기준에 가까움), 91 −5~+2%, 교체 시간 비율 ±0.5%p 안 → 통과.
② 9개 조합 모두 순위 같고 차이 ±1.5%p 안.
보정할 곳 후보: 재방문 되먹임(MAX_GAP 12시간), 고장 모형(MTTF/MTTR 지수분포), PM 길이, 최소 연속 처리 해석, 동시각 결정 순서.

실행: python3 calibration/check_fidelity.py [--max_gap_h 12] [--workers 8]
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import implant_env as E  # noqa: E402

FAB_COV = {"Implant_128": (5, 240), "Implant_132": (3, 240), "Implant_91": (5, 480)}  # 공장 2년 실행에 쓴 값
RULES = ["lssu", "hold15", "hold30", "cov"]
REF = ROOT / "data" / "fab_reference"


def pol(rule, fam):
    if rule == "lssu":
        return E.lssu
    if rule.startswith("hold"):
        return E.hold(int(rule[4:]) * 60)
    q, t = FAB_COV[fam]
    return E.thr(q, t * 60, cov=True)


def job(x):
    fam, rec, rule, ss, gap = x
    arr = E.load_chains(fam, rec, gap)[0]
    m = E.summarize(E.Sim(fam, arr, pol(rule, fam), seed=ss).run())
    return x, m["wait_h"], m["setup_ratio"]


def fab_values():
    out = {}
    for rule in RULES:
        for s in (1, 2, 3):
            d = json.loads((REF / f"{rule}_730d_s{s}" / "implant_info.json").read_text())
            for f, (n, w) in d["lot_waiting_at_machine"].items():
                ft = d["family_time"][f]
                # 장비 시간(setup_s)은 1년 뒤 ResetEvent로 0이 되어 2년 차 365일 누적이다
                out[(f, rule, s)] = (w / n / 3600, ft["setup_s"] / (ft["n"] * 365 * 86400))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max_gap_h", type=float, default=12, help="재방문 되먹임으로 연결할 최대 왕복 시간")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    jobs = [(f, s, r, ss, a.max_gap_h * 3600) for f in E.FAMILIES for s in (1, 2, 3) for r in RULES for ss in (1, 2)]
    rep = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for (f, s, r, ss, _), w, sr in ex.map(job, jobs):
            d = rep.setdefault((f, r, s), [0.0, 0.0])
            d[0] += w / 2
            d[1] += sr / 2
    fab = fab_values()
    mean = lambda L, i: sum(x[i] for x in L) / len(L)  # noqa: E731
    print("① 절대값 (lssu): 대기 재생−공장 %, 시드 1/2/3 | 교체 시간 비율 차이 %p")
    for f in E.FAMILIES:
        dw = [100 * (rep[(f, "lssu", s)][0] / fab[(f, "lssu", s)][0] - 1) for s in (1, 2, 3)]
        ds = [100 * (rep[(f, "lssu", s)][1] - fab[(f, "lssu", s)][1]) for s in (1, 2, 3)]
        ok = all(abs(x) <= 15 for x in dw) and all(abs(x) <= 2 for x in ds)
        print(f"  {f:12s} " + " / ".join(f"{x:+.0f}%" for x in dw) + " | " + " / ".join(f"{x:+.1f}" for x in ds)
              + ("  통과" if ok else "  불통"))
    print("\n② 기본 규칙 대비 대기 변화 (시드 1~3 평균): 공장 vs 재생")
    for f in E.FAMILIES:
        Fl = [fab[(f, "lssu", s)] for s in (1, 2, 3)]
        Rl = [rep[(f, "lssu", s)] for s in (1, 2, 3)]
        for r in RULES[1:]:
            F = [fab[(f, r, s)] for s in (1, 2, 3)]
            R = [rep[(f, r, s)] for s in (1, 2, 3)]
            cf, cr = mean(F, 0) / mean(Fl, 0) - 1, mean(R, 0) / mean(Rl, 0) - 1
            print(f"  {f:12s} {r:7s} 대기 공장 {mean(F, 0):.2f}h 재생 {mean(R, 0):.2f}h | 변화 공장 {100 * cf:+5.0f}% "
                  f"재생 {100 * cr:+5.0f}% | 차이 {100 * (cr - cf):+.1f}%p" + ("" if abs(cr - cf) <= 0.05 else "  기준 밖"))


if __name__ == "__main__":
    main()
