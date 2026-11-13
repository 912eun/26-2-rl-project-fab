"""공장 최종 검증에 넣을 정책을 고른다 (2026-11-13, 결과를 보기 전에 정한 규칙).

규칙: 알고리즘마다 (학습 시드 0~4 × 앞 공정 정보 유무) 10개 중 검증 점수(best_val, 시드 1 기록의 고정 구간)가
가장 낮은 하나. 작은 환경 시험(기록 시드 2·3) 결과로는 고르지 않는다.
출력: 한 줄에 "종류 정책파일 앞공정여부(up|noup)"
"""
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"
for kind, d, pat, stem in (("ppo", R / "rl_ppo", "result_*_v4_s*.json", "ppo_{}"), ("dqn", R / "rl_dqn", "result_*_v4_s*.json", "dqn_{}")):
    best = None
    for f in sorted(d.glob(pat)):
        if "smoke" in f.name:
            continue
        v = json.loads(f.read_text())["best_val"]
        tag = f.stem[len("result_"):]
        if best is None or v < best[0]:
            best = (v, tag)
    if best:
        print(kind, d / f"{stem.format(best[1])}.pt", "noup" if best[1].startswith("noup") else "up")
