#!/bin/bash
# 공장 전체 시뮬레이터 PySCFabSim-revised(SMT2020 데이터 포함)를 external/에 받는다.
# 작은 환경(src/)과 RL 학습은 data/records의 도착 기록만 있으면 돌아가므로, 공장 실행이 필요할 때만 받으면 된다.
# 상위 레포에 라이선스 표기가 없어 코드를 이 레포에 복사하지 않고 고정 커밋으로 받는다.
set -e
cd "$(dirname "$0")"
DIR=external/PySCFabSim-revised
if [ -d "$DIR/simulation" ]; then echo "이미 있음: $DIR"; exit 0; fi
mkdir -p external
git clone https://github.com/david-dd/PySCFabSim-revised.git "$DIR"
git -C "$DIR" checkout ae3d55ef08cd0d6bc9cd114eb800e6c920dbede1
echo "완료: $DIR @ ae3d55e"
