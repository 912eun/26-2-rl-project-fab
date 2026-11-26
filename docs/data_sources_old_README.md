# 강화학습개론 프로젝트 (2026-2)

문제 정의 전 단계. 데이터 EDA부터 다시 시작한다.

## 폴더 구조

```
project/
├── data/
│   ├── raw/          # 원본 데이터. 읽기 전용, 절대 수정하지 않음
│   │   ├── SMT2020_HVLM/   # High-Volume / Low-Mix (제품 2종, route_3·4)
│   │   ├── SMT2020_LVHM/   # Low-Volume / High-Mix (제품 10종, route_1~10)
│   │   ├── SMAT2022/       # SMT2020 LVHM + AMHS(반송) 레이아웃 엑셀, docs/에 형식 명세
│   │   └── MiniFab/        # Intel 5-machine 6-step 원 사양(Wayback 보관본) + 교차검증용 JSON
│   └── processed/    # EDA·전처리로 만든 파생 데이터
├── notebooks/        # EDA 노트북
├── src/              # 환경·알고리즘 코드 (문제 정의 후)
├── docs/             # 문제 정의, 선행연구 정리, 회의록
│   ├── dataset_research.md   # SMT2020 배경·선행연구·관련 데이터셋
│   ├── minifab_research.md   # MiniFab 사양·선행연구·프로젝트 적합성
│   └── papers/               # 받아둔 논문 PDF
└── results/          # 그림, 표, 실험 결과
```

## 데이터 출처

- **SMT2020** (Kopp, Hassoun, Kalir & Mönch, IEEE TSM 33(4), 2020)
  - 원 배포처: https://p2schedgen.fernuni-hagen.de/index.php/downloads/simulation
  - 이 사본은 `INSIGHT/26-1-2nd-insicon/external/PySCFabSim/datasets/`에서 복사
    (PySCFabSim-release @ `77103ed`, datasets 폴더는 수정 이력 없음, 2026-11-04 체크섬 확인)

- **SMAT2022** (Lee, Song, Chang & Park, 아주대, WSC 2022 발표 예정으로 README에 표기)
  - 출처: https://github.com/kwoo-lee/smat2022 @ `a6df7fe` (2022-11-30), 2026-11-04 클론
  - SMT2020에 AMHS(자동 반송 시스템) 정보를 더한 확장판이다.
  - README는 모델 4개(HVLM, LVHM, HVLM_E, LVHM_E)라고 하지만 레포에는 `LVHM_E.xlsx` 하나만 있다.
  - 엑셀에는 **AMHS 시트만** 들어 있고 공정 데이터는 없다. 공정은 SMT2020_LVHM을 함께 써야 한다.

| 시트 | 행 수 | 내용 |
|---|---|---|
| Address | 2858 | 레일 위 좌표점 |
| Rail | 3424 | 레일 구간(주소 간 연결) |
| ZCU | 734 | 구역 제어 단위(교차로 충돌 관리) |
| Bay | 43 | 베이 |
| Equipment | 1116 | 장비 위치 |
| PortType / Port | 6 / 22120 | 장비 포트 |
| VehicleType / Vehicle | 1 / 500 | 반송 차량(OHT) |

시트별 상세 명세는 `data/raw/SMAT2022/docs/SMAT_2022_generic.docx`에 있다.

- **MiniFab** (Kempf, Intel/ASU, 1994 / 표준 인용 Spier & Kempf, ASMC 1995)
  - 데이터 파일이 아니라 텍스트 사양서다. 원 주소가 사라져 Wayback 보관본을 받았다(2026-11-04).
  - `fab-capacity-planner/`는 제3자가 사양을 JSON으로 옮긴 것(MIT, github.com/LUOaini1213/fab-capacity-planner). 교차검증용으로만 쓴다.

## 파일 형식

탭 구분 텍스트(TSV). 첫 줄이 헤더이고, 헤더가 `IGNORE`인 열은 무시한다.
시간 열은 값 + 단위 열(`min`, `hr` 등) 쌍으로 되어 있다.

```python
import pandas as pd
df = pd.read_csv("data/raw/SMT2020_HVLM/route_3.txt", sep="\t")
```

| 파일 | 내용 |
|---|---|
| `tool.txt.1l` | 장비 그룹(STNFAM)별 대수, 디스패칭 규칙, 배치 기준, 로드/언로드 시간 |
| `route_*.txt` | 제품별 공정 순서. 단계마다 장비 그룹, 처리시간 분포, 배치 최소/최대, 셋업, 리워크·샘플링 |
| `part.txt` | 제품 → route 파일 매핑 |
| `order.txt` | 신규 lot 투입 계획 (투입 간격, 납기, 우선순위, hot lot) |
| `WIP.txt` | 시뮬레이션 시작 시점의 재공(초기 lot과 현재 단계) |
| `setup.txt` / `setupgrp.txt` | 셋업 전환 시간 / 셋업 그룹별 최소 연속 처리 수 |
| `downcal.txt` / `pmcal.txt` | 고장(MTBF/MTTR) / 예방정비 캘린더 |
| `fromto.txt` | 장비 간 반송 시간 |
| `attach.txt` | 장비 그룹에 붙는 고장·PM 이벤트 연결 |

참고: PySCFabSim 로더(`simulation/read.py`)는 파일명에 `.txt`가 들어간 것을 모두 읽고,
환경변수(`NOWIP`, `NOBREAKDOWN`, `NOPM`, `NOREWORK`, `NOSAMPLING`)로 해당 요소를 뺀다.
