# MiniFab (Intel Five-Machine Six-Step Mini-Fab) 조사

작성: 2026-11-04 (웹 리서치 + 원 사양 보관본 대조)

**확인 수준.** [원문] 원문 직접 확인 · [초록] 초록/메타데이터만 · [2차] 다른 문헌 경유. 서지는 대부분 Crossref로 대조했다(일부는 응답 오류로 대조 실패, 표시함).

**로컬 자료**
- 원 사양: `data/raw/MiniFab/fabspec.html` (Wayback 보관본), `fabspec.txt` (텍스트 추출)
- 교차검증용 파라미터: `data/raw/MiniFab/fab-capacity-planner/minifab.json` (+ `sources.md`, `model.md`, MIT LICENSE). 코드는 받지 않았다.
- 논문 PDF: `docs/papers/` (van den Berk 2004, WSC 2009, WSC 2011, Stöckermann 2025, Andelfinger 2025)

---

## 1. 모델

### 출처

- **원 사양**: K. Kempf, "Intel Five-Machine Six Step Mini-Fab Description", Intel/ASU Report, 1994 (연도는 van den Berk 2004 참고문헌 기준, 문서에는 연도 없음)
  - 원 주소 `http://www.eas.asu.edu/~aar/research/intel/papers/fabspec.html`는 사라짐
  - [보관본](http://web.archive.org/web/20080719233006/http://www.eas.asu.edu/~aar/research/intel/papers/fabspec.html) [원문]
- **표준 인용**: J. Spier, K. Kempf, "Simulation of Emergent Behavior in Manufacturing Systems", IEEE/SEMI ASMC 1995, pp. 90–94, DOI 10.1109/ASMC.1995.484347 [Crossref, 본문 미확인]
  - SMT2020(WSC 2019), Tassel 2023, Stöckermann 2025, Andelfinger 2025 모두 이걸 인용
- 연도 표기가 문헌마다 다름: 1994 / 1995 / 1996 / "1990년대 중반"
- **주의**: WSC 2009는 "Mini-Fab = MIMAC 1"이라고 쓰지만, MIMAC DS1은 장비 211대 규모라 같은 모델일 수 없다.
- 2차 정리: van den Berk, "Analysis of the Intel Five-Machine Six Step Mini-Fab", TU/e 석사논문, 2004 — [PDF](https://dc.wtb.tue.nl/lefeber/do_download_pdf.php?id=74) [원문]; McGinnis, "MBSE and the Intel MiniFab Case", 2023 — [PDF](https://leonmcginnis.com/wp-content/uploads/2023/04/IntelMiniFabCaseStudy02012021.pdf) (내용 미확인)

### 구조

| 셀 | 장비 | 공정 | 특징 |
|---|---|---|---|
| C1 확산 | Ma, Mb | S1, S5 | 3 lot 병렬 배치 |
| C3 이온주입 | Mc, Md | S2, S4 | 배치·셋업 없음, 돌발 고장 |
| C2 리소 | Me (1대) | S3, S6 | 셋업 있음, **병목** |

- 경로: S1 확산 → S2 주입 → S3 리소 → S4 주입 → S5 확산 → S6 리소 (모든 제품 동일, 각 셀 2회 방문 = 재진입)
- 레이아웃: 투입 창고 – C1 – C2 – C3 – 완성 창고
- 투입(주당 lot): Pa 51, Pb 30, 테스트 웨이퍼 3, 합계 84 (교대당 약 6 lot)
- 운영: 24시간, 교대 720분 × 2

### 처리시간 [원문]

| 공정 | 장비 | load / run / unload (분) | 작업자 |
|---|---|---|---|
| S1 | Ma/Mb | 20 / 225 / 40 (배치) | PO1 |
| S2 | Mc/Md | 15 / 30 / 15 | PO1 또는 PO2 |
| S3 | Me | 10 / 55 / 10 | PO2 |
| S4 | Mc/Md | 15 / 50 / 15 | PO1 또는 PO2 |
| S5 | Ma/Mb | 20 / 255 / 40 (배치) | PO1 |
| S6 | Me | 10 / 10 / 10 | PO2 |

### 제약과 확률 요소 [원문]

- **배치 (Ma/Mb, 크기 3)**
  - S1은 제품 혼합 가능, S5는 혼합 불가
  - 테스트 lot은 배치당 1개까지
  - S1 대기 lot과 S5 대기 lot은 한 배치에 못 섞음
  - 배치 크기 2~3 선택은 사양이 "확장안"으로 제시
- **리소 셋업**: step만 바뀌면 10분, 제품만 바뀌면 5분, 둘 다 12분. 대칭, 작업자 필요
- **테스트 lot**: 같은 장비를 두 번 못 지남 (Me는 예외)
- **PM**: Ma/Mb 하루 75분, Mc/Md 교대당 120분, Me 교대당 30분. 시작 가능 창이 정해져 있음
- **돌발 고장 (Mc/Md만)**: 간격 50±26시간, 수리 420±60분. 가동 중에만 고장, 처리 중 lot은 폐기
- **인력**: 작업자 PO1(Ma·Mb·Mc·Md), PO2(Mc·Md·Me), 교대당 540분 가용. 기술자 MT1 교대당 600분. 비선점
- **반송**: 운반 장치 1대, 1 lot씩, 인접 셀 4분 + 적재·하역 각 1분
- **버퍼 한도**: C1 18, C2 12, C3 12 lot
- **사양의 용량 분석**: 확산 87.9%, 주입 77.8%, 리소 91.3%, 작업자 77.8%, 기술자 77.5%, 반송 61.7%
- **의사결정 주체**: 작업자(장비·step·lot), 기술자(EM vs PM), 운반 장치, 투입 담당, "Murphy"(고장)
- **평가지표**: 완성 lot 수·표준편차, TPT·표준편차, WIP, 가동률

### 문헌 간 불일치

- S5 처리시간: WSC 2009는 225분, 원 사양·van den Berk는 **255분**
- 고장 빈도: 사양 의도는 장비당 주 2회인데 50±26시간을 달력 시간으로 읽으면 주 약 2.95회 (`sources.md` 지적). van den Berk는 주 2회로 보정
- 배치 크기: McGinnis는 "최대 3", 원 사양·van den Berk는 "정확히 3"

---

## 2. 데이터 접근

| 자원 | 형식 | 접근 / 라이선스 | 확인 |
|---|---|---|---|
| Kempf 원 사양 | HTML 텍스트 사양 (데이터 파일 아님) | Wayback 보관본만. 라이선스 표기 없음 | 원문 |
| FernUni (p2schedgen) | WSC 2019가 배포처로 서술 | 접속 실패, 형식 미확인 | 미확인 |
| [fab-capacity-planner](https://github.com/LUOaini1213/fab-capacity-planner) | `minifab.json`, SimPy DES, 큐잉 근사, MILP | MIT, 2026-10 개인 프로젝트 | JSON이 원 사양과 일치. 코드 미검증 |
| [MiniFab_simpy](https://github.com/GinnyLin1227/MiniFab_simpy) | SimPy MiniFab + ES 학습 | 라이선스 없음, 2026-05 | 파일 목록만 |
| Stöckermann 2025 코드 | D-SIMCON (상용) | 비공개 | 원문 |
| Andelfinger 2025 코드 | D-SIMCON | 공개 언급 없음 | 미확인 |
| MiniFab Gym 환경 | — | **찾지 못함** | — |

→ 환경은 직접 구현해야 한다. 원 사양 텍스트를 1차 소스로, `minifab.json`을 교차검증용으로 쓴다.

---

## 3. 선행 연구 (시대순)

| 연도 · 서지 | 분류 | 내용·결과 | 확인 |
|---|---|---|---|
| 1996 El Adl, Rodriguez, Tsakalis, IEEE CDC, pp. 1736–1742, DOI 10.1109/CDC.1996.572810 | 제어 | ASU 계층적 제어. MiniFab 레이아웃 출처로 인용됨 | Crossref |
| 2001 Vargas-Villamil, Rivera, Computers in Industry 45(1):45–57, DOI 10.1016/S0166-3615(01)00080-X | MPC | 교대마다 LP를 푸는 MPC. Intel 라인 사례 | 초록·2차 |
| 2003 Vargas-Villamil, Rivera, Kempf, IEEE TCST 11(4):578–587, DOI 10.1109/TCST.2003.813368 | MPC | 3층 계층 MPC + 온라인 파라미터 추정 | 초록·2차 |
| 2004 van den Berk (TU/e 석사) | 분석 | 아래 상세 | 원문 |
| 2005 Ramírez-Hernández, Fernandez, IEEE CDC, pp. 2158–2163, DOI 10.1109/CDC.2005.1582481 | DP/RL | MiniFab이 아닌 **소형 재진입 라인**에서 DP 최적해 vs 시뮬레이션 기반 방법 | 2차 |
| 2007 Ramírez-Hernández, Fernandez, ADPRL pp. 201–208 / ICMLA pp. 330–335 | RL | 소형 재진입 라인에서 표 형식 Q-learning, 선형 TD. "최적 정책에 가깝게 근사" | 2차 |
| 2009 Ramírez-Hernández, Fernandez, WSC, pp. 1634–1645, DOI 10.1109/WSC.2009.5429179 | ADP | 아래 상세 | 원문 |
| 2010 Ramírez-Hernández, Fernandez, IEEE CDC, pp. 3944–3949, DOI 10.1109/CDC.2010.5717523 | PM 스케줄 ADP | MiniFab 사용 여부·수치 미확인 | Crossref |
| 2011 Akhavan-Tabatabaei, Ruiz Salazar, WSC, pp. 1971–1980, DOI 10.1109/WSC.2011.6147911 | 투입 제어 | 병목 WIP 임계값 투입. 변형 MiniFab. 무제어 대비 cycle time 9.1~39.3% 개선 (throughput −1.2~+1.5%), CONWIP 9~40% | 원문 |
| 2022 Liu et al., Complex & Intelligent Systems 8(6):4641–4662, DOI 10.1007/s40747-022-00844-0 | 딥RL | CNN-A3C로 투입 제어 + 복합 디스패칭. 기본 A3C·디스패칭 규칙보다 우수 | 초록. 수치 미확인. MiniFab 사용은 Stöckermann Table 3 기준 |
| 2023 Ma et al., Neural Comput. Appl. 35(30):22281–22296, DOI 10.1007/s00521-023-08877-3 | DDQN | 자기적응 스케줄링 | 2차. 결과 미확인 |
| 2025 Stöckermann et al., Int. J. Adv. Manuf. Technol. 139(9–10):4395–4415 ([arXiv](https://arxiv.org/abs/2505.11135)) | PPO / CMA-ES | 아래 상세 | 원문 |
| 2025 Andelfinger et al. ([arXiv 2509.15767](https://arxiv.org/abs/2509.15767)) | GNN + PPO 용량계획 | 아래 상세 | 원문 |

### 주요 결과 상세

**van den Berk (2004)**
- χ 언어로 전체 모델(작업자·기술자·반송 포함) 구현
- FIFO: 요구 84 lot/주에 throughput 47.4/주 (56.4%). WIP가 안정되는 최대 투입 57.5/주
- 병목은 리소(Me). LP 완화로는 1주 생산을 0.97주에 처리 가능
- 개선 휴리스틱으로 80.4/주 (고장 제외)

**Ramírez-Hernández & Fernandez (WSC 2009)**
- 평균비용 TD(λ) Actor-Critic + 선형 근사(기저 19개), AutoMod
- MDP: 상태 = 버퍼 12개 + 고장 2 + PM 5, **|S| ≈ 2.9×10^16**. 행동 = 스테이션별 경쟁 버퍼 선택(1↔5, 2↔4, 3↔6), 비유휴 가정으로 최대 2^3. 비용 = WIP 선형
- 10년 시뮬레이션 50~100회, 시나리오 4개
- **LBFS가 최선**, ADP는 LBFS보다 최대 약 3.6% 나쁨, LWNQ보다는 나음. FIFO·FBFS·SPT·LPT는 성능이 나빠 제외

**Stöckermann et al. (2025)**
- D-SIMCON. 납기를 RPT × FF (FF ~ U[2.1, 2.5])로 생성, 공정시간을 제품별로 수정한 변형
- 휴리스틱(정규화): SRPT 최선(완성 100, 지연 61.5), FIFO 98.1 / 100
- RL은 리소 1대만 제어
  - PPO(전체 에피소드): SRPT 대비 지연 −8.1%, throughput −0.1%
  - PPO(truncated): 지연 −2.5%
  - CMA-ES: 지연 −15.1%, throughput +0.1%
- SMT2020·산업 데이터에서는 개선이 미미

**Andelfinger et al. (2025)**
- 일 단위 장비 가동시간·효율 결정. 시드 16개
- 휴리스틱·랜덤 대비 완성 lot +1.59%, cycle time −2.70%. 무행동 대비 throughput +13.85%, cycle time −14.96%

### 공백

- **MiniFab의 배치 결정(언제, 몇 lot으로 확산로를 돌릴지)만 다룬 연구는 못 찾음.** 기존 연구는 배치 크기 3 고정. 가장 가까운 건 WSC 2009의 S1↔S5 버퍼 선택.
- Dai & Gluzman 2022(Stochastic Systems 12(1):30–67)와 Kumar 계열 재진입 라인 연구에서 MiniFab을 썼다는 근거는 못 찾음.

---

## 4. 우리 프로젝트에 맞는가 (5주, 3명, VI 정답지 + Q-learning/DQN/PPO)

### 원본 그대로는 VI 불가

- 상태: 버퍼만 약 2.9×10^16 (WSC 2009). 인력·반송·셋업·잔여 처리시간까지 넣으면 더 큼
- 행동: 결정 시점마다 작고 이산적 (최대 2^3, 배치 구성 선택을 넣으면 조금 늘어남)
- → DQN/PPO는 가능, VI는 불가

### VI가 가능한 축소판 (조사 측 추정, 공개 정답 없음)

- 제거: 작업자, 기술자, 반송, PM 창
- 이산시간 + 기하분포 서비스 (또는 균일화한 CTMC), 제품 1~2종, 버퍼 상한 K, 배치 크기 2~3, Mc만 고장 on/off
- 크기: 버퍼 6개 × K=3 → 4^6 = 4,096 / 장비 5대 busy → 2^5 / 고장 → 2^2 ⇒ **약 52만 상태**. 희소 전이로 VI 가능. K=2면 약 9.3만
- 행동: 스테이션별 버퍼 선택 2^3 + 확산로 "지금 시작 / 대기"
- 근거: Ramírez-Hernández & Fernandez 2005·2007이 소형 재진입 라인에서 DP 최적해 vs Q-learning 비교 (2차)

### 비교 기준

- 원본: van den Berk FIFO (47.4/주, 최대 57.5/주) + 사양 용량 분석
- 버퍼 선택 MDP: WSC 2009의 LBFS, LWNQ, ADP (WIP·cycle time 95% CI)
- 납기 변형: Stöckermann 2025의 SRPT·CR·EDD·FIFO·SPT와 PPO·ES 개선율
- **주의**: 연구마다 변형(S5 시간, 고장 해석, 납기 생성)이 달라 수치를 직접 비교할 수 없다. 우리 구현에서 FIFO·LBFS를 다시 돌려 기준선을 만들어야 한다.

### 권고 (조사 측 의견)

1. 원 사양을 파라미터 1차 소스로, `minifab.json`은 교차검증용으로
2. 2단 구성
   - 1단: 축소판 MDP에서 VI 정답 → Q-learning·DQN 수렴 확인
   - 2단: 원본에 가까운 이산사건 시뮬레이션에서 DQN/PPO vs LBFS·SRPT·FIFO
3. 확산로 배치 결정("지금 시작 / 대기", 배치 크기 2~3)을 핵심 행동으로 두면 기존 연구와 겹치지 않음. 배치 크기 선택은 사양이 직접 확장안으로 제시

## 미확인 항목

- Spier & Kempf 1995 본문, McGinnis 2023 내용
- FernUni MiniFab 배포 형식
- Mönch·Fowler·Mason 교재 수치표
- Liu 2022, Ma 2023 정량 결과
- Dai & Gluzman, Kumar 계열의 MiniFab 사용 여부
- GitHub 개인 구현 2종 품질
- 일부 Crossref 대조
