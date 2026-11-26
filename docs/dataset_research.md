# SMT2020 배경 · 선행연구 · 관련 데이터셋 조사

작성: 2026-11-04 (웹 리서치 + 로컬 데이터 대조)

**출처 표기.** [원문] PDF 본문 확인 · [초록] 초록/저장소 README만 확인 · [2차] 다른 문헌이나 검색 요약의 인용. 서지(권호·쪽·DOI)는 표시가 없으면 Crossref로 대조했다.

**한계.** SMT2020 저널 원문(IEEE TSM)은 유료라 읽지 못했다. FernUni Hagen 배포 페이지는 조사 시점에 접속되지 않았다.

---

## 0. 먼저 알아둘 점: 공식 수치가 출처마다 다르다

| 항목 | 문헌 값 | **로컬 파일 실측** |
|---|---|---|
| HV/LM 장비 수 | 1,054대, 105 toolset (WSC 2019, *예비판*) / 1,071 (Tassel 2023) / 1,050 (synthetic-fab-simulator README) | tool group 106개, `STNQTY` 합 1,443 |
| HV/LM route 길이 | 632 / 374 step (WSC 2019) / 583 / 343 (synthetic-fab-simulator) | **583 / 343** |
| LV/HM 장비 수 | 913대, 105 tool group (Kopp WSC 2020, Andelfinger WSC 2025) / 1,265대, 106 (Tassel 2023) | tool group 106개, `STNQTY` 합 1,313 |
| LV/HM route 길이 | 242~583 step (WSC 2020) | **242~583** (route_1~10) |
| LV/HM_E 장비 수 | 1,068 (SMAT2022 논문) | SMAT2022 Equipment 시트 1,116행 |
| 초기 WIP | LV/HM 2,700 lot, HV/LM 3,635 lot (Tassel 2023) | (EDA에서 확인) |
| 투입량 | HV/LM 주 5,100 wafer, hot lot 2.5% (WSC 2019) / LV/HM 제품당 주 1,000 wafer, 병목 목표 가동률 약 97% (WSC 2020) | (EDA에서 확인) |

- 로컬 route 길이는 WSC 2019 예비판이 아니라 최종판(TSM) 쪽과 맞는다.
- 장비 대수는 어느 문헌과도 맞지 않는다. `STNQTY`가 실제 대수인지, 다른 의미(예: 동시 처리 수)가 섞였는지 EDA에서 확인해야 한다.
- **보고서에는 로컬 파일에서 센 값을 쓰고, 문헌 값과 다르다는 점을 함께 적는다.**

---

## 1. 배경: SMT2020은 왜 만들어졌나

### 문헌 계보

1. **Hassoun & Kalir (WSC 2017)** "Towards a New Simulation Testbed for Semiconductor Manufacturing", pp. 3612–3623, DOI 10.1109/WSC.2017.8248074 [원문] — [PDF](https://www.informs-sim.org/wsc17papers/includes/files/302.pdf)
   - 새 테스트베드의 요구사항 제안. MIMAC 이후 22년간 팹에서 바뀐 점을 정리했다.
     - 25 wafer lot 표준화, 자동 반송, 낱장 공정
     - lot-to-lens dedication, 동적 qualification, CQT 증가
     - metrology가 병목으로 바뀜, engineering lot과 hot lot
   - 팹 구성 두 가지: LV/HM(약 10개 제품, 주문생산, 납기·CR 규칙) / HV/LM(제품 2종, 재고생산, FCFS)
2. **Hassoun, Kopp, Mönch, Kalir (WSC 2019)** "A New High-Volume/Low-Mix Simulation Testbed…", pp. 2419–2428, DOI 10.1109/WSC40007.2019.9004654 [원문] — [PDF](https://www.informs-sim.org/wsc19papers/232.pdf)
   - dataset 1(HV/LM)을 먼저 공개. AutoSched AP로 구현.
3. **Kopp, Hassoun, Kalir, Mönch (IEEE TSM 2020)** "SMT2020—A Semiconductor Manufacturing Testbed", 33(4):522–531, DOI 10.1109/TSM.2020.3001933 [초록/2차]
   - 모델 4개: 기반 HV/LM, LV/HM + engineering lot을 넣은 확장판 2개(HVLM_E, LVHM_E = dataset 3, 4)
   - production lot과 engineering lot 디스패칭 사례연구 포함
4. **Kopp et al. (WSC 2020)** "Integrating Critical Queue Time Constraints into SMT2020 Simulation Models", pp. 1813–1824, DOI 10.1109/WSC48552.2020.9383889 [원문] — [PDF](https://informs-sim.org/wsc20papers/194.pdf)
   - dataset 2(LV/HM)에 CQT 고려 디스패칭 2종과 lot stopping 전략. 제품당 CQT 구간 최대 42개.

**저자 소속** (WSC 2019): Hassoun – Ariel Univ. / Kopp, Mönch – FernUni Hagen / Kalir – Intel, Ben-Gurion Univ.

**배포처**: http://p2schedgen.fernuni-hagen.de/index.php?id=296 (WSC 2019), https://p2schedgen.fernuni-hagen.de/index.php/downloads/simulation (SMAT2022 README). Excel과 Instructions.pdf로 배포. 두 주소 모두 조사 시점에 접속 안 됨.

### MIMAC과의 관계와 한계

- **MIMAC**: Fowler & Robinson (1995), SEMATECH TR 95062861A-TR. 모델 수는 문헌마다 6개(WSC 2019) 또는 7개(WSC 2017).
  - 규모: 장비 최대 260대, workcenter 최대 85, 제품 최대 21종, step 최대 280(본문은 340) [WSC 2019 Table 1]
- **WSC 2019 Table 1의 다른 기존 모델**: MiniFab(장비 5, 제품 2, 6 step), Harris(장비 12, 제품 3, 최대 22 step), SEMATECH 300mm(장비 275, 제품 1, 364 step). SEMATECH 300mm을 빼고 FernUni에서 공개.
- **MIMAC의 한계** (WSC 2017):
  - PM과 고장을 구분하지 않고 공통 MTBF 하나로 다룸
  - 시퀀스 의존 셋업이 제한적, metrology 거의 없음, litho 층 10~20개뿐
  - CQT, dedication, engineering lot 없음
  - load/unload에 사람 작업자 전제
  - → 현대 팹의 복잡도가 부족함

### SMT2020이 새로 넣은 요소 (WSC 2019, HV/LM 기준)

- **규모**: route 500 step 이상, tool group 100개 이상, 공정 영역 11개
- **가용성**: PM(SDT)과 고장(UDT) 분리
  - PM: diffusion·litho·metrology는 시간 기준(월간·분기, litho는 주간 추가), 나머지는 카운터 기준
  - 고장: TBF·TTR 지수분포, MTBF 1주
  - litho는 다운타임의 50%가 고장, metrology·planar는 90%가 PM
- **처리시간**: 명목값 ±5% 균등분포. wafer/lot/batch 단위 + load/unload
- **반송**: 5~10분 균등분포 지연으로만 표현 (`fromto.txt`)
- **배치(확산)**: 최대 100~150 wafer. 1 lot 모자라도 시작 허용(최대 5 lot이면 4 lot으로 시작 가능)
- **캐스케이딩**: wet etch, implant, planar, thin film 전체 + dielectric 약 절반
- **셋업**: litho reticle 교체 5~20분 / dry etch 2대 2그룹 / implant 가스별 3그룹 (시퀀스 의존)
- **샘플링**: litho metrology 약 절반 100%·나머지 10~50%, thin film 30~40%, defect 전공정 50~60%·후공정 10%
- **리워크**: litho 공정 약 1/3에 경로, 비율 0.5~2%. scrap 없음
- **LTLD**: stepper 3그룹. **CQT**: 정의만 있고 진입 제한 없음
- **디스패칭 계층** (Tassel 2023 정리): CQT → 우선순위(super hot > hot > regular) → 셋업 회피 → tie-break(HV/LM FCFS, LV/HM CR)

---

## 2. 선행 연구 (SMT2020 사용)

### (a) RL · 디스패칭

| 서지 | 요약 | 링크 | 확인 |
|---|---|---|---|
| Kovács et al., ASMC 2022, pp. 1–6, DOI 10.1109/ASMC54647.2022.9792520 | 순수 Python 시뮬레이터 PySCFabSim. 상용 시뮬레이터와 KPI 일치 검증 | [슬라이드](https://prosysscience.github.io/PySCFabSim-release/assets/img/portfolio/asmc_presentation.pdf) | Crossref, 2차 |
| Kovács, Tassel, Gebser, Seidel, WSC 2022, pp. 2663–2674, DOI 10.1109/WSC57314.2022.10015524 | PySCFabSim RL 환경 (아래 상세) | [PDF](https://www.informs-sim.org/wsc22papers/270.pdf), [GitHub](https://github.com/prosysscience/PySCFabSim-release) | 원문 |
| Tassel et al., WSC 2023, pp. 1924–1935, DOI 10.1109/WSC60868.2023.10407747 | 전역 디스패처를 자기지도 사전학습 + ES로 학습. 6개월·시드 100개 평가. 비용 HV/LM −16.68%, LV/HM −7.66% (SMT2020 휴리스틱 대비) | [arXiv 2302.07162](https://arxiv.org/abs/2302.07162) | 원문 |
| Stöckermann et al., WSC 2023, pp. 3047–3058, DOI 10.1109/WSC60868.2023.10408625 | Infineon 실제 팹 디지털 트윈, Flow Factor 최소화(self-attention + CMA-ES). **SMT2020 미사용**, 한계만 지적 | [PDF](https://www.informs-sim.org/wsc23papers/doc102.pdf) | 원문 |
| Stöckermann et al., Int. J. Adv. Manuf. Technol. 139(9–10):4395–4415, 2025 | Minifab, SMT2020, 실데이터로 RL 확장성 비교 | [arXiv 2505.11135](https://arxiv.org/abs/2505.11135) | 초록, 2차 |
| Immordino et al., J. Intell. Manuf. 37(5):1999–2015 | RL 디스패칭 정책의 설명가능성 | DOI 10.1007/s10845-025-02631-3 | Crossref만. SMT2020 사용 여부 미확인 |
| Andelfinger et al., arXiv 2509.15767 (2025) | 이종 GNN + Deep RL 용량계획. Minifab + SMT2020 예비 실험, throughput·cycle time 각 약 1.8% 개선 | [arXiv](https://arxiv.org/abs/2509.15767) | 초록 |

**Kovács et al. WSC 2022 상세**
- Gym 인터페이스, 단일·다중 에이전트 지원. RL이 맡지 않은 장비군은 CR 그리디가 처리.
- 행동 공간 4종: 장비에 넣을 공정 선택 / lot을 장비 큐에 배정 / lot 선택 / 휴리스틱 선택(FIFO, CR, 셋업 회피, most-delay)
- 보상: dense(납기, WIP, 가동률) + sparse(lot 완료, coupling 위반) 조합
- 자동 다운스케일. 2년 시뮬레이션 약 20분, 메모리 약 200MB
- **PPO로 Diffusion 장비군을 제어했지만 CR을 못 이김.** 원인으로 sparse 보상과 부분관측을 지목.

**Stöckermann WSC 2023이 지적한 SMT2020 한계**: 제품 최대 10종(실제 팹은 10배 이상), tool dedication 단순, 같은 공정의 장비별 처리시간 차이 표현 불가.

### (b) 배치(확산로) · 스케줄링

| 서지 | 내용 | 확인 |
|---|---|---|
| Ali, El-Kholany, Gebser, PADL 2023, LNCS pp. 85–95, DOI 10.1007/978-3-031-24841-2_6 | SMT2020에서 FJSP 인스턴스를 뽑아 Hybrid ASP로 풂. 후속 확장에 batch processing, 다목적 포함 | Crossref, [GitHub](https://github.com/prosysscience/FJSP-SMT2020) README. 본문 미확인 |
| Kopp et al. WSC 2020 (CQT) | LV/HM CQT 디스패칭. 배치와 CQT 상호작용 | 원문 |

- **SMT2020 확산로 배치만 따로 다룬 논문은 찾지 못함.** 일반 배치 벤치마크는 4-4절.

### (c) 기타 시뮬레이션 활용

- **Andelfinger et al., "Sensitivity Analysis of the SMT2020 Testbed for Risk-Based Maintenance"**, WSC 2025, pp. 1804–1814, DOI 10.1109/WSC68292.2025.11338982 [원문]
  - LV/HM(913대)에서 장비군 특성 11개와 KPI 관계 분석. 고장 영향 민감도 → PM/고장 파라미터 근거로 유용.
- **Völker, Uzsoy, Mönch**, WSC 2022 [원문] — [PDF](https://informs-sim.org/wsc22papers/356.pdf). 제품 믹스 변화 생산계획. SMT2020은 향후 과제로만 언급.
- **Lee, Jeon, Park, "Multi-Resolution Modeling Method for AMHS in Semiconductor FABs"**, WSC 2023, pp. 2345–2356, DOI 10.1109/WSC60868.2023.10408363 [원문]
  - SMT2020 dataset 4(LV/HM_E) + SMAT2022 (OHT 500대, intrabay 40, interbay 3). OHT 디스패칭 SVF·NJF, 경로 Dijkstra.

---

## 3. SMAT2022

### WSC 2022 원 논문 [원문]

- Kwanwoo Lee, Siyong Song, Daesoon Chang, Sangchul Park (아주대 산업공학과), "A New AMHS Testbed for Semiconductor Manufacturing", WSC 2022, pp. 3318–3325, DOI 10.1109/WSC57314.2022.10015476 — [PDF](https://informs-sim.org/wsc22papers/339.pdf)
- Kopp/Hassoun 그룹과 별개인 아주대 독자 작업. 공정 데이터는 SMT2020을 그대로 재사용.
- 테이블 8개: TrackNode, TrackLink, Bay, Equipment, Station, VehicleType, Vehicle, ACID
- OHT 사양: 직선 5.0 m/s, 곡선 1.0 m/s, 가속 2.0 m/s², 감속 3.5 m/s² (한국 'S'사 자료 인용)
- 레이아웃: interbay 360 m, intrabay 60 m, interbay 2층, 중간 shortcut
- **반송시간: SMT2020은 5~10분 균등 가정인데, AMHS를 넣으면 평균 71.31초, 대부분 3분 이내.**
- T2T(장비 간 직접 반송) 비율 약 60%
- 다음 버전에서 4개 모델 전부와 층간·건물간 반송 추가 예고. 실제 공개 여부 미확인.
- 논문은 테이블 8개, 레포는 시트 9개(Address, ZCU, Rail, Bay, Equipment, PortType, Port, VehicleType, Vehicle). 레포가 이후 바뀐 것으로 보임.

### 나머지 3개 모델

- `kwoo-lee/smat2022`와 미러 [DRSCPARK/SMAT2022](https://github.com/DRSCPARK/SMAT2022)(박상철 교수 계정으로 추정, 2024-12) 모두 LVHM_E xlsx 하나뿐. 미러에는 WSC 2022·2023 논문 PDF도 있음.
- HVLM, LVHM, HVLM_E용 SMAT xlsx는 **찾지 못함(미공개로 보임).**
- 대안: [U-FAST](https://github.com/DaeeunLim/ufast) (MIT)
  - SMAT2022 원본을 변환한 `SMAT2022.rail` (노드 2,858, 링크 3,424, section 1,698, Equipment 1,115)
  - SMT2020 HVLM·LVHM에 더해 LVLM(route 4개)을 PySCFabSim 형식으로 동봉. LVLM은 공식 모델이 아니므로 출처 확인 필요.

### KCI 논문 [초록]

- 김동욱, 이관우, 박상철 (아주대), "SMAT2022를 이용한 시뮬레이션 기반 큐타임 예측 방법론", 한국CDE학회 논문집 28(2):137–145, 2023, DOI 10.7315/CDE.2023.137 — [KCI](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002963519)
- 큐타임 위반으로 생기는 폐기·재작업을 줄이려고 두 예측법을 비교: (1) 실험 결과로 학습한 ML 모델 (2) 다중 해상도 시뮬레이션. 정확도와 예측 시간으로 비교.
- SMAT2022 기반 시뮬레이션으로 검증한, 같은 그룹의 후속 활용 사례.

### 후속 · 관련

- 같은 그룹: Lee, Jeon, Park WSC 2023 (MRM), 위 KCI 논문
- **[synthetic-fab-simulator](https://github.com/aletheia-ops/synthetic-fab-simulator)** (2026-10): SMT2020 dataset 1을 SMAT2022 레이아웃에서 돌리고 OHT 물리 반송을 시뮬레이션해 LOT/장비/OHT 이벤트 로그를 JSONL로 생성. 파생 데이터는 별도 라이선스(DATA_NOTICE). **반송 로그가 필요하면 가장 직접적인 자원.**
- OHT RL 논문 (SMAT2022 사용 여부 **미확인**)
  - 다중 에이전트 RL OHT 디스패칭, Computers & Industrial Engineering 2025, S0360835225002554
  - 동적 링크 가중치 RL OHT 혼잡 완화, IJPR 2026, DOI 10.1080/00207543.2026.2656773
- 아주대 석사논문 "FAB내 물류 자동화 시스템의 유휴 차량 운영 및 차량 디스패칭" — [aurora.ajou](https://aurora.ajou.ac.kr/handle/2018.oak/24508). SMT2020 기반이라는 검색 요약만 있음(미확인).

---

## 4. 다른 팹 데이터

### 4-1. 공개 팹 시뮬레이션 · 스케줄링 테스트베드

| 이름 | 출처 | 접근 | 규모 · 포함 요소 | 우리 쓸모 |
|---|---|---|---|---|
| SMT2020 (4개 모델) | FernUni, PySCFabSim 레포(HVLM/LVHM TSV) | 직접 다운로드 (FernUni는 접속 실패) | 배치, 캐스케이드, 셋업, 고장/PM, 리워크, 샘플링, CQT, LTLD, 반송은 균등분포 | 확산로 배치 크기·공정시간·다운타임 **1차 소스** |
| SMAT2022 | [GitHub](https://github.com/kwoo-lee/smat2022) | LVHM_E만 공개 | OHT, 레일 그래프, 포트, 버퍼 | OHT/AMR 반송 레이아웃·차량 사양 **1차 소스** |
| MIMAC | Fowler & Robinson 1995. WSC 2019는 FernUni 공개라고 서술 | FernUni 경유로 보임(미확인). ASU 배포처 못 찾음 | 장비 최대 260, 제품 최대 21. DS1: 제품 2, toolset 68, 장비 211, 200+ step, 작업자 83 [WSC 2017]. DS6: 제품 9, 최대 355 step, tool group 104, 장비 228 [2차]. DS2~5 미확인 | 비교용 소형 모델. 확산 배치 포함 |
| MiniFab (Spier & Kempf 1995, Intel) | FernUni (WSC 2019 서술) | 공개로 서술(미확인) | 장비 5, workcenter 3, 제품 2, 6 step. 배치, 시퀀스 의존 셋업, 재진입 | **RL 프로토타입용 소형 환경으로 적합.** Stöckermann 2025, Andelfinger 2025도 사용 |
| Harris (Kayton et al. 1997) | FernUni (서술) | 미확인 | 장비 12, 제품 3, 최대 22 step | 제한적 |
| SEMATECH 300mm (Campbell & Ammenheuser 2000) | SEMATECH TR 99113848B-ENG | **비공개** | 장비 275, 제품 1, 364 step, AMHS 포함 | 접근 불가 |
| ASU/Intel DEVS 시계열 (Pendyala et al. 2024) | [arXiv 2408.09307](https://arxiv.org/abs/2408.09307) | 다운로드 링크 미확인 | MiniFab 기반 시계열 | 낮음 |
| 대만·한국 그룹 공개 팹 데이터 | — | 아주대 SMAT2022 외 못 찾음 | — | — |

### 4-2. 바로 쓸 수 있는 공개 RL 환경

| 이름 | URL | 내용 | 확인 |
|---|---|---|---|
| PySCFabSim | https://github.com/prosysscience/PySCFabSim-release (MIT) | SMT2020 Gym 환경, 행동·보상 설정 가능 | 원문 |
| U-FAST | https://github.com/DaeeunLim/ufast (MIT) | PySCFabSim 파생 생산 시뮬 + section 기반 OHT 반송 co-simulation, 혼잡 모델 4종 | README. RL 인터페이스 유무 미확인 |
| synthetic-fab-simulator | https://github.com/aletheia-ops/synthetic-fab-simulator | SMT2020 DS1 + SMAT2022 OHT 이벤트 로그 생성기 (RL 환경 아님) | README |
| Semiconductor_ClusterTool_Simulator | https://github.com/jongwon8707/Semiconductor_ClusterTool_Simulator | 클러스터 툴 내부 단위 Gymnasium 환경 + DQN | 검색 요약 |
| JSSEnv | https://github.com/prosysscience/JSSEnv (MIT) | 일반 JSSP Gym 환경 (Tassel et al., arXiv 2104.03760) | GitHub API |
| Event-driven RL (Yeganeh et al. 2026) | [arXiv 2606.10705](https://arxiv.org/abs/2606.10705) | RL 코드만 공개로 서술, 시뮬레이터·데이터는 독점 | 초록, 2차 |

### 4-3. 팹 운영 실데이터 (장비 로그, WIP, 반송 로그)

- **공개 사례 없음.** Stöckermann(Infineon), Andelfinger(D-SIMLAB)가 쓴 실데이터는 비공개.
- 대안: SMT2020 기반 시뮬레이터로 로그 직접 생성, 또는 synthetic-fab-simulator 생성 로그 사용.

### 4-4. 배치 기계 · 잡샵 벤치마크 인스턴스

| 이름 | URL | 접근 | 규모 | 쓸모 |
|---|---|---|---|---|
| Chiang, Cheng, Fu, C&OR 37(12):2257–2269, 2010 | https://web.ntnu.edu.tw/~tcchiang/publications/cor2010.htm ("COR2010-Problem instances.rar") | 직접 다운로드 | 4,860개. 병렬 배치기계, 비호환 family, 동적 도착, TWT 최소화. 확산 공정이 동기 | **확산로 배치 RL 오프라인 검증에 가장 적합** |
| Rolim, Tomazella, Nagano, Mendeley Data (2025) | https://data.mendeley.com/datasets/nmfsh6ygtk/3 (CC BY 4.0) | 직접 다운로드 | 비동일 병렬 배치기계 + family 셋업. 장비 2~12, 작업 10~400, family 2/4/6, 용량 50 | 셋업 있는 배치 변형 |
| Gahm et al. serial-batch | Mendeley [dr6bk5wv5c](https://data.mendeley.com/datasets/dr6bk5wv5c), [rxc695hj2k](https://data.mendeley.com/datasets/rxc695hj2k) | 미검증 | 93,360개로 서술 | serial-batch라 확산로와 다름 |
| Huertas & Van Hentenryck, IEEE TSM 38(4):901–916, 2025 | [arXiv 2410.11981](https://arxiv.org/abs/2410.11981) | 인스턴스 공개 미확인 | 비호환 family 병렬 배치를 CP로 풂 | 최신 비교 대상 |
| JSSP: Taillard, OR-Library | [JSPLIB](https://github.com/tamy0612/JSPLIB), [OR-Library](http://people.brunel.ac.uk/~mastjjb/jeb/orlib/jobshopinfo.html) | 직접 다운로드 | 결정론적 정적 JSSP | 성격 다름(배치·고장·재진입 없음) |

- DMU는 JSPLIB에 없고 원 배포처 미확인. Taillard 원 사이트는 접속 실패.

### 4-5. 참고: 성격이 다른 데이터 (품질·센서·수율)

스케줄링·반송 파라미터 소스로는 쓸 수 없다.

| 이름 | URL | 성격 |
|---|---|---|
| SECOM (UCI) | https://archive.ics.uci.edu/dataset/179/secom | 공정 센서 신호 + 합격/불량 |
| WM-811K | https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map | 웨이퍼 맵 불량 패턴 |
| Bosch Production Line | https://www.kaggle.com/c/bosch-production-line-performance | 생산라인 측정값·불량 예측 (반도체 아님) |

---

## 5. 우리 프로젝트에 주는 시사점

1. **파라미터 소스**
   - 확산로 배치 크기·공정시간·PM/고장: SMT2020 `tool`/`route`/`downcal`/`pmcal`
   - AMR/OHT 반송: SMAT2022 OHT 사양, 레일 그래프, "평균 71초 vs 5~10분" 결과
2. **선행 결과의 교훈**: Kovács et al. 2022에서 확산 장비군만 PPO로 제어해도 CR을 못 이겼다(sparse 보상, 부분관측). 보상 설계와 범위 축소(MiniFab, 다운스케일)의 근거로 인용할 수 있다.
3. **오프라인 검증**: 확산로 배치만 떼어 내면 Chiang 2010 인스턴스를 비교 기준으로 쓸 수 있다.

## 미확인 항목

- SMT2020 TSM 원문 본문
- FernUni 배포 페이지 현재 상태
- MIMAC DS2~5 상세
- SMAT2022 나머지 3개 모델 xlsx
- OHT RL 논문들의 SMAT2022 사용 여부
- SMT2020 확산로 배치 전용 연구
- 로컬 `STNQTY` 합(1,443 / 1,313)이 문헌 장비 수와 다른 이유
