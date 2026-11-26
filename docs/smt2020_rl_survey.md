# SMT2020 기반 RL · 학습형 의사결정 선행연구 전수 조사 (2020~2026)

작성: 2026-11-04 · 웹 리서치(서브에이전트) + 일부 직접 대조
PDF: `docs/papers/` (Kovács 박사논문 2023, WSC 2022~2024 11편, IJCAI 2022 2편, Yeganeh 2026 등)

**확인 수준.** [원문] 본문 직접 확인 · [초록] 초록/메타데이터 · [2차] 다른 문헌·검색 요약 · [인용만] 인용 목록에만 있고 실제 사용 미확인

## 핵심 결론

1. **SMT2020을 실제로 돌려서 학습형 의사결정을 한 연구는 약 8건뿐이고, 거의 한 연구진에 몰려 있다.** Klagenfurt대(Gebser), Infineon, D-SIMLAB 쪽이다.
2. **결정 문제는 대부분 lot 디스패칭(우선순위 매기기)이다.** 그 밖에는 CQT 위반 예측 1건, 용량계획 1건 정도다.
3. **셋업 전환, 확산로 배치 시작/대기, 노광 디스패칭을 전용으로 다룬 RL 연구는 없다.** 노광만 제어해 본 실험은 있지만 효과가 미미했다.
4. **RL이 휴리스틱을 이긴 경우는 ES·GP로 전 팹을 제어했을 때다.** PPO로 한 장비군만 제어하면 졌다.
5. **SMT2020 공식 개정판은 없다.** 대신 2025~2026년에 제3자 재현 자료가 나왔다. 그중 하나는 "배치 구성 정책이 원 사양에 불충분하게 정의되어 있고, CT가 이에 민감하다"를 주제로 한다.

## 1. 조사 방법

- **OpenAlex 인용 추적**: 아래 6편을 인용한 논문을 모았다(중복 제거 105건).
  - Kopp 2020 TSM(64회), Hassoun 2019 WSC(19), Kovács 2022 WSC(3), Kovács 2022 ASMC(10), Kopp 2020 WSC CQT(15), Tassel 2023 WSC(20)
- **arXiv**: `abs:SMT2020` 2건, `abs:PySCFabSim` 0건
- **Zenodo**: `SMT2020` 6건(관련 4건)
- **GitHub, WebSearch**: 영어·중국어 쿼리
- **본문 대조**: WSC PDF 본문에서 "SMT2020"을 검색해 실제로 썼는지 가렸다.
- **한계**: Semantic Scholar는 요청 한도 초과(429)로, Google Scholar는 조회 수단이 없어 쓰지 못했다. Scholar에만 잡히는 학위논문이나 중국어 문헌은 빠졌을 수 있다.

## 2. 논문 표

### 2-A. SMT2020을 실제로 쓴 학습형 의사결정 연구

| 서지 | 데이터 | 결정 문제 | 범위 | 상태·행동·보상 | 알고리즘 | 기준선 | 결과 | 코드 | 확인 |
|---|---|---|---|---|---|---|---|---|---|
| Kovács et al. WSC 2022 | SMT2020 (DEMO_ENV_2, 모델 미확인) | lot 디스패칭 | Diffusion + DefMet 2대 | 행동 4종(공정 선택, 큐 배정, lot 선택, 휴리스틱 선택). 보상 dense(납기·WIP·가동률) + sparse(완료, coupling 위반) | PPO(SB3), 1M step, 튜닝 없음 | CR | **CR보다 나쁨** (sparse 보상, 부분관측) | MIT ([PySCFabSim](https://github.com/prosysscience/PySCFabSim-release)) | 원문 |
| Kovács, IJCAI 2022 박사과정 컨소시엄 ([PDF](https://www.ijcai.org/proceedings/2022/0831.pdf)) | dataset 1 | 디스패칭(GP·RL 계획) | 전 팹 | 시뮬레이터 검증표, 초기 구상 | — | 원 논문 수치 | 재현성만 제시 | — | 원문 |
| Kovács, Tassel, Gebser, GECCO 2023, pp. 1374–1382 (박사논문 5장) | PySCFabSim (모델 미확인) | 장비군별 lot 정렬 규칙 | 전 팹(장비군마다 다른 규칙) | 계층형 정렬 기준을 GP 트리로 진화 | 유전 프로그래밍(개체 48, 40세대) | 기본 계층 휴리스틱(coupled 우선 → 우선순위 → 최소 셋업 → CR) | 납기 지연 lot 비율 77% → 40% 미만, 완성 +0.31%, 일반 lot CT 약 1일 단축(2년, 10회). **implant 규칙은 셋업 시간을 최우선으로 학습** | 저장소에 재현 안내 | 박사논문 원문 (GECCO 본문 미열람) |
| Kovács 박사논문 2023 7장 ([PDF](https://netlibrary.aau.at/obvuklhs/download/pdf/9992946)) | PySCFabSim | CQT 위반 예측 → 우선순위 상향 | 전 팹 | lot 특징 → DNN 분류기. 위반이 예측되면 디스패칭 최상위로 | DNN(샘플 180만) | 계층 휴리스틱(위반율 7% 이상) | 정확도 89.76%, **위반 8% 이상 감소** | 부록에 언급 | 원문 |
| Tassel et al. WSC 2023, pp. 1924–1935 | **HV/LM, LV/HM** (6개월, 시드 100개) | 전역 lot 우선순위 | 전 팹, 단일 에이전트 | 상태: lot당 특징 13개(셋업 호환 장비 수, 배치 크기 등). 행동: 합법 lot 순서. 장비 배정은 SMT2020 규칙. 보상: 지연 패널티 + CT + WIP의 에피소드 목적함수 | self-attention + 자기지도 사전학습 + ES | SMT2020 원 계층 휴리스틱 | 비용 **HV/LM −16.68%, LV/HM −7.66%**. HV/LM 일반 lot 납기 준수 44.91%, CT −26.21%(CR 대비) | [RL4SemiconductorFabSched](https://github.com/ingambe/RL4SemiconductorFabSched) | 원문 |
| Stöckermann et al. IJAMT 2025 | SMT2020 (D-SIMCON, 변형 미확인) | lot 디스패칭 | litho만 / 장비군 조합 / 전 장비 | 큐의 lot별 관측, KPI 보상 | PPO, CMA-ES | **CR** | litho만: PPO 지연 **−5.7%(악화)**, throughput +3.1% / CMA-ES 지연 +1.4%, throughput +1.5%. 전 장비 ES: 지연 두 자릿수 % 개선(부록 그림, 수치 미독). 50일 학습 → 180일 시험에서 일부 손실. 한 장비군만 제어하면 일반화 나쁨 | 비공개 | 원문 |
| Immordino et al. JIM 37(5) 2025 | SMT2020 등 | RL 디스패칭 정책의 XAI | — | — | XAI | — | "큰 SMT2020에서는 방법을 바꿔야 했고 결과가 덜 만족스러웠다" | 미확인 | 2차 |
| Andelfinger et al. 2025 (arXiv 2509.15767) | SMT2020 예비(D-SIMCON) | **용량계획**(가동시간·효율, dedication) | 전 팹, 주 단위 | 이종 그래프(장비–공정) | HGNN + n-step PPO | 무행동 / 랜덤 / WIP 휴리스틱 | throughput +1.8%, CT −1.8% | 미확인 | 원문 |

### 2-B. SMT2020을 썼지만 학습형이 아니거나 도구·데이터 성격

| 서지 | 내용 | 확인 |
|---|---|---|
| Ali, El-Kholany, Gebser, PADL 2023 | SMT2020에서 FJSP 인스턴스 추출, ASP로 풂(정적·결정론). 후속 저장소에서 배치·다목적으로 확장 | Crossref, README |
| "Hybrid ASP-Based Multi-objective Scheduling…", LNCS 2023, DOI 10.1007/978-3-031-43619-2_17 | ASP 다목적 스케줄링 | 인용만 |
| Kovács, Tassel, Gebser, "PySCFabSimUI", ICAPS 2023 데모 | PySCFabSim GUI. 학습 정책과 휴리스틱을 시각적으로 비교 | 원문 |
| [PySCFabSim-revised](https://github.com/david-dd/PySCFabSim-revised) (Heik, Kovács, Rodmann, 2025, CC BY 4.0) | 리팩터링판. 학위논문용 RL 실험 설정 포함(HVLM, Diffusion, FIFO 기반 PPO로 추정). 결과 미확인 | 저장소 구조만 |
| Heik, Leißau, Reichelt 외, "Implementing the SMT2020 Wafer Fab Data Sets Using Different Simulation Packages" (IEEE TSM 투고 초안, 2025) | 시뮬레이터별 재구현 비교. 원자료 [Zenodo 15798604](https://zenodo.org/records/15798604), [AnyLogic 템플릿](https://github.com/mdlnlss/smt2020-anylogic-template) | 게재 여부 미확인 |
| Kim, Lee, Park WSC 2023 | SMT2020 + AMHS 큐타임 예측(ML 예측, 의사결정 아님) | 원문 |
| "Reverse Engineering the Future" WSC 2023 | LV/HM 역방향 시뮬레이션(비학습) | 원문 |
| Andelfinger et al. WSC 2025 | LV/HM PM 우선순위 민감도 분석 | 원문 |

### 2-C. SMT2020을 인용만 하고 실제로는 쓰지 않은 학습형 연구 (방법론 참고용)

- **Yedidsion et al. WSC 2022**: QTC 구간 진입 lot 투입을 DRL로 결정. 기준선 7종보다 우수. 자체 모델 [원문]
- **Zhang, Kabak, Heavey, Rose WSC 2023**: litho RL. 단순화한 자체 팹에서 tabular Q-learning으로 규칙 선택 [원문]
- **Chan, Gan, Cai WSC 2023**: 상황 인식 ML 디스패칭. SMT2020은 향후 과제로만 언급 [원문]
- **Norman, Dawadi, Yedidsion WSC 2024**: GNN-RL로 디스패칭 규칙 파라미터 조정 [원문]
- **Bosse et al. WSC 2024**: DRL 표준 인터페이스. "MiniFab → SMT2020 무수정 전환 가능"이라고만 서술 [원문]
- **Yeganeh et al. arXiv 2606.10705**: event-driven RL. 독점 시뮬레이터, RL 코드만 MIT 공개 ([EventDrivenTD-RL](https://github.com/YavarYeganeh/EventDrivenTD-RL)) [원문]

### 2-D. 인용은 했지만 실제 사용 여부를 확인하지 못한 학습형 연구 (전부 미확인)

- Ali, Eftekhari, Gebser, Leitner, Friedrich, "Swarm Intelligence-Driven Dispatching Rules…", LNNS 2025, pp. 156–181 — 공저자 Eftekhari가 [PySCFabSimRenewed](https://github.com/Peymanef2020/PySCFabSimRenewed)("RL based Scheduling for SMT2020")를 운영하므로 사용 가능성 높음
- Lin, Tseng, Tsai, "Two-Level Pareto-Based HRL…", IEEE TSM 39(3):590–613, 2026
- Xiang, Zhang, Zhang, Wang, IJPR 64(7):2731–2752, 2025 — **배치 장비 RL**(배치 구성 + 배치 배정 에이전트), 흐름시간 20~25% 이상 감소. 벤치마크 출처 미확인
- Kuo, Hong, Chien, C&IE 208, 2025 (DRL 디지털 트윈 생산계획)
- Leißau & Laroque WSC 2025 / Mondesire et al. "FABSIM" WSC 2025 / Na & Kim SSRN 2026 / Cho, Lee, Park C&IE 2026 / Cho, Kim, Park WSC 2025

## 3. 종합

### 결정 문제별 현황

| 결정 문제 | SMT2020에서의 현황 |
|---|---|
| 전역 lot 우선순위 / 디스패칭 | **많음**: Tassel(ES), Kovács(GP), Stöckermann(ES/PPO), Immordino(XAI) |
| 특정 장비군 디스패칭 | 적고 결과 약함: Kovács 2022 Diffusion PPO(CR에 패배), Stöckermann litho만(PPO 악화, ES +1.4%) |
| CQT 관리 | 적음: Kovács ML 예측(위반 −8%). RL 투입 제어는 자체 모델만(Yedidsion) |
| 용량계획 | 1건(Andelfinger, +1.8%) |
| PM | 민감도 분석만. RL 없음 |
| **셋업 전환** (implant 가스, litho reticle) | **암묵적으로만 다뤄짐.** Kovács GECCO 2023(박사논문 5장)의 GP 장비군별 lot 정렬 규칙이 implant에서 셋업 시간을 지배적 기준으로 학습함(정성적 서술, 박사논문 5.5절 원문 확인). lot 순서를 고르면 셋업 전환이 따라 정해지는 구조라 implant 가스 전환 결정이 포함돼 있음. 다만 (1) implant만 떼어 본 결과나 셋업 비율 보고가 없고 (2) 장비–가스 배정, 일부러 기다리기, 미리 교체하기 같은 결정은 없으며 (3) 최적해 비교가 없음. 기준선에 이미 "셋업 시간 최소 우선"이 3순위로 들어 있음. Tassel 특징에 셋업 호환 장비 수 포함 |
| **배치 시작/대기** (확산로) | **전용 연구 없음.** Stöckermann: "확산 배치 제어는 다른 장비와 함께 제어할 때만 효과"(MiniFab 관찰, SMT2020도 유사하다고 서술). Shibata 2026: 배치 구성 정책이 원 사양에 불충분하게 정의됨 |
| 투입 (release) | SMT2020 RL 사례 없음 |

### RL이 이긴 경우와 진 경우

- **이김**: 미분 없는 최적화(ES, GP) + 에피소드 단위 목적함수 + 전 팹·다수 장비군 제어. Tassel(−7.7~−16.7%), GP(지연 lot 약 50%↓), Stöckermann 전 장비 ES
- **짐**: PPO + sparse·지연 보상 + 단일 장비군. Kovács 2022 Diffusion, Stöckermann litho만 PPO(−5.7%)
- **저자들이 꼽은 원인**: 행동과 결과 사이 긴 지연, 노이즈 많은 KPI, 부분관측
- **기준선 선택이 결과를 좌우한다.** SMT2020에서 지연 기준으로는 CR, 완성량 기준으로는 FIFO가 강하다(Stöckermann Table 5).

### 우리 프로젝트에 주는 시사점

- 축소 MDP로 범위를 좁히면 위의 실패 원인(긴 보상 지연, 노이즈)을 직접 줄일 수 있다.
- 동시에 "한 장비군만 제어하면 팹 전체 개선 폭이 작다"는 선행 결과가 있다. 그래서 KPI를 팹 전체가 아니라 장비군 단위로 한정하고, 이 한계를 보고서에 명시해야 한다.
- 셋업 전환과 배치 시작/대기는 SMT2020에서 비어 있는 영역이다.

## 4. 최신 데이터 현황

- **SMT2020 공식 개정판, Mönch·Kopp·Hassoun 그룹의 2023~2026년 신규 testbed**: 찾지 못함
- **FernUni 배포 페이지**: `p2schedgen.fernuni-hagen.de` 접속 불가(본교 사이트는 정상). 운영 여부 미확인
- **Zenodo P-30** (Shibata, SimTrek, 2026, CC BY 4.0) — README 직접 확인(`docs/zenodo_P30_README.md`). DOI는 README 기준 `10.5281/zenodo.20635151`
  - SMT2020 전 팹 4년 시뮬레이션 30회 반복, 반복당 약 650MB(전체 약 20GB)
  - lot×step 처리 로그(반복당 약 3,800만 행), 장비 정지(PM/BD), KPI·일별 WIP CSV
  - 엔진(GoSoL)은 비공개. Saleable/Engineering 구분이 있어 _E 변형으로 추정(미확인)
  - 짝 논문: "Independent Reproduction of the SMT2020 Full-Fab Benchmark: Capacity Equivalence and Cycle-Time Sensitivity to an Underspecified Batch-Composition Policy" (2026, 게재처 미확인)
- **기타**: [PySCFabSim-revised](https://github.com/david-dd/PySCFabSim-revised), [Zenodo 15798604](https://zenodo.org/records/15798604), [AnyLogic 템플릿](https://github.com/mdlnlss/smt2020-anylogic-template)
- **이전 보고분**: SMAT2022, U-FAST, synthetic-fab-simulator, ASU DEVS 시계열(arXiv 2408.09307)
- **실팹 운영 데이터 공개**: 없음. Infineon, D-SIMLAB, Procedia CIRP 2026 멀티팹, Deenen 2026 litho는 모두 산업 데이터

## 미확인 항목

- Semantic Scholar·Google Scholar 교차확인
- 2-D 9건의 SMT2020 사용 여부
- GECCO 2023, Immordino 2025 본문
- Stöckermann 전 장비 ES 정확 수치
- Shibata 짝 논문 본문·게재처
- FernUni 페이지 현황

## 부록 (2026-11-05): SMT2020 밖에서 셋업만 떼어 다룬 연구. 초록만 확인, 본문 미열람

목표 지표는 팹 전체 CT가 아니라 그 장비 구간의 지표(총 지연, 총 셋업 시간)다. 문제는 "셋업이 있는 비동일 병렬기계 스케줄링(UPMSP)"으로 정식화한다.
- Paeng, Park, Park, IEEE Access 2021: 시퀀스 의존 family 셋업이 있는 비동일 병렬기계, 총 지연 최소화, DQN. 메타휴리스틱·규칙·다른 RL보다 낫다고 보고. https://doaj.org/article/5af43b17cadd4729b3aa6b41d4132ab4
- WSC 2024 (con258): "이온주입은 장비·시퀀스 의존 셋업이 있는 UPMSP로 모델링할 수 있다." 출시일·장비 자격 포함, 가중 지연 최소화, RL. https://informs-sim.org/wsc24papers/con258.pdf
- Computers & Industrial Engineering 2022: 이온주입 스케줄링을 UPMSP(셋업·출시·만료 시간)로 보고 tabu search로 풂(비RL).
- WSC 2024 (con185): 동일 병렬기계, 셋업 시간 + 지연을 함께 다루는 DRL. https://informs-sim.org/wsc24papers/con185.pdf
- 다목적 판(총 가중 지연 + 총 셋업 시간), graph PPO: arXiv 2602.08052

**본문 확인 (2026-11-06):**
- Cho, Kim, Mönch, WSC 2024 (con258): 이온주입은 서론에서 "이렇게 모델링할 수 있다"고 한 문장 언급할 뿐이다. 실험은 무작위로 생성한 정적 인스턴스(12~50 job, 3~6대)이고, SMT2020이나 실제 implant 데이터는 쓰지 않았다. 정해진 job 묶음 전체의 일정을 짜는 문제이며, GNN + RL로 총 가중 지연을 줄인다. 12 job에서 평균 지연은 RL 195.22, ATC 계열 201.32, MIP 최적해 144.20이다.
- Nam, Baek, Cho, Woo, WSC 2024 (con185): 동일 병렬기계 5대, job 100개, 도착이 이어지는 동적 환경이다. PPO로 휴리스틱 4개(SSPT, ATCS, MDD, COVERT) 중 하나를 고르며, 평균 셋업 약 5% 감소를 보고했다. 저자들이 파라미터를 임의로 정한 가상 사례이고 반도체 데이터가 아니다.

**본문 확인 (2026-11-06):** Wang, Yang, Lin, *C&IE* 211 (2026) 111620, doi:10.1016/j.cie.2025.111620. 본문을 읽었다. PDF 원본은 사용자의 Downloads에 있다.
- 다룬 것: implant 병렬기계 스케줄링, 셋업 횟수·makespan·폐기량 복합 보상, 실제 파운드리 데이터(작업 100개, 장비 10대, 논문에 "confidential"로 명시), 긴급 주문·몰림 시나리오.
- 안 다룬 것: 일부러 기다리기(행동은 "다음 작업을 어느 장비에 넣나"뿐), 재진입 도착 흐름, 고장·PM(가정으로 제외), 공장 전체 효과, 튜닝된 대기 규칙과의 비교, SMT2020.
- 방법: 표 형식 double Q-learning. 상태는 작업 번호이고 표 크기는 작업 수 × 장비 수(100 × 10)다. 장비의 현재 가스는 상태 표에 없다.
- 주의: 초록은 SPT와 비교했다고 하지만 결과표에는 없다. 작업 선택 규칙이 FIFO(행동 절)와 SPT(학습 절)로 다르게 적혀 있다. Table 5의 Objective(QRL 0.9113)와 Table 6(같은 makespan·setup·scrap에서 0.32)이 다르다. Table 8의 M, L Objective는 Table 5의 Set C, B 소형 값과 같다. 통계 검정은 없고 30회 평균/표준편차만 있다.
