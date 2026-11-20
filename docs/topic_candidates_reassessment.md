# 주제 후보 재평가: implant 교체 전 대기 vs Lassoued 동적 job shop (2026-11-06)

## 결론

- **implant 교체 전 대기를 주 후보로 둔다.** 구간 이득과 장비 시간 절약은 실측이 있다. RL 몫은 아직 모른다.
- **Lassoued 고장 위험 관측안은 철회한다.** 공개 코드에서는 장비마다 한 판에 고장이 한 번뿐이고, 시작 직후에 끝나며, 사용량과 무관하다.

## Lassoued 코드 확인 (PetriRL_DJSSP, 코드 읽기만 함)

- `env/envs/dft/simulator.py:132` reset에서 `gen_machines_down`을 한 번 호출한다. `:199-205`는 시각이 [고장, 수리 끝) 안이면 장비를 막는다. 즉 장비당 고장 1회다.
- `env/utils/time_gen.py`: 고장 시각 = Weibull(1.5) × MTBF, MTBF = U(0.5, 1) × 최대 처리시간, 수리 = U(0.2, 0.5) × 최대 처리시간.
- 추정: 최대 처리시간 99이면 고장은 평균 약 67 step, 수리는 약 35 step이다. 논문의 15×15 makespan 1545의 2%쯤이고, 앞부분에 몰린다.
- 논문 실험이 이 코드와 같은 설정인지는 확인하지 못했다. 논문은 수리 시간을 정규분포로 적었고 코드는 균등분포다.

## implant 실측 근거 (EDA 05)

- 30분 대기: 교체 시간 비율 15.2→9.7% (10대), 11.1→7.4% (8대), 14.4→11.4% (6대). 장비 시간으로 약 0.55 + 0.30 + 0.18 ≈ 1대분 / 24대 (계산).
- 공장 CT 이득은 검출되지 않았다. 이 이득은 고정 규칙의 몫이고 RL의 몫이 아니다.
- 다음 측정: 튜닝 고정 대기, 대기열 임계값 규칙, 도착을 미리 아는 경우의 상한.

## 선행 연구

implant: Wang C&IE 2026, Norman WSC 2024, Cho·Kim·Mönch WSC 2024, Soykan WSC 2025, Kovács GECCO 2023, Chen 외 C&IE 2022, polling 이론(Hu·Dong·Perry AAP 2022, Duenyas·Van Oyen 1996, Günalay, Performance Evaluation Review 2026).
Lassoued: Lassoued 2026, Zeng arXiv 2201.00548, Xu 2024, Tang 2026, Li·Shaw 1998, Sobaszek 2018, Yan Machines 2022, Polimi 학위논문, Zhejiang 2026. 확인 수준은 대화 답변과 `reports/제조 스케줄링 RL 선행연구와 데이터.md` 참고.
