# 보고서용 그림

모두 PNG(160 dpi). 다시 만들기:

```
python3 experiments/plot_results.py    # e3_*, e6_*
python3 experiments/plot_ppo_diag.py   # ppo_diag
```

| 파일 | 내용 | 데이터 |
|---|---|---|
| `e3_test_vs_cov.png` | E3 시험: 장비군별·전체 lot당 지연의 cov 대비(%). 학습 시드 10개씩 점, 평균(◆)과 t 95% 구간. DQN 전체 −13.0%, PPO −5.7% | `results/eval/*_e3_s*.json` (analyze.py와 같은 계산) |
| `e6_implant_wait.png` | E6 공장 검증: implant 장비 앞 대기의 cov 대비(%). 왼쪽 = 한 장비군에만 RL(주 결과), 오른쪽 = 세 장비군 동시(보조). 공장 시드 4·5·6 점, ◆ 평균 | `results/implant_setup/SMT2020_HVLM_fifo_{covfb,rl_*_e3_*}_Demand_730d_s{4,5,6}` (fab_report.py와 같은 계산) |
| `e6_ontime_vs_ct.png` | E6 27회 실행의 공장 평균 CT(일반 lot)와 납기 준수율. 상관 −0.83, CT 1일당 약 −6.7%p. 준수율 변화는 implant보다 공장 전체 CT 흔들림을 따라감 | 위와 같음 |
| `ppo_diag.png` | PPO 학습 진단 지표(10장 24쪽): 엔트로피, 근사 KL, clip fraction, value 손실, explained variance, 학습 보상. 최종 설정 시드 0·1 | `results/rl_ppo/result_up_lt_e1w40diag_s{0,1}.json` |

색: DQN 파랑, PPO 주황, cov 회색 (모든 그림 같음).
