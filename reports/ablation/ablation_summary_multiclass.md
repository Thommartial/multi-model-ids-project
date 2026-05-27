# Ablation results -- multiclass task

Primary metric: **test_macro_f1**  |  seeds: [42, 43, 44, 45, 46]

## Per-condition summary

| condition | n_features | test_macro_f1_mean | test_macro_f1_std | test_macro_f1_ci95_lo | test_macro_f1_ci95_hi | runtime_s |
|---|---|---|---|---|---|---|
| rfa_rf_proposal | 14 | 0.6310 | 0.0026 | 0.6289 | 0.6330 | 18.2627 |
| filter_top30 | 30 | 0.5738 | 0.0036 | 0.5713 | 0.5768 | 39.8126 |
| baseline_all_features | 190 | 0.5295 | 0.0069 | 0.5243 | 0.5349 | 43.6644 |
| rfa_svm_original | 190 | 0.5288 | 0.0043 | 0.5256 | 0.5321 | 42.0421 |
| rfa_plus_flow_pair | 394 | 0.5042 | 0.0072 | 0.4982 | 0.5092 | 69.7327 |

## Pairwise comparisons (Holm--Bonferroni corrected, effect-size floor 0.005)

| cond_a | cond_b | mean_a | mean_b | delta_a_minus_b | p_wilcoxon | p_holm | claim |
|---|---|---|---|---|---|---|---|
| baseline_all_features | filter_top30 | 0.5295 | 0.5738 | -0.0443 | 0.0625 | 0.6250 | False |
| baseline_all_features | rfa_svm_original | 0.5295 | 0.5288 | 0.0007 | 0.8125 | 0.8125 | False |
| baseline_all_features | rfa_rf_proposal | 0.5295 | 0.6310 | -0.1015 | 0.0625 | 0.6250 | False |
| baseline_all_features | rfa_plus_flow_pair | 0.5295 | 0.5042 | 0.0253 | 0.0625 | 0.6250 | False |
| filter_top30 | rfa_svm_original | 0.5738 | 0.5288 | 0.0449 | 0.0625 | 0.6250 | False |
| filter_top30 | rfa_rf_proposal | 0.5738 | 0.6310 | -0.0572 | 0.0625 | 0.6250 | False |
| filter_top30 | rfa_plus_flow_pair | 0.5738 | 0.5042 | 0.0696 | 0.0625 | 0.6250 | False |
| rfa_svm_original | rfa_rf_proposal | 0.5288 | 0.6310 | -0.1021 | 0.0625 | 0.6250 | False |
| rfa_svm_original | rfa_plus_flow_pair | 0.5288 | 0.5042 | 0.0246 | 0.0625 | 0.6250 | False |
| rfa_rf_proposal | rfa_plus_flow_pair | 0.6310 | 0.5042 | 0.1268 | 0.0625 | 0.6250 | False |
