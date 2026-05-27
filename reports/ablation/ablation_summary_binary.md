# Ablation results -- binary task

Primary metric: **test_macro_f1**  |  seeds: [42, 43, 44, 45, 46]

## Per-condition summary

| condition | n_features | test_macro_f1_mean | test_macro_f1_std | test_macro_f1_ci95_lo | test_macro_f1_ci95_hi | runtime_s |
|---|---|---|---|---|---|---|
| filter_top30 | 30 | 0.9183 | 0.0004 | 0.9180 | 0.9186 | 34.0255 |
| baseline_all_features | 190 | 0.9175 | 0.0006 | 0.9170 | 0.9180 | 34.9695 |
| rfa_svm_original | 190 | 0.9170 | 0.0006 | 0.9166 | 0.9175 | 35.0474 |
| rfa_rf_proposal | 14 | 0.9091 | 0.0004 | 0.9088 | 0.9094 | 15.0579 |
| rfa_plus_flow_pair | 394 | 0.9048 | 0.0004 | 0.9045 | 0.9051 | 61.5232 |

## Pairwise comparisons (Holm--Bonferroni corrected, effect-size floor 0.005)

| cond_a | cond_b | mean_a | mean_b | delta_a_minus_b | p_wilcoxon | p_holm | claim |
|---|---|---|---|---|---|---|---|
| baseline_all_features | filter_top30 | 0.9175 | 0.9183 | -0.0008 | 0.0625 | 0.6250 | False |
| baseline_all_features | rfa_svm_original | 0.9175 | 0.9170 | 0.0005 | 0.4375 | 0.6250 | False |
| baseline_all_features | rfa_rf_proposal | 0.9175 | 0.9091 | 0.0084 | 0.0625 | 0.6250 | False |
| baseline_all_features | rfa_plus_flow_pair | 0.9175 | 0.9048 | 0.0127 | 0.0625 | 0.6250 | False |
| filter_top30 | rfa_svm_original | 0.9183 | 0.9170 | 0.0013 | 0.0625 | 0.6250 | False |
| filter_top30 | rfa_rf_proposal | 0.9183 | 0.9091 | 0.0092 | 0.0625 | 0.6250 | False |
| filter_top30 | rfa_plus_flow_pair | 0.9183 | 0.9048 | 0.0135 | 0.0625 | 0.6250 | False |
| rfa_svm_original | rfa_rf_proposal | 0.9170 | 0.9091 | 0.0079 | 0.0625 | 0.6250 | False |
| rfa_svm_original | rfa_plus_flow_pair | 0.9170 | 0.9048 | 0.0122 | 0.0625 | 0.6250 | False |
| rfa_rf_proposal | rfa_plus_flow_pair | 0.9091 | 0.9048 | 0.0043 | 0.0625 | 0.6250 | False |
