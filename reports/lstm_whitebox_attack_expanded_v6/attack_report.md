# LSTM 白盒攻击 Smoke 结果

- 样本数：62
- clean_loss：0.01431117
- fgsm_loss：0.10467355
- pgd_loss：0.28165913
- clean Spearman 对齐：0.7513031653697968
- clean 特征 MAE：0.034524187445640564
- clean 特征 RMSE：0.09823775291442871
- clean gate 阈值：`{'min_clean_grad_mean_abs': 1e-06, 'min_spearman_to_reference': 0.09, 'max_feature_mae_to_reference': 0.05, 'max_feature_rmse_to_reference': 0.12, 'max_feature_max_abs_to_reference': 0.7}`
- FGSM 平均预测偏移：0.19706690
- PGD 平均预测偏移：0.36676064
- 样本级明细：`reports/lstm_whitebox_attack_expanded_v6/sample_metrics.csv`
- 汇总 JSON：`reports/lstm_whitebox_attack_expanded_v6/attack_summary.json`