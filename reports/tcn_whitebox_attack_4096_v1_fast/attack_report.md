# tcn 白盒攻击 Smoke 报告

## 实验设置
- 样本数: `3983`
- 资产目录: `/home/chainsmoker/qlib_test/.worktrees/multi-model-whitebox-attack/artifacts/tcn_attack_4096_v1_fast`
- 配置文件: `/home/chainsmoker/qlib_test/.worktrees/multi-model-whitebox-attack/origin_model_pred/TCN/model/model_config.json`
- 权重文件: `/home/chainsmoker/qlib_test/origin_model_pred/TCN/model/tcn_state_dict.pt`
- price_epsilon: `0.01`
- volume_epsilon: `0.02`
- pgd_steps: `5`
- pgd_step_size: `0.25`

## Clean Gate
- clean_loss: `0.024793`
- clean_grad_mean_abs: `7.006160e-07`
- clean_grad_finite_rate: `1.000000`
- spearman_to_reference: `0.9966208564478164`
- feature_mae_to_reference: `0.0`
- feature_rmse_to_reference: `0.0`
- feature_max_abs_to_reference: `0.0`

## 攻击结果
- FGSM MSE: `0.065019`
- PGD MSE: `0.144405`
- FGSM 平均价格预算使用率: `0.431328`
- PGD 平均价格预算使用率: `0.265414`
- FGSM 平均成交量预算使用率: `0.995635`
- PGD 平均成交量预算使用率: `0.731585`
