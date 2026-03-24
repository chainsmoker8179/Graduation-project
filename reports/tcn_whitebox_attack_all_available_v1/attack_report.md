# tcn 白盒攻击 Smoke 报告

## 实验设置
- 样本数: `124`
- 资产目录: `/home/chainsmoker/qlib_test/.worktrees/multi-model-whitebox-attack/artifacts/tcn_probe_assets`
- 配置文件: `/home/chainsmoker/qlib_test/.worktrees/multi-model-whitebox-attack/origin_model_pred/TCN/model/model_config.json`
- 权重文件: `/home/chainsmoker/qlib_test/origin_model_pred/TCN/model/tcn_state_dict.pt`
- price_epsilon: `0.01`
- volume_epsilon: `0.02`
- pgd_steps: `5`
- pgd_step_size: `0.25`

## Clean Gate
- clean_loss: `0.024401`
- clean_grad_mean_abs: `2.041333e-05`
- clean_grad_finite_rate: `1.000000`
- spearman_to_reference: `0.9922958300550746`
- feature_mae_to_reference: `0.034121543169021606`
- feature_rmse_to_reference: `0.09709380567073822`
- feature_max_abs_to_reference: `0.6123437881469727`

## 攻击结果
- FGSM MSE: `0.057216`
- PGD MSE: `0.130696`
- FGSM 平均价格预算使用率: `0.432661`
- PGD 平均价格预算使用率: `0.262733`
- FGSM 平均成交量预算使用率: `0.995161`
- PGD 平均成交量预算使用率: `0.710862`
