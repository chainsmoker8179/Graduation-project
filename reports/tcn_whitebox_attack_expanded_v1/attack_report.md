# tcn 白盒攻击 Smoke 报告

## 实验设置
- 样本数: `62`
- 资产目录: `/home/chainsmoker/qlib_test/.worktrees/multi-model-whitebox-attack/artifacts/tcn_probe_assets`
- 配置文件: `/home/chainsmoker/qlib_test/.worktrees/multi-model-whitebox-attack/origin_model_pred/TCN/model/model_config.json`
- 权重文件: `/home/chainsmoker/qlib_test/origin_model_pred/TCN/model/tcn_state_dict.pt`
- price_epsilon: `0.01`
- volume_epsilon: `0.02`
- pgd_steps: `5`
- pgd_step_size: `0.25`

## Clean Gate
- clean_loss: `0.019327`
- clean_grad_mean_abs: `3.538420e-05`
- clean_grad_finite_rate: `1.000000`
- spearman_to_reference: `0.9941577900329884`
- feature_mae_to_reference: `0.034922510385513306`
- feature_rmse_to_reference: `0.09917979687452316`
- feature_max_abs_to_reference: `0.6099975109100342`

## 攻击结果
- FGSM MSE: `0.053335`
- PGD MSE: `0.120037`
- FGSM 平均价格预算使用率: `0.433065`
- PGD 平均价格预算使用率: `0.268926`
- FGSM 平均成交量预算使用率: `0.994960`
- PGD 平均成交量预算使用率: `0.735585`
