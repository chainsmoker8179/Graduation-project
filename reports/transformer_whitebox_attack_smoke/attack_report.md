# transformer 白盒攻击 Smoke 报告

## 实验设置
- 样本数: `32`
- 资产目录: `/home/chainsmoker/qlib_test/.worktrees/multi-model-whitebox-attack/artifacts/transformer_probe_assets`
- 配置文件: `/home/chainsmoker/qlib_test/.worktrees/multi-model-whitebox-attack/origin_model_pred/Transformer/model/model_config.json`
- 权重文件: `/home/chainsmoker/qlib_test/origin_model_pred/Transformer/model/transformer_state_dict.pt`
- price_epsilon: `0.01`
- volume_epsilon: `0.02`
- pgd_steps: `5`
- pgd_step_size: `0.25`

## Clean Gate
- clean_loss: `0.008377`
- clean_grad_mean_abs: `2.846189e-05`
- clean_grad_finite_rate: `1.000000`
- spearman_to_reference: `0.9970674486803517`
- feature_mae_to_reference: `0.03521132841706276`
- feature_rmse_to_reference: `0.09940407425165176`
- feature_max_abs_to_reference: `0.607471227645874`

## 攻击结果
- FGSM MSE: `0.044397`
- PGD MSE: `0.056549`
- FGSM 平均价格预算使用率: `0.433789`
- PGD 平均价格预算使用率: `0.361694`
- FGSM 平均成交量预算使用率: `1.000000`
- PGD 平均成交量预算使用率: `0.862891`
