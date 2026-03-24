# 部分股票白盒攻击回测实验报告

## 攻击设置

- constraint_mode: physical_stat
- tau_ret: 0.005
- tau_body: 0.005
- tau_range: 0.01
- tau_vol: 0.05
- lambda_ret: 0.8
- lambda_candle: 0.4
- lambda_vol: 0.3
- selected_count: 52391
- selected_available_count: 50871
- selected_missing_count: 1520
- attackable_count: 50871
- selected_ratio: 0.050081
- attackable_ratio: 0.048628

## 四组主指标

### reference_clean

- annualized_excess_return_with_cost: 0.35583675732017683
- max_drawdown_with_cost: -0.06930818385370102
- rank_ic_mean: 0.0753396735420719

### partial_clean

- annualized_excess_return_with_cost: 0.338210109602982
- max_drawdown_with_cost: -0.07252457257122874
- rank_ic_mean: 0.07534488060728906

### partial_fgsm

- annualized_excess_return_with_cost: 0.33502563501022564
- max_drawdown_with_cost: -0.09528344095591021
- rank_ic_mean: 0.06426738838219886

### partial_pgd

- annualized_excess_return_with_cost: 0.3052117674944737
- max_drawdown_with_cost: -0.09986788768617735
- rank_ic_mean: 0.06448195268520163

## 主比较差值

|                                     |   annualized_excess_return_with_cost |   annualized_excess_return_without_cost |      ic_mean |         icir |   information_ratio_with_cost |   information_ratio_without_cost |   max_drawdown_with_cost |   max_drawdown_without_cost |   rank_ic_mean |    rank_icir |
|:------------------------------------|-------------------------------------:|----------------------------------------:|-------------:|-------------:|------------------------------:|---------------------------------:|-------------------------:|----------------------------:|---------------:|-------------:|
| reference_clean                     |                           0.355837   |                              0.404298   |  0.0315578   |  0.269899    |                     2.65346   |                        3.01444   |              -0.0693082  |                 -0.0586682  |    0.0753397   |  0.581247    |
| partial_clean                       |                           0.33821    |                              0.386613   |  0.0316166   |  0.270246    |                     2.49362   |                        2.85007   |              -0.0725246  |                 -0.0635702  |    0.0753449   |  0.581171    |
| partial_fgsm                        |                           0.335026   |                              0.383682   |  0.0166274   |  0.153665    |                     2.43095   |                        2.7838    |              -0.0952834  |                 -0.0878146  |    0.0642674   |  0.505759    |
| partial_pgd                         |                           0.305212   |                              0.35355    |  0.0163632   |  0.15148     |                     2.17362   |                        2.51772   |              -0.0998679  |                 -0.0925496  |    0.064482    |  0.506914    |
| partial_clean_minus_reference_clean |                          -0.0176266  |                             -0.0176849  |  5.87751e-05 |  0.000347639 |                    -0.159845  |                       -0.164374  |              -0.00321639 |                 -0.00490202 |    5.20707e-06 | -7.55067e-05 |
| partial_fgsm_minus_partial_clean    |                          -0.00318447 |                             -0.00293086 | -0.0149892   | -0.116582    |                    -0.0626678 |                       -0.0662668 |              -0.0227589  |                 -0.0242444  |   -0.0110775   | -0.0754125   |
| partial_pgd_minus_partial_clean     |                          -0.0329983  |                             -0.0330631  | -0.0152534   | -0.118766    |                    -0.319998  |                       -0.332344  |              -0.0273433  |                 -0.0289794  |   -0.0108629   | -0.0742572   |
