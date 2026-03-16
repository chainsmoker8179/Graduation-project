# 部分股票白盒攻击回测实验报告

## 攻击设置

- selected_count: 49812
- selected_available_count: 1
- selected_missing_count: 49811
- attackable_count: 0
- selected_ratio: 0.050093
- attackable_ratio: 0.000000

## 四组主指标

### reference_clean

- annualized_excess_return_with_cost: 0.1723255525541322
- max_drawdown_with_cost: -0.09241775031847635
- rank_ic_mean: 0.07140255290169284

### partial_clean

- annualized_excess_return_with_cost: 0.1723255525541322
- max_drawdown_with_cost: -0.09241775031847635
- rank_ic_mean: 0.07140255290169284

### partial_fgsm

- annualized_excess_return_with_cost: 0.1723255525541322
- max_drawdown_with_cost: -0.09241775031847635
- rank_ic_mean: 0.07140255290169284

### partial_pgd

- annualized_excess_return_with_cost: 0.1723255525541322
- max_drawdown_with_cost: -0.09241775031847635
- rank_ic_mean: 0.07140255290169284

## 主比较差值

|                                     |   annualized_excess_return_with_cost |   annualized_excess_return_without_cost |   ic_mean |     icir |   information_ratio_with_cost |   information_ratio_without_cost |   max_drawdown_with_cost |   max_drawdown_without_cost |   rank_ic_mean |   rank_icir |
|:------------------------------------|-------------------------------------:|----------------------------------------:|----------:|---------:|------------------------------:|---------------------------------:|-------------------------:|----------------------------:|---------------:|------------:|
| reference_clean                     |                             0.172326 |                                0.219839 |  0.032305 | 0.293294 |                       1.38581 |                          1.76616 |               -0.0924178 |                  -0.0843136 |      0.0714026 |    0.552231 |
| partial_clean                       |                             0.172326 |                                0.219839 |  0.032305 | 0.293294 |                       1.38581 |                          1.76616 |               -0.0924178 |                  -0.0843136 |      0.0714026 |    0.552231 |
| partial_fgsm                        |                             0.172326 |                                0.219839 |  0.032305 | 0.293294 |                       1.38581 |                          1.76616 |               -0.0924178 |                  -0.0843136 |      0.0714026 |    0.552231 |
| partial_pgd                         |                             0.172326 |                                0.219839 |  0.032305 | 0.293294 |                       1.38581 |                          1.76616 |               -0.0924178 |                  -0.0843136 |      0.0714026 |    0.552231 |
| partial_clean_minus_reference_clean |                             0        |                                0        |  0        | 0        |                       0       |                          0       |                0         |                   0         |      0         |    0        |
| partial_fgsm_minus_partial_clean    |                             0        |                                0        |  0        | 0        |                       0       |                          0       |                0         |                   0         |      0         |    0        |
| partial_pgd_minus_partial_clean     |                             0        |                                0        |  0        | 0        |                       0       |                          0       |                0         |                   0         |      0         |    0        |
