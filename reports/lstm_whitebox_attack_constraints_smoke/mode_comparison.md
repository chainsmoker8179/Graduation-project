# 三模式对照表

| 模式 | clean_loss | fgsm_loss | pgd_loss | FGSM物理合法 | PGD物理合法 | FGSM严格成功 | PGD严格成功 |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| none | 0.03384351 | 0.13246885 | 0.27077109 | False | False | False | False |
| physical | 0.03384351 | 0.10416338 | 0.14425640 | True | True | True | True |
| physical_stat | 0.03384351 | 0.10416338 | 0.10394776 | True | True | False | True |