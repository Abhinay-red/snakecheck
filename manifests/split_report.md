# Split report

Generated 2026-11-25T08:55:34+00:00. Val/test are FROZEN.

| class | train | val | test | test_external | excluded |
|---|---|---|---|---|---|
| non_venomous | 422 | 91 | 91 | 122 | 29 |
| cobra | 420 | 91 | 91 | 66 | 12 |
| russells_viper | 419 | 91 | 91 | 40 | 8 |
| common_krait | 418 | 91 | 91 | 19 | 8 |
| sawscaled_viper | 419 | 90 | 90 | 7 | 7 |
| not_a_snake | 419 | 91 | 91 | 108 | 46 |
| **total** | 2517 | 545 | 545 | 362 | 110 |

Observers per split: train 622, val 158, test 124, test_external 104
Observer overlap between splits: 0 (checked)

Hard negatives (look-alike harmless snakes): train 313, val 55, test 74, test_external 88
Largest single-observer share of train: non_venomous 5.9%, cobra 6.0%, russells_viper 5.7%, common_krait 6.0%, sawscaled_viper 4.5%, not_a_snake 5.7%

Excluded: external photo shares observer/photo with main pool (51); near duplicate (29); same observer/photo as frozen val (26); same observer/photo as frozen test (4)
