# Robustness / ablation summary

## Exp 3

MultiPIC R@20 increases from 0.482 ± 0.026 (m=1) to 0.533 ± 0.018 (m=2), 0.560 ± 0.016 (m=3), and 0.575 (m=4). New-overtaker counts fall from 3573.9 ± 291.6 to 2211 as views increase, while defeated counts rise from 3578.1 ± 162.1 to 3913. This indicates increasing coverage together with changing distractor opportunity, with the net crossing balance improving over this range.

## Exp 10

At N=50/100/200, MultiPIC R@20 is 0.874/0.722/0.575; Calibration is 0.904/0.809/0.630; Calibration + Support is 0.904/0.790/0.660. The false-winner rate rises with pool size for every method; Calibration remains above MultiPIC at all three sizes. N=50/100 use five frozen pool seeds and aggregate seeds within query before aggregating 200 queries.

## Exp 12

OFAT beta, TopK, and alpha sensitivity uses outer-train-only background statistics and fixed Exp 1 fold choices for non-swept parameters. Calibration R@20 spans 0.575–0.665 over beta and 0.595–0.630 over TopK; Support R@20 spans 0.560–0.660 over beta, 0.630–0.660 over TopK, and 0.630–0.670 over alpha. These are sensitivity descriptions, not parameter selection.
