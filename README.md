# earned-influence

A minimal illustrative simulation accompanying the paper *Earned Influence as a Contribution-Based Basis for Vote-Weighting: Normative and Computational Questions* (Luis Razo, Augmented Democracies workshop, IJCAI-ECAI 2026).

## What this is, and what it is not

This is a separate Bernoulli-voting toy model. It shows how a comparison between contribution-weighted voting and equal weighting can be computed under one set of stipulated assumptions.

- It does **not** test the paper's empirical premise, that people who contribute more are quicker to correct course after an error. The model stipulates that correcting an error improves an agent's competence, and it sets the dependence between contribution and the propensity to correct as an adjustable parameter. Its results therefore cannot be evidence for that dependence.
- It does **not** implement the experiment proposed in the paper (Figure 1): there are no language-model agents, no survey-calibrated preferences, no collective stocks and no comparison with other aggregation rules.
- It does **not** model real collective decision-making: votes are draws from individual probabilities of being right, and there is no deliberation, information, turnout, informal influence or strategy.
- Its numbers are outputs of the chosen parameters, not findings about real electorates.

## The model

Each round an electorate of N agents decides a binary issue with a correct answer. Agent *i* votes correctly with probability *p_i*. After a wrong vote, the agent either corrects (probability *s_i*; *p_i* moves toward 1) or entrenches (*p_i* moves toward a floor, never upward). A correct vote changes nothing. Every SHIFT rounds the world changes and competence is reset near chance; contribution and correction propensity stay fixed.

Contribution is *x_i* = AVTR_i · t_i in [0, 1], with the contribution rate and the duration share drawn independently. A rule with cap K gives weight 1 + (K − 1) *x_i*. K = 1 is equal weighting; **K = 2 is the earned-influence weight proposed in the paper**; K = 3, 5, 10 and 20 are counterfactual comparisons with more unequal weights. All rules are scored on the same votes. A collective decision is correct if the weight on the correct answer exceeds half the total; ties count as incorrect.

The dependence between contribution and correction propensity is set by `dep`, a latent parameter of a rank-based Gaussian-copula construction. **`dep` is not the correlation.** The script reports the mean, across simulated populations, of the within-population Pearson and Spearman correlations between contribution and correction propensity (for example, `dep` = 0.9 gives a Pearson correlation of about 0.80). Any threshold in `dep` is therefore not a threshold in the correlation the paper calls ρ.

## Results

Run on Python 3.12.3, NumPy 2.4.6, SciPy 1.15.2. Full outputs, standard errors, seeds and versions are in `results/`.

### Main sweep (base parameters, 400 simulated populations per setting)

Difference in the share of correct collective decisions relative to equal weighting, in percentage points (standard error in brackets). Equal weighting is right about 93% of the time at these parameters.

| `dep` | Pearson r | cap 2 | cap 3 | cap 5 | cap 10 | cap 20 |
|---|---|---|---|---|---|---|
| −0.9 | −0.80 | −3.58 (0.06) | −7.74 (0.10) | −16.38 (0.18) | −32.27 (0.25) | −45.83 (0.24) |
| −0.5 | −0.44 | −1.94 (0.05) | −3.92 (0.10) | −7.91 (0.17) | −15.67 (0.27) | −23.58 (0.35) |
| 0.0 | 0.00 | −0.15 (0.03) | −0.39 (0.05) | −1.07 (0.08) | −2.55 (0.14) | −4.39 (0.19) |
| 0.1 | 0.09 | +0.16 (0.03) | +0.14 (0.05) | −0.07 (0.07) | −0.96 (0.11) | −2.13 (0.15) |
| 0.2 | 0.18 | +0.50 (0.03) | +0.76 (0.05) | +0.88 (0.07) | +0.60 (0.10) | −0.03 (0.12) |
| 0.5 | 0.44 | +1.22 (0.04) | +1.96 (0.06) | +2.65 (0.09) | +3.06 (0.11) | +3.07 (0.12) |
| 0.9 | 0.80 | +1.99 (0.04) | +3.00 (0.06) | +3.82 (0.07) | +4.36 (0.08) | +4.48 (0.09) |

The full grid (`dep` from −0.9 to 0.9 in steps of 0.1) is in `results/results_main.json`.

Realised weights are far below the caps: under cap 2 the mean weight is 1.12 (maximum about 1.9), under cap 20 it is 3.21. The effective electorate, (Σw)²/Σw², is 99% of N under cap 2 and 68% under cap 20.

### What the outputs show, within this model

1. **Sign pattern.** In the base run and in each of the twelve one-at-a-time parameter variants examined (see below), cap 2 loses at `dep` = −0.9 and gains at `dep` = +0.9, and cap 10 gives larger absolute differences in both directions. This does not establish robustness to simultaneous parameter changes or to other model structures.
2. **Sign changes.** Where a difference changes sign between adjacent grid points, its location is linearly interpolated. These are exploratory, resolution-dependent estimates without uncertainty intervals, not established thresholds. In the base run they lie at `dep` ≈ 0.05 (cap 2), 0.07 (cap 3), 0.11 (cap 5), 0.16 (cap 10) and 0.20 (cap 20). In the twelve sensitivity variants, each shows one detected sign change for cap 2, interpolated between about −0.003 and 0.094.
3. **Wider caps.** In the base run, sign changes move toward stronger positive dependence as the cap increases. At the strongest positive dependence examined, gains grow only slightly from cap 10 to cap 20, while losses under negative dependence keep growing. At weak positive dependence, higher caps can do worse than equal weighting (at `dep` = 0.1, cap 10 is −0.96 against +0.16 for cap 2). This favours a low cap within the model; it is not a general result.
4. **Asymmetry.** Losses under strong negative dependence are larger than gains under equally strong positive dependence (for cap 2, −3.58 against +1.99 at `dep` = ∓0.9). Within this model this reflects several features together: the stipulated competence dynamics move influence toward whichever agents the assumed dependence makes more or less competent; equal weighting is already right about 93% of the time, which limits possible gains; and unequal weights reduce the effective electorate, which costs a little even when weights carry no information (every cap is slightly below equal weighting at `dep` = 0). The simulation does not separate these mechanisms. The asymmetry means the case for adopting the rule rests on how confident one can be that the dependence is clearly positive, which is an empirical question this code cannot answer.
5. **Magnitudes are fragile.** Effect sizes change by more than an order of magnitude with single parameter changes, partly because baseline accuracy changes with them.

### Sensitivity (200 populations per setting, one parameter changed at a time)

| Change from base | Equal-weighting accuracy (at `dep` = 0) | cap 2, `dep` −0.9 | cap 2, `dep` +0.9 | cap 10, `dep` −0.9 | cap 10, `dep` +0.9 | Sign change, cap 2 |
|---|---|---|---|---|---|---|
| N = 21 | 75.3% | −2.25 | +1.51 | −20.13 | +9.15 | 0.078 |
| N = 51 | 85.1% | −4.15 | +2.87 | −27.50 | +8.07 | 0.059 |
| N = 201 | 97.6% | −1.84 | +0.75 | −33.77 | +1.34 | 0.051 |
| floor 0.05 | 69.9% | −11.52 | +9.36 | −46.16 | +25.23 | −0.003 |
| floor 0.40 | 98.3% | −0.29 | +0.11 | −4.12 | +0.15 | 0.051 |
| starting competence 0.50 | 84.2% | −5.10 | +3.36 | −37.12 | +8.65 | 0.013 |
| starting competence 0.65 | 98.8% | −1.09 | +0.51 | −17.61 | +0.85 | 0.031 |
| reset every 10 rounds | 88.0% | −1.56 | +1.04 | −11.47 | +2.34 | 0.094 |
| reset every 100 rounds | 95.1% | −3.15 | +1.67 | −37.90 | +3.26 | −0.001 |
| learning rate 0.03 | 88.9% | −2.33 | +1.55 | −15.95 | +3.99 | 0.054 |
| learning rate 0.20 | 96.3% | −2.22 | +1.06 | −32.51 | +2.05 | 0.031 |
| contribution rate Beta(2, 2) | 92.5% | −6.07 | +2.90 | −35.36 | +4.64 | 0.057 |

Base parameters: N = 101, T = 400 rounds, reset every 50 rounds, learning rate 0.10, starting competence 0.55, floor 0.20, contribution rate Beta(1.2, 4). All caps and standard errors are in `results/results_sensitivity.json`.

## Reproducing

```
pip install -r requirements.txt
python earned_influence_toy.py main          # about 45 seconds
python earned_influence_toy.py sensitivity   # about 5 minutes
python -m pytest test_earned_influence_toy.py
```

Seeds are fixed, so the outputs above are reproduced exactly with the stated versions.

## Licence and citation

Code: MIT (`LICENSE`). Results in `results/`: CC BY 4.0 (`LICENSE-data`). If you use this code, please cite the paper; citation metadata is in `CITATION.cff`.
