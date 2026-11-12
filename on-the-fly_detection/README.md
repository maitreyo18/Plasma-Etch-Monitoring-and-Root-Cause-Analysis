# Robust on-the-fly fault detection (LAM 9600 machine data)

Standard PCA with Hotelling T² and Q on the 78 existing features (`results/features_machine.csv`),
with a **self-starting baseline** for each new experiment and **Q-contribution** root cause.
No labels are used while monitoring, and nothing is tuned on this data set.
Full explanation from scratch, with worked numbers: `OTF_robust_methods.pdf`.

## Method
**Phase I (once, 70 known-normal wafers of exps 29 and 31)**
1. Autoscale every feature within its own experiment (mean and sd of that experiment's normal wafers).
2. PCA; number of components by 7-fold cross-validated PRESS: **k = 10** (54.0 % of variance).
3. T² and Q limits = 99th percentile of leave-one-out Phase I scores: **T² 23.06, Q 113.10**.

**Phase II (new experiment, wafers in run order, no labels)**
1. Mean reset (Gallagher et al. 1997): the new experiment's centre and spread come only from its own earlier wafers.
2. Start-up: wait for 3 wafers and judge each against the other two (the minimum that gives a spread).
3. Wafer n+1 is judged against wafers 1..n: `t = (x − mean)/(sd·√(1+1/n))` ~ Student t (n−1 dof), then
   `z = Φ⁻¹(F_t(t))` (Quesenberry 1991 self-starting Q-chart): an exact small-sample correction, so ±3 always means 0.27 %.
   A feature with sd exactly 0 so far uses its Phase I within-experiment sd.
4. Score z with the frozen PCA: T², Q; alarm if either exceeds its limit. Root cause = e_j²/Q ranked.
5. Add the wafer to the reference, whether or not it alarmed.

Fixed choices: 99 % level, k by PRESS, 3-wafer start-up. No prior weight, floor, forgetting factor or outlier gate.

## Results
| wafers | scaling | faults caught | false alarms |
|---|---|---|---|
| exps 29 + 31 | own Phase I baseline | 14/14 | 1/70 (3101) |
| exp 33 | on the fly, no labels | 6/6 | 1/37 (3331) |
| **all** | | **20/20** | **2/107** |

The earlier on-the-fly method (prior weight 10, spread floor, outlier gate) gave 20/20 and 3/107.

Each experiment treated in turn as the new one (faults caught, false alarms):

| rule | exp 33 (6, 37) | exp 29 (9, 34) | exp 31 (5, 36) |
|---|---|---|---|
| uses labels (not possible live) | 6, 2 | 7, 1 | 4, 1 |
| no adaptation (Phase I mean, sd) | 6, 37 | 9, 34 | 5, 36 |
| running mean + sd, plain z | 6, 13 | 4, 5 | 3, 6 |
| running mean + sd, only alarm-free added | 6, 3 | 3, 0 | 4, 0 |
| **running mean + sd, t-corrected (final)** | **6, 1** | **2, 0** | **3, 0** |

## Limitations
- A fault among the first 3 wafers is not caught and inflates the reference (later detection in exp 33 drops to 2–5 of 6;
  false alarms unchanged). Remedy: start each experiment with metrology-checked qualification wafers.
- Sensitivity grows with run length (a 3-sigma event needs 4.3 s at n = 10, 3.2 s at n = 40), so faults early in a run can be missed
  (held-out exps 29 and 31).
- Equal-weight running mean tracks slow within-experiment drift with lag.
- Assumes roughly normal, independent wafers; some features are autocorrelated.

## Run
```
cd lam9600_spc/on-the-fly_detection_robust
python run_otf_robust.py        # ~15 s; writes results/ and figures/
```
Live use: `score_new_wafer(model, x, R)` in `otf_robust.py` returns T², Q, alarm and Q contributions for one wafer.

## Files
- `otf_robust.py` — method; `run_otf_robust.py` — evaluation and figures; `OTF_robust_methods.pdf` — ground-up methods document
- `results/`: `otf_scores_all_wafers.csv`, `otf_summary.csv`, `otf_method_comparison.csv`, `otf_start_robustness.csv`,
  `otf_q_contributions_exp33_alarms.csv`, `phase1_variance.csv`
- `figures/`: `otf_monitor_machine.png`, `otf_scaling_strip.png`, `otf_baseline_tracking_exp33.png`,
  `otf_method_comparison.png`, `otf_q_contributions_exp33.png`

## References
Gallagher, Wise, Butler, White, Barna, IFAC ADCHEM'97 (1997); Quesenberry, J. Qual. Technol. 23:213 (1991);
Hawkins, The Statistician 36:299 (1987); Wise et al., J. Chemometrics 13:379 (1999); Wold, Technometrics 20:397 (1978);
Jackson & Mudholkar, Technometrics 21:341 (1979); Miller, Swanson, Heckler, Appl. Math. Comp. Sci. 8:775 (1998); Qin, J. Chemometrics 17:480 (2003).
