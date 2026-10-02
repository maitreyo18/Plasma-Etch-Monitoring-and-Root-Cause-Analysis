# LAM 9600 etch: SPC and PCA root-cause workflow

Statistical process control, multivariate (PCA) monitoring and root-cause analysis on the LAM 9600 metal-etch data
(129 wafers, 3 experiments, 21 induced faults; machine, optical-emission and RF-monitor blocks).

## Usage

**Requirements.** The `lam9600` conda env (Python 3.10 with pandas, numpy, scipy, scikit-learn, matplotlib).
The three CSVs must be in `data/`: `MACHINE_Data.csv`, `OES_DATA.csv`, `RFM_DATA.csv`.

**Run everything** (about 15 s):
```
conda activate lam9600
python run_pipeline.py
```
It prints the excluded wafers, two data checks (OES endpoint vs clear time, RFM vs machine timing), the detection
table, the classification table and the clear-time link. CSVs go to `results/`, charts to `figures/`.

**Change a setting.** Edit `analysis/config.py`, then re-run:
- `PHASE1_EXPS`, `PHASE2_EXP`: which experiments build the model and which are monitored.
- `ALPHA` (control-limit level), `KMAX`, `CV_FOLDS`: PCA settings.
- `KEY_VARS`: variables that get I-MR charts.
- `CLEAR_TIME_SPEC = (LSL, USL)`: set to get Cp/Cpk; `None` reports process spread only.
- `Endpt_WINDOW`, `OES_SKIP_FIRST`, `TRUNCATED_FRAC`: feature and exclusion choices.

**Use the pieces in Python.**
```python
from analysis import load, features as F
from analysis.model import PCAMonitor

meta = load.wafer_meta()                      # one row per wafer: experiment, role, fault, family
m = load.load_machine()                       # wide: one row per wafer and sample
fm = F.machine_features(m)                    # one row per wafer
use = meta[meta.role != "excluded"]
X = fm.drop(columns=["clear_time", "step5_time"]).loc[use.index]
train = X.index[meta.loc[X.index, "role"] == "phase1"]

mon = PCAMonitor("experiment").fit(X, meta, train)   # centre per experiment, choose k by cross-validation
mon.calibrate_loo(X, meta, train)                    # leave-one-out control limits
scores, Z, T, E = mon.score(X, meta)                 # t2, q, ratios to limit, alarm flag
```

## Workflow (what `run_pipeline.py` does, in order)

| step | what happens | module | main outputs |
|---|---|---|---|
| 1. Load and clean | Pivot each long CSV to wide form; exclude wafers whose record is truncated (sample count below half the median in any step or block); assign roles: Phase I, Phase II, fault | `load.py` | `wafer_inventory.csv` |
| 2. Features | One row per wafer. Machine: mean and std per channel and step, clear time, Endpt A plateau and slope. OES: main-etch mean of each line and position up to the endpoint, Al/Cl and AlCl/BCl ratios, position asymmetry. RFM: per-step mean and std (phase as sin and cos), harmonic distortion | `features.py` | `features_machine.csv`, `features_oes.csv`, `features_rfm.csv` |
| 3. Exploratory checks | Experiment effect per feature (eta-squared), clear-time drift per experiment, correlation heatmap | `eda.py`, `plots.py` | `eda_experiment_effect.csv`, `eda_clear_time_trend.csv`, `eda_*.png` |
| 4. Univariate SPC | I-MR per experiment with run rules and EWMA; process spread of clear time; X-bar/R on subgroups inside each wafer | `spc.py` | `spc_imr_summary.csv`, `spc_fault_z.csv`, `spc_capability_clear_time.csv`, `spc_xbar_r_within_summary.csv`, `imr_*.png`, `xbar_r_He_Press.png` |
| 5. PCA monitoring | Centre per experiment, autoscale on Phase I, fit PCA, choose k by cross-validated PRESS, T2 and Q, leave-one-out limits. Run on machine, OES, RFM and fused blocks with per-experiment scaling (primary), pooled scaling and global centring, plus late fusion across blocks | `model.py` | `pca_detection_summary.csv`, `pca_scores_<block>.csv`, `pca_scree_*.png`, `pca_monitor_*.png`, `pca_scores_*.png` |
| 6. Root cause | Contributions to T2 and Q (classical and reconstruction-based), top contributors per fault wafer, fault classifier, link to clear time, undetected faults | `rca.py` | `rca_fault_report_<block>.csv`, `rca_undetected_faults_<block>.csv`, `rca_classification_accuracy.csv`, `rca_clear_time_link.csv`, `rca_contrib_*.png` |

Steps 5 and 6 are run for the `machine` and `fused` blocks in the reports; the classifier grid covers all four blocks.

## Layout
| file | job |
|---|---|
| `run_pipeline.py` | runs every step |
| `analysis/config.py` | paths, analysis choices, key variables, interpretation hints |
| `analysis/load.py` | read the CSVs, find truncated wafers, wafer table and roles |
| `analysis/features.py` | one row per wafer: machine, OES, RFM features |
| `analysis/eda.py` | experiment effect, clear-time trend |
| `analysis/spc.py` | I-MR, run rules, EWMA, capability, within-wafer X-bar/R |
| `analysis/model.py` | PCA monitor: T2, Q, cross-validated k, leave-one-out limits |
| `analysis/rca.py` | contributions, fault report, classifier |
| `analysis/plots.py` | all figures |

## What comes from the data, and what is a choice
Derived from the data: excluded wafers, wafer roles (from `set` and `experiment`), phase channels (from `unit`),
features, all control limits, number of components.
Choices (in `config.py`): Phase I = normals of experiments 29 and 31, Phase II = normals of experiment 33; Endpt A
plateau window; first OES spectra skipped; alpha = 0.99; `MECHANISM` grouping (pressure, gas flow, power, He chuck).
The OES endpoint uses the Al lines (394.4 and 395.8 nm); the data supports it (corr 0.78 with machine clear time).
`HINTS` are physical readings used only to label contributors; no statistic uses them.
No spec limits exist in the data, so Cp/Cpk are skipped unless `CLEAR_TIME_SPEC` is set.

## Method notes
- Every feature is centred and scaled per experiment (mean and standard deviation of that experiment's normal wafers;
  the scale is floored at half the Phase I spread). `scale="pooled"` (Phase I spread) is kept for comparison.
  Phase II false alarms are scored with the experiment's centre and scale recomputed without the wafer being scored.
- The F and Jackson-Mudholkar limits were far too tight (70 training wafers, 78 to 498 features), so the reported
  limits are the 99th percentile of leave-one-out T2 and Q on Phase I wafers. Parametric limits stay in the summary.
- Fused model: blocks are weighted so each carries equal total variance.
- `clear_time`, `step5_time` and the OES endpoint index are kept out of PCA (outcomes) and used in SPC and the outcome link.

## Results (see `results/`)
- Experiments differ strongly (`eda_experiment_effect.csv`); clear time drifts down within each experiment.
- PCA, 99% limit (`pca_detection_summary.csv`), per-experiment centring and scaling: machine 20/20 faults, false alarms
  1.4% (Phase I) and 2.7% (Phase II, experiment 33, centre and scale computed without the wafer scored). Machine with
  pooled scaling: 19/20 at 19% Phase II false alarms (the throttle-valve step-5 mean and std are about 4.6 and 2.3 times
  noisier in experiment 33 than in training). Global centring alarms on all Phase II normals.
- Other blocks: OES 6/19, RFM 12/19; fused (one model) 14/19 with per-experiment scaling, 16/19 with pooled scaling;
  late fusion (one model per block, alarm if any block alarms, alpha split over blocks and statistics) 18/20 at 2.9% and 5.4%.
  Adding OES and RFM did not beat machine data alone on these induced faults.
- PRESS keeps falling to the component cap (k = 10), so the component count is limited by `KMAX`, not by a minimum.
- Root cause (`rca_fault_report_machine.csv`): pressure faults point to Vat Valve; TCP faults to Endpt A, TCP Tuner,
  RF Load; He Chuck to He Press spread; BCl3 -5 to BCl3 flow.
- Classification (`rca_classification_accuracy.csv`, leave-one-experiment-out, 20 faults): using the size of each
  deviation instead of its sign lifts the machine block from 0.45 to 0.65 (family; majority baseline 0.30). Mechanism
  grouping with shrinkage LDA gives 0.80 (baseline 0.45); that is the best of 64 reported variants, so it is optimistic.
  RFM and the fused model are weaker or inconsistent (0.11 to 0.53 on family); OES reaches 0.58 but only 6 of its 19 faults alarm.
- Alarm size is not related to the clear-time shift (Spearman 0.07).

## Limits
I-MR run rules on drifting clear time alarm on normal wafers (seasoning); no detrending is applied.
Multiway PCA is not implemented. Experiment-33 limits rely on 70 training wafers from other experiments.
With 19 to 20 faults, detection and accuracy figures carry wide uncertainty.
