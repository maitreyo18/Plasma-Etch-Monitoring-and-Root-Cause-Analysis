"""Robust on-the-fly PCA fault detection (LAM 9600 machine data, 78 features).

Standard PCA with Hotelling T2 and Q (SPE). Nothing is tuned:
  Phase I  : each known experiment is mean-centred and scaled by its own normal wafers
             (ordinary autoscaling per experiment); PCA; k by cross-validated PRESS;
             T2 and Q limits = 99th percentile of leave-one-out Phase I scores.
  Phase II : a new experiment resets the mean (Gallagher et al. 1997) and estimates its
             centre and scale from the wafers already seen. The small-sample uncertainty
             of those estimates is handled exactly by the self-starting (Q-chart)
             standardisation of Quesenberry (1991):
                 t_j = (x_j - mean_j) / (sd_j * sqrt(1 + 1/n)),   t_j ~ Student t, n-1 dof
                 z_j = Phi^-1( F_t(t_j; n-1) )                     z_j ~ N(0, 1)
             The z-vector is scored with the frozen Phase I PCA, T2 and Q.
Root cause: per-feature Q contributions e_j^2 (residual of each feature).
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import ndtri_exp

OUTCOME_COLS = ["clear_time", "step5_time"]          # outcomes stored with the features; not model inputs
ALPHA = 0.99
KMAX, CV_FOLDS = 10, 7                               # the codebase's PRESS search (analysis/config.py)


def load_features(root):
    """The 78 per-wafer features of results/features_machine.csv and the wafer table (truncated wafers removed)."""
    from analysis import load
    meta = load.wafer_meta()
    X = pd.read_csv(f"{root}/results/features_machine.csv", index_col=0).drop(columns=OUTCOME_COLS)
    return X.loc[X.index.intersection(meta.index[meta.role != "excluded"])], meta


# ------------------------------------------------------------------ Phase I
def _autoscale(X, exp):
    """Per-experiment mean and sd of the wafers in X (one entry per experiment)."""
    return ({e: X[exp == e].mean() for e in exp.unique()}, {e: X[exp == e].std(ddof=1) for e in exp.unique()})


def _pca(Z, k):
    _, S, Vt = np.linalg.svd(Z, full_matrices=False)
    return Vt[:k].T, S[:k] ** 2 / (len(Z) - 1), S ** 2 / (len(Z) - 1)


def t2_q(model, z):
    """Hotelling T2, Q and the residual vector e of one standardised wafer z."""
    t = z @ model["P"]
    e = z - t @ model["P"].T
    return float((t ** 2 / model["lam"]).sum()), float((e ** 2).sum()), e


def fit_phase1(X, meta, train, k=None):
    """Standard PCA on the known-normal Phase I wafers, per-experiment autoscaling."""
    from analysis.model import press
    Xt, e = X.loc[train], meta.loc[train, "experiment"]
    mu, sd = _autoscale(Xt, e)
    cols = [c for c in X.columns if all(sd[g][c] > 0 for g in sd)]
    Z = np.vstack([((Xt.loc[w, cols] - mu[e[w]][cols]) / sd[e[w]][cols]).values for w in train])
    if k is None:
        k = int(np.argmin(press(Z, min(KMAX, len(Z) // 3), CV_FOLDS))) + 1
    P, lam, lam_all = _pca(Z, k)
    Xc = pd.concat([Xt[e == g][cols] - mu[g][cols] for g in mu])
    model = {"cols": cols, "mu": mu, "sd": sd, "P": P, "lam": lam, "lam_all": lam_all, "k": k, "n": len(train),
             "sd_pool": Xc.std(ddof=1)}                 # pooled within-experiment Phase I sd (used only if a new
                                                        # experiment has not yet shown any variation in a feature)
    # limits: 99th percentile of leave-one-out scores (each Phase I wafer scored by a model fitted without it)
    loo = []
    for w in train:
        rest = [x for x in train if x != w]
        m2, s2 = _autoscale(X.loc[rest], meta.loc[rest, "experiment"])
        Z2 = np.vstack([((X.loc[x, cols] - m2[e[x]][cols]) / s2[e[x]][cols]).values for x in rest])
        P2, l2, _ = _pca(Z2, k)
        z = ((X.loc[w, cols] - m2[e[w]][cols]) / s2[e[w]][cols]).values
        loo.append(t2_q({"P": P2, "lam": l2}, z)[:2])
    loo = np.array(loo)
    model["t2lim"], model["qlim"] = np.quantile(loo[:, 0], ALPHA), np.quantile(loo[:, 1], ALPHA)
    return model


def score_known(model, X, meta, wafers):
    """Wafers of a Phase I experiment, scored against that experiment's known Phase I centre and scale."""
    rows, resid = [], {}
    for w in wafers:
        e = meta.experiment[w]
        z = ((X.loc[w, model["cols"]] - model["mu"][e][model["cols"]]) / model["sd"][e][model["cols"]]).values
        t2, q, r = t2_q(model, z)
        rows.append(_row(model, w, t2, q)); resid[w] = r
    return pd.DataFrame(rows).set_index("wafer"), pd.DataFrame(resid, index=model["cols"]).T


def _row(model, w, t2, q, **extra):
    return {"wafer": w, "t2": t2, "q": q, "t2_ratio": t2 / model["t2lim"], "q_ratio": q / model["qlim"],
            "alarm": bool(t2 > model["t2lim"] or q > model["qlim"]), **extra}


# ------------------------------------------------------------------ Phase II, on the fly
def self_start(x, R, sd_fallback, t_correct=True):
    """Standardise wafer x against reference wafers R (Quesenberry 1991).
    A feature that has shown no variation at all in R (sd = 0, e.g. a reflected power that has
    read exactly 0 so far) cannot be scaled by R; its Phase I spread sd_fallback is used instead.
    t_correct=False gives the plain z = (x - mean) / sd, for comparison only."""
    n = len(R)
    m, s = R.mean(), R.std(ddof=1)
    s = s.where(s > 0, sd_fallback[s.index])
    if not t_correct:
        return (x - m) / s
    t = (x - m) / (s * np.sqrt(1 + 1 / n))
    # z = Phi^-1(F_t(t)), computed on the log scale so that large |t| does not round to +-inf
    z = np.sign(t) * -ndtri_exp(stats.t.logsf(np.abs(t), df=n - 1))
    return pd.Series(z, index=x.index)


def run_online(model, Xe, t_correct=True, update="all"):
    """Monitor one new experiment, wafers in run order, no labels.

    The first 3 wafers (the minimum that gives a standard deviation with 1 degree of freedom
    for each) are judged together, each against the other two. Every later wafer n is judged
    against the wafers 1 .. n-1 of the same experiment.
    update='all'     : every judged wafer is added to the reference (final method)
    update='in_ctrl' : only wafers that did not alarm are added (comparison only)
    """
    cols, W = model["cols"], list(Xe.index)
    rows, resid, ref = [], {}, []

    def judge(w, R, decided_at):
        z = self_start(Xe.loc[w, cols], Xe.loc[R, cols], model["sd_pool"], t_correct)
        z = z.values
        t2, q, r = t2_q(model, z)
        rows.append(_row(model, w, t2, q, n_ref=len(R), decided_at=decided_at)); resid[w] = r
        return rows[-1]["alarm"]

    for w in W[:3]:
        judge(w, [b for b in W[:3] if b != w], W[2])
        ref.append(w)                                   # the first wafers define the new mean (mean reset)
    for w in W[3:]:
        alarm = judge(w, list(ref), w)
        if update == "all" or not alarm:
            ref.append(w)
    return pd.DataFrame(rows).set_index("wafer"), pd.DataFrame(resid, index=cols).T


def run_oracle(model, Xe, normal):
    """Label-using reference: centre and sd from the experiment's KNOWN normal wafers, leaving the
    scored wafer out. Not available on the fly; shown only for comparison."""
    cols, rows = model["cols"], []
    for w in Xe.index:
        R = Xe.loc[[x for x in normal if x != w], cols]
        t2, q, _ = t2_q(model, ((Xe.loc[w, cols] - R.mean()) / R.std(ddof=1)).values)
        rows.append(_row(model, w, t2, q))
    return pd.DataFrame(rows).set_index("wafer")


def run_fixed(model, Xe, centre, scale):
    """No adaptation: every wafer against one fixed centre and scale (comparison only)."""
    cols, rows = model["cols"], []
    for w in Xe.index:
        t2, q, _ = t2_q(model, ((Xe.loc[w, cols] - centre[cols]) / scale[cols]).values)
        rows.append(_row(model, w, t2, q))
    return pd.DataFrame(rows).set_index("wafer")


def score_new_wafer(model, x, R):
    """Live use: judge ONE new wafer x (a Series of the 78 features) against the earlier wafers R
    (DataFrame, >= 2 rows) of the same experiment. Returns T2, Q, alarm and the Q contributions
    (e_j^2, largest first) for root-cause analysis. Afterwards append x to R."""
    z = self_start(x[model["cols"]], R[model["cols"]], model["sd_pool"]).values
    t2, q, e = t2_q(model, z)
    contrib = pd.Series(e ** 2, index=model["cols"]).sort_values(ascending=False)
    return {**_row(model, x.name, t2, q), "q_contributions": contrib}
