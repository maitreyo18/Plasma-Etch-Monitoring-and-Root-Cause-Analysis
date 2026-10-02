import numpy as np
import pandas as pd
from .config import CLEAR_TIME_SPEC

NORMAL = ["phase1", "phase2"]
A2, D4 = 0.577, 2.114          # X-bar/R constants for subgroup size 5


def _run_rules(z):
    z = np.asarray(z)
    r2, r3 = np.zeros(len(z), bool), np.zeros(len(z), bool)
    for i in range(len(z)):
        w = z[max(0, i - 2):i + 1]
        r2[i] = (w > 2).sum() >= 2 or (w < -2).sum() >= 2
        if i >= 7:
            v = z[i - 7:i + 1]
            r3[i] = (v > 0).all() or (v < 0).all()
    return r2, r3


def ewma(z, lam=0.2, L=3):
    e, out, lim = 0.0, [], []
    for i, v in enumerate(z, 1):
        e = lam * v + (1 - lam) * e
        out.append(e)
        lim.append(L * np.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * i))))
    return np.array(out), np.array(lim)


def imr(df, var):
    """I-MR chart per experiment; limits from that experiment's normal wafers."""
    parts = []
    for e, g in df.groupby("experiment"):
        g = g.sort_values("run_order")
        nor = g.loc[g.role.isin(NORMAL), var]
        mrbar = nor.diff().abs().mean()
        c, sigma = nor.mean(), mrbar / 1.128
        z = (g[var] - c) / sigma
        r2, r3 = _run_rules(z.values)
        ew, ewlim = ewma(z.values)
        parts.append(pd.DataFrame({
            "wafer": g.index, "experiment": e, "run_order": g.run_order.values, "role": g.role.values,
            "fault": g.fault.values, "var": var, "x": g[var].values, "center": c, "sigma": sigma,
            "z": z.values, "mr": g[var].diff().abs().values, "mr_ucl": 3.267 * mrbar,
            "r1": (z.abs() > 3).values, "r2": r2, "r3": r3, "ewma": ew, "ewma_lim": ewlim,
        }))
    out = pd.concat(parts)
    out["ewma_flag"] = out.ewma.abs() > out.ewma_lim
    out["mr_flag"] = out.mr > out.mr_ucl
    out["any_rule"] = out[["r1", "r2", "r3", "ewma_flag", "mr_flag"]].any(axis=1)
    return out


def imr_summary(res):
    rows = []
    for (v, e), g in res.groupby(["var", "experiment"]):
        n, f = g[g.role.isin(NORMAL)], g[g.role == "fault"]
        rows.append({"var": v, "experiment": e, "n_normal": len(n), "n_fault": len(f),
                     "false_alarm_3sigma": int(n.r1.sum()), "false_alarm_any": int(n.any_rule.sum()),
                     "detected_3sigma": int(f.r1.sum()), "detected_any": int(f.any_rule.sum())})
    return pd.DataFrame(rows)


def capability(df, var="clear_time", spec=CLEAR_TIME_SPEC):
    """Process spread per experiment and pooled; Cp/Cpk only if spec limits are supplied."""
    rows = []
    for e, g in df[df.role.isin(NORMAL)].groupby("experiment"):
        x = g.sort_values("run_order")[var]
        mu, s = x.mean(), x.diff().abs().mean() / 1.128
        r = {"group": e, "mean": mu, "sigma_within": s, "spread_6sigma": 6 * s}
        if spec:
            r |= {"Cp": (spec[1] - spec[0]) / (6 * s), "Cpk": min(spec[1] - mu, mu - spec[0]) / (3 * s)}
        rows.append(r)
    x = df.loc[df.role.isin(NORMAL), var]
    mu, s = x.mean(), x.std()
    r = {"group": "pooled", "mean": mu, "sigma_within": s, "spread_6sigma": 6 * s}
    if spec:
        r |= {"Pp": (spec[1] - spec[0]) / (6 * s), "Ppk": min(spec[1] - mu, mu - spec[0]) / (3 * s)}
    rows.append(r)
    return pd.DataFrame(rows)


def subgroups(g, var, step=4, n=5):
    x = g.loc[g.step == step, var].values
    x = x[:len(x) // n * n].reshape(-1, n)
    return x.mean(1), np.ptp(x, axis=1)


def xbar_r_limits(m, meta, var, exp, step=4, n=5):
    """Limits from the normal wafers of one experiment; also returns every wafer's subgroups."""
    mm = m[m.wafer.map(meta.experiment) == exp]
    sg = {w: subgroups(g, var, step, n) for w, g in mm.groupby("wafer")}
    wn = [w for w in sg if meta.role[w] in NORMAL]
    xb = np.concatenate([sg[w][0] for w in wn])
    rb = np.concatenate([sg[w][1] for w in wn]).mean()
    return (xb.mean() - A2 * rb, xb.mean() + A2 * rb, D4 * rb), sg


def xbar_r_within(m, meta, vars_, step=4, n=5):
    """Share of within-wafer subgroups outside X-bar / R limits, per wafer."""
    rows = []
    for var in vars_:
        for e in sorted(meta.experiment.unique()):
            (lo, hi, rhi), sg = xbar_r_limits(m, meta, var, e, step, n)
            for w, (a, r) in sg.items():
                rows.append({"wafer": w, "experiment": e, "var": var, "role": meta.role[w],
                             "frac_xbar_out": float(((a < lo) | (a > hi)).mean()),
                             "frac_r_out": float((r > rhi).mean())})
    return pd.DataFrame(rows)
