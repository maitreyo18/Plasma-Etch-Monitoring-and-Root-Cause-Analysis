import numpy as np
import pandas as pd
from .config import Endpt_WINDOW, OES_SKIP_FIRST

OES_AL = ("394.4nm", "395.8nm")


def machine_features(m):
    ch = [c for c in m.columns if c not in ("wafer", "sample", "time_s", "step")]
    agg = m.groupby(["wafer", "step"])[ch].agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    f = agg.unstack("step")
    f.columns = [f"{c}_s{st}" for c, st in f.columns]

    dur, plateau = {}, {}
    for w, g in m.groupby("wafer"):
        dt = g.time_s.diff().median()
        s4, s5 = g[g.step == 4], g[g.step == 5]
        d4 = s4.time_s.max() - s4.time_s.min() + dt
        d5 = s5.time_s.max() - s5.time_s.min() + dt if len(s5) else np.nan
        mid = s4["Endpt A"].iloc[int(Endpt_WINDOW[0] * len(s4)):int(Endpt_WINDOW[1] * len(s4))]
        slope = np.polyfit(np.arange(len(mid)), mid, 1)[0] if len(mid) > 2 else np.nan
        dur[w] = (d4, d5)
        plateau[w] = (mid.mean(), slope)
    f["clear_time"] = pd.Series({w: d[0] for w, d in dur.items()})
    f["step5_time"] = pd.Series({w: d[1] for w, d in dur.items()})
    f["Endpt A_plateau"] = pd.Series({w: p[0] for w, p in plateau.items()})
    f["Endpt A_slope"] = pd.Series({w: p[1] for w, p in plateau.items()})
    return f


def _oes_one(g):
    g = g.reset_index(drop=True).drop(columns=["sample"])
    tot = g.sum(axis=1)
    g = g[tot > 0.05 * tot.max()].reset_index(drop=True)          # drop trailing blank spectra
    al = g[[c for c in g.columns if c.startswith(OES_AL)]].mean(axis=1).values
    ip = int(np.argmax(al))
    below = np.where(al[ip:] < 0.5 * al[ip])[0]
    e = ip + int(below[0]) if len(below) else len(g)               # endpoint sample index
    win = g.iloc[OES_SKIP_FIRST:max(e - 1, OES_SKIP_FIRST + 1)]                                   # main-etch plateau window
    f = win.mean()
    for p in (1, 2, 3):
        f[f"ratio_Al_Cl_pos{p}"] = f[f"395.8nm_pos{p}"] / f[f"725nm_pos{p}"]
        f[f"ratio_AlCl_BCl_pos{p}"] = f[f"261.8nm_pos{p}"] / f[f"272.2nm_pos{p}"]
    a1, a3 = f["395.8nm_pos1"], f["395.8nm_pos3"]
    f["Al_asym_pos1_pos3"] = (a1 - a3) / (a1 + a3)
    f["oes_endpoint_idx"] = e
    return f


def oes_features(o):
    rows = {w: _oes_one(g.drop(columns="wafer")) for w, g in o.groupby("wafer")}
    return pd.DataFrame(rows).T


def rfm_features(r, m, phase):
    t4 = m[m.step == 4].groupby("wafer").time_s.max()
    r = r.copy()
    r["step"] = np.where(r.time_s <= r.wafer.map(t4), 4, 5)

    ch = [c for c in r.columns if c not in ("wafer", "sample", "time_s", "step")]
    for c in phase:                                                 # phase wraps: use sin/cos
        rad = np.deg2rad(r[c])
        r[c + "_sin"], r[c + "_cos"] = np.sin(rad), np.cos(rad)
    for s in ("S1", "S2"):                                          # total harmonic distortion
        r[f"{s}_thd"] = np.sqrt(sum(r[f"{s}V{h}"] ** 2 for h in range(2, 6))) / r[f"{s}V1"]
    plain = [c for c in ch if c not in phase]
    trig = [c + t for c in phase for t in ("_sin", "_cos")]

    a = r.groupby(["wafer", "step"])[plain].agg(["mean", "std"])
    a.columns = [f"{c}_{s}" for c, s in a.columns]
    b = r.groupby(["wafer", "step"])[trig + ["S1_thd", "S2_thd"]].mean()
    f = a.join(b).unstack("step")
    f.columns = [f"{c}_s{st}" for c, st in f.columns]
    return f
