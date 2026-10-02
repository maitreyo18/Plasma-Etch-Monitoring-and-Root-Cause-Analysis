import pandas as pd
from .config import DATA, TRUNCATED_FRAC

FILES = {"machine": "MACHINE_Data.csv", "oes": "OES_DATA.csv", "rfm": "RFM_DATA.csv"}


def find_excluded():
    """Wafers whose record is truncated: far fewer samples than the median in any step or block."""
    bad = set()
    m = pd.read_csv(DATA / FILES["machine"], usecols=["wafer", "step", "variable"])
    c = m[m.variable == m.variable.iloc[0]].groupby(["wafer", "step"]).size().unstack(fill_value=0)
    bad |= set(c.index[(c < TRUNCATED_FRAC * c.median()).any(axis=1)])
    for k in ("oes", "rfm"):
        d = pd.read_csv(DATA / FILES[k], usecols=["wafer", "variable"])
        n = d[d.variable == d.variable.iloc[0]].groupby("wafer").size()
        bad |= set(n.index[n < TRUNCATED_FRAC * n.median()])
    return sorted(bad)


def _read(name, usecols=None):
    df = pd.read_csv(DATA / name, usecols=usecols)
    return df[~df.wafer.isin(find_excluded())]


def load_machine():
    m = _read(FILES["machine"], ["wafer", "sample", "time_s", "step", "variable", "value"])
    w = m.pivot_table(index=["wafer", "sample", "time_s", "step"], columns="variable", values="value")
    return w.reset_index()


def load_oes():
    o = _read(FILES["oes"], ["wafer", "sample", "variable", "value"])
    return o.pivot_table(index=["wafer", "sample"], columns="variable", values="value").reset_index()


def load_rfm():
    r = _read(FILES["rfm"], ["wafer", "sample", "time_s", "variable", "unit", "value"])
    phase = sorted(r.loc[r.unit == "deg.", "variable"].unique())
    w = r.pivot_table(index=["wafer", "sample", "time_s"], columns="variable", values="value")
    return w.reset_index(), phase


def wafer_meta():
    """One row per wafer: experiment, run order, fault label, block availability and role."""
    cols = ["wafer", "experiment", "run_order", "set", "fault"]
    met = {k: pd.read_csv(DATA / f, usecols=cols).drop_duplicates("wafer") for k, f in FILES.items()}
    meta = met["machine"].copy()
    meta["in_oes"] = meta.wafer.isin(met["oes"].wafer)
    meta["in_rfm"] = meta.wafer.isin(met["rfm"].wafer)
    meta["family"] = meta.fault.where(meta.fault != "normal").str.split().str[0]
    meta["role"] = "fault"
    normal = meta.set == "calibration"
    from .config import PHASE1_EXPS, PHASE2_EXP
    meta.loc[normal & meta.experiment.isin(PHASE1_EXPS), "role"] = "phase1"
    meta.loc[normal & (meta.experiment == PHASE2_EXP), "role"] = "phase2"
    meta.loc[meta.wafer.isin(find_excluded()), "role"] = "excluded"
    return meta.sort_values("wafer").set_index("wafer")
