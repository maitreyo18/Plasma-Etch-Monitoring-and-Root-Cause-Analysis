import itertools
import numpy as np
import pandas as pd
from lam9600 import load, features as F, eda, spc, rca, plots
from lam9600.config import RESULTS, KEY_VARS
from lam9600.model import PCAMonitor

OUTCOME_COLS = ["clear_time", "step5_time", "oes_endpoint_idx"]


def save(df, name):
    df.to_csv(RESULTS / f"{name}.csv")


def evaluate(mon, X, meta, block, center):
    sc, Z, T, E = mon.score(X, meta)
    role = meta.loc[sc.index, "role"]
    f = sc[role == "fault"]
    row = {"block": block, "centering": center, "k": mon.k, "n_features": len(mon.cols), "n_train": mon.n, "t2_limit": mon.t2lim, "q_limit": mon.qlim,
           "t2_limit_parametric": mon.t2lim_f, "q_limit_parametric": mon.qlim_jm,
           "faults": len(f), "detected": int(f.alarm.sum()),
           "detected_T2": int((f.t2_ratio > 1).sum()), "detected_Q": int((f.q_ratio > 1).sum()),
           "false_alarm_phase1": sc[role == "phase1"].alarm.mean(),
           "false_alarm_phase2": sc[role == "phase2"].alarm.mean()}
    return row, sc, Z, T


def main():
    RESULTS.mkdir(exist_ok=True)
    meta = load.wafer_meta()
    print("excluded (truncated) wafers:", load.find_excluded())

    m, o = load.load_machine(), load.load_oes()
    r, phase_cols = load.load_rfm()
    fm, fo, fr = F.machine_features(m), F.oes_features(o), F.rfm_features(r, m, phase_cols)
    for n, f in (("machine", fm), ("oes", fo), ("rfm", fr)):
        save(f, f"features_{n}")

    use = meta[meta.role != "excluded"]
    inv = use.join(fm[["clear_time", "step5_time"]]).join(fo.oes_endpoint_idx)
    save(inv, "wafer_inventory")
    both = inv[["clear_time", "oes_endpoint_idx"]].dropna().astype(float)
    print(f"OES endpoint index vs machine clear time, corr = {both.corr().iloc[0, 1]:.2f}")
    t4 = m[m.step == 4].groupby("wafer").time_s.agg(["min", "max"])
    rt = r.groupby("wafer").time_s.agg(["min", "max"]).join(t4, rsuffix="_m").dropna()
    inside = ((rt["min"] >= rt.min_m) & (rt["max"] <= m.groupby("wafer").time_s.max().reindex(rt.index))).mean()
    print(f"RFM sample times inside machine record: {inside:.0%} of wafers")

    # ---- EDA
    df = use.join(fm)
    plots.clear_time(df)
    save(eda.clear_time_trend(df).set_index("experiment"), "eda_clear_time_trend")
    normal = use.index[use.role.isin(["phase1", "phase2"])]
    allf = pd.concat({"machine": fm, "oes": fo, "rfm": fr}, axis=1).dropna()
    allf.columns = [f"{a}:{b}" for a, b in allf.columns]
    nn = normal.intersection(allf.index)
    save(eda.eta_squared(allf.loc[nn].drop(columns=[c for c in allf if c.split(":")[1] in OUTCOME_COLS]),
                         use.loc[nn, "experiment"]).head(30).to_frame("eta_squared"), "eda_experiment_effect")
    plots.corr_heatmap(fm[[c for c in fm if c.endswith("_mean_s4")]].loc[normal.intersection(fm.index)],
                       "eda_corr_machine_step4_means", "Machine step-4 means: correlation (normal wafers)")

    # ---- Univariate SPC
    res = {v: spc.imr(df, v) for v in KEY_VARS}
    for v in KEY_VARS:
        plots.imr_chart(res[v], v)
    allres = pd.concat(res.values())
    save(spc.imr_summary(allres).set_index("var"), "spc_imr_summary")
    fz = allres[allres.role == "fault"].pivot(index="wafer", columns="var", values="z")
    save(fz.join(use.fault), "spc_fault_z")
    save(spc.capability(df).set_index("group"), "spc_capability_clear_time")
    xr = spc.xbar_r_within(m, meta[meta.role != "excluded"],
                           ["He Press", "Cl2 Flow", "BCl3 Flow", "RF Btm Pwr", "TCP Top Pwr", "Pressure"])
    save(xr.groupby(["var", "role"])[["frac_xbar_out", "frac_r_out"]].mean(), "spc_xbar_r_within_summary")
    he = use.index[use.fault == "He Chuck"]
    if len(he):
        e = use.experiment[he[0]]
        lim, sg = spc.xbar_r_limits(m, use, "He Press", e)
        ref = use.index[(use.experiment == e) & (use.role == "phase1")][10]
        plots.xbar_r({ref: sg[ref], he[0]: sg[he[0]]}, "He Press", lim,
                     f"X-bar/R inside wafers, He Press (exp {e}): normal {ref} vs He Chuck {he[0]}")

    # ---- PCA monitoring
    blocks = {"machine": fm, "oes": fo, "rfm": fr}
    X = {k: v.drop(columns=[c for c in OUTCOME_COLS if c in v]) for k, v in blocks.items()}
    X["fused"] = pd.concat({k: X[k] for k in blocks}, axis=1).dropna()
    bmap = pd.Series({f"{a}:{b}": a for a, b in X["fused"].columns})
    X["fused"].columns = [f"{a}:{b}" for a, b in X["fused"].columns]
    rows, models = [], {}
    for name, Xb in X.items():
        Xb = Xb.loc[Xb.index.intersection(use.index)]
        train = Xb.index[meta.loc[Xb.index, "role"] == "phase1"]
        for center in ("experiment", "global"):
            mon = PCAMonitor(center, blocks=bmap if name == "fused" else None).fit(Xb, meta, train)
            mon.calibrate_loo(Xb, meta, train)
            row, sc, Z, T = evaluate(mon, Xb, meta, name, center)
            rows.append(row)
            models[(name, center)] = (mon, sc, Z, T)
            if center == "experiment":
                plots.scree(mon, name)
                plots.monitor_chart(mon, sc, meta, name)
                plots.score_plot(mon, T, meta, name)
                save(sc.join(meta[["experiment", "run_order", "role", "fault"]]), f"pca_scores_{name}")
    summ = pd.DataFrame(rows)
    summ.to_csv(RESULTS / "pca_detection_summary.csv", index=False)
    print(summ.round(3).to_string(index=False))

    # ---- Root cause: contributions per fault wafer (machine block is the most interpretable)
    for name in ("machine", "fused"):
        mon, sc, Z, T = models[(name, "experiment")]
        con = rca.contributions(mon, Z, T)
        rep = rca.fault_report(mon, sc, Z, con, meta)
        rep.to_csv(RESULTS / f"rca_fault_report_{name}.csv", index=False)
        rep[~rep.alarm].to_csv(RESULTS / f"rca_undetected_faults_{name}.csv", index=False)
        top = {w: (meta.fault[w], (con["rbc"].loc[w] / con["rbc"].loc[w].sum()).nlargest(6)) for w in rep.wafer}
        plots.contribution_grid(top, name)
        print(f"{name}: undetected faults {(~rep.alarm).sum()} of {len(rep)}")
    mon, sc, Z, T = models[("fused", "experiment")]
    acc = []
    for name in X:
        _, scb, Zb, _ = models[(name, "experiment")]
        for rep_, clf, target, by in itertools.product(("abs", "signed"), ("centroid", "lda"),
                                                       ("family", "mechanism"), ("experiment", "wafer")):
            cl = rca.classify(Zb, meta, rep_, clf, target, by)
            det = cl.join(scb.alarm).alarm
            acc.append({"block": name, "representation": rep_, "classifier": clf, "target": target,
                        "leave_one_out_by": by, "n": len(cl), "accuracy": cl.correct.mean(),
                        "accuracy_detected_only": cl.correct[det].mean(),
                        "majority_baseline": cl.true.value_counts(normalize=True).max()})
    acc = pd.DataFrame(acc)
    acc.to_csv(RESULTS / "rca_classification_accuracy.csv", index=False)
    best = acc[(acc.representation == "abs") & (acc.leave_one_out_by == "experiment")]
    print(best.round(2).to_string(index=False))
    rca.classify(models[("machine", "experiment")][2], meta).to_csv(RESULTS / "rca_classification_machine_abs_centroid.csv")
    link, rho = rca.clear_time_link(sc, res["clear_time"])
    link.join(meta[["fault", "role"]]).to_csv(RESULTS / "rca_clear_time_link.csv")
    print(f"Spearman(worst alarm ratio, |clear time z|) = {rho:.2f}")


if __name__ == "__main__":
    main()
