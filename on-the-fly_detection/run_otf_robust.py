"""Robust on-the-fly fault detection: run, evaluate, plot.

    cd lam9600_spc/on-the-fly_detection_robust && python run_otf_robust.py

Fault labels are used ONLY to count detections and false alarms and to colour the plots.
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import font_manager
from otf_robust import (load_features, fit_phase1, score_known, run_online, run_oracle, run_fixed, self_start, t2_q)

RES, FIG = HERE / "results", HERE / "figures"
RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
NORMAL, NEW_EXP = ["phase1", "phase2"], 33
BLUE, AQUA, ORANGE, GREY, INK2, PURPLE = "#2a78d6", "#1baf7a", "#eb6834", "#8a8985", "#52514e", "#7d4fc4"
for f in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"):
    if Path(f).exists():
        font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 14, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e1", "legend.frameon": False,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "savefig.dpi": 200})


def counts(alarm, truth):
    f = (truth == "fault").values
    a = alarm.values
    return int(a[f].sum()), int(f.sum()), int(a[~f].sum()), int((~f).sum())


# ---------------------------------------------------------------- 1. Phase I
X, meta = load_features(ROOT)
role, exp = meta.loc[X.index, "role"], meta.loc[X.index, "experiment"]
train = list(X.index[(exp != NEW_EXP) & role.isin(NORMAL)])
model = fit_phase1(X, meta, train)
cum = np.cumsum(model["lam_all"]) / model["lam_all"].sum()
print(f"Phase I: {len(train)} wafers x {len(model['cols'])} features, k = {model['k']} ({100 * cum[model['k'] - 1]:.1f} % variance), "
      f"T2 limit {model['t2lim']:.2f}, Q limit {model['qlim']:.2f}")

# ---------------------------------------------------------------- 2. score every wafer
old = [w for w in X.index if exp[w] != NEW_EXP]
sc_old, E_old = score_known(model, X, meta, old)
sc_old["mode"] = "known Phase I baseline"
E33 = list(meta.loc[X.index[exp == NEW_EXP]].sort_values("run_order").index)
X33 = X.loc[E33]
sc33, E_33 = run_online(model, X33)
sc33["mode"] = "on the fly"
scores = pd.concat([sc_old, sc33]).join(meta[["experiment", "run_order", "role", "fault"]])
scores.to_csv(RES / "otf_scores_all_wafers.csv")
d1, n1, f1, m1 = counts(sc_old.alarm, role[old])
d2, n2, f2, m2 = counts(sc33.alarm, role[E33])
summary = pd.DataFrame([
    {"group": "exps 29 + 31 (known Phase I baseline)", "faults_detected": d1, "faults": n1, "false_alarms": f1, "normals": m1},
    {"group": "exp 33 (on the fly, no labels)", "faults_detected": d2, "faults": n2, "false_alarms": f2, "normals": m2},
    {"group": "all", "faults_detected": d1 + d2, "faults": n1 + n2, "false_alarms": f1 + f2, "normals": m1 + m2}])
summary.to_csv(RES / "otf_summary.csv", index=False)
pd.DataFrame({"n_PCs": np.arange(1, len(cum) + 1), "eigenvalue": model["lam_all"], "cum_var": cum}).to_csv(RES / "phase1_variance.csv", index=False)
print(summary.to_string(index=False))

# ---------------------------------------------------------------- 3. comparisons (exp 33, and each experiment held out)
variants = {
    "uses labels: mean/sd of known normals (reference)": "oracle",
    "no adaptation: Phase I mean and sd": "fixed",
    "running mean + running sd, plain z": dict(t_correct=False),
    "running mean + running sd, only alarm-free wafers added": dict(update="in_ctrl"),
    "running mean + running sd, t-corrected (final)": dict(),
}
rows = []
for held in (33, 29, 31):
    tr = list(X.index[(exp != held) & role.isin(NORMAL)])
    m_ = model if held == NEW_EXP else fit_phase1(X, meta, tr)
    E = list(meta.loc[X.index[exp == held]].sort_values("run_order").index)
    Xe, truth = X.loc[E], role[E]
    for name, kw in variants.items():
        if kw == "oracle":
            r = run_oracle(m_, Xe, [w for w in E if truth[w] in NORMAL])
        elif kw == "fixed":
            Xt = X.loc[tr]
            Xc = pd.concat([Xt[exp[tr] == g] - Xt[exp[tr] == g].mean() for g in exp[tr].unique()])
            r = run_fixed(m_, Xe, Xt.mean(), Xc.std(ddof=1))
        else:
            r = run_online(m_, Xe, **kw)[0]
        d, n, fa, m = counts(r.alarm, truth)
        rows.append({"new_experiment": held, "method": name, "k": m_["k"], "faults_detected": d, "faults": n,
                     "false_alarms": fa, "normals": m,
                     "missed": " ".join(str(w) for w in r.index if truth[w] == "fault" and not r.alarm[w]),
                     "false_alarm_wafers": " ".join(str(w) for w in r.index if truth[w] != "fault" and r.alarm[w])})
comp = pd.DataFrame(rows)
comp.to_csv(RES / "otf_method_comparison.csv", index=False)

# faults placed inside the first 3 wafers (real exp-33 wafers re-ordered; method unchanged)
flt = [w for w in E33 if role[w] == "fault"]
cases = [("actual order", E33)]
for fw in flt:
    rest = [w for w in E33 if w != fw]
    cases.append((f"1 fault among first 3: {fw} ({meta.fault[fw]})", rest[:1] + [fw] + rest[1:]))
for a, b in ((flt[0], flt[3]), (flt[1], flt[5]), (flt[2], flt[4])):
    rest = [w for w in E33 if w not in (a, b)]
    cases.append((f"2 faults among first 3: {a}, {b}", [a, rest[0], b] + rest[1:]))
brows = []
for tag, order in cases:
    r = run_online(model, X.loc[order])[0]
    d, n, fa, m = counts(r.alarm, role[order])
    bf = [w for w in order[:3] if role[w] == "fault"]
    brows.append({"case": tag, "first3_faults_caught": f"{int(r.alarm[bf].sum())}/{len(bf)}",
                  "faults_detected": d, "faults": n, "false_alarms": fa, "normals": m})
pd.DataFrame(brows).to_csv(RES / "otf_start_robustness.csv", index=False)

# Q contributions of every exp-33 alarm
qrows = []
for w in sc33.index[sc33.alarm]:
    e2 = E_33.loc[w] ** 2
    for rank, (f, v) in enumerate(e2.sort_values(ascending=False).head(5).items(), 1):
        qrows.append({"wafer": w, "fault": meta.fault[w], "rank": rank, "feature": f, "e2": v, "share_of_Q": v / e2.sum()})
pd.DataFrame(qrows).to_csv(RES / "otf_q_contributions_exp33_alarms.csv", index=False)

# ---------------------------------------------------------------- 4. figures
ROLE_COLOR = {"phase1": BLUE, "phase2": AQUA, "fault": ORANGE}
ROLE_LABEL = {"phase1": "Phase I normal", "phase2": "Phase II normal", "fault": "Fault"}

# 4a. monitoring chart
d = scores.sort_values(["experiment", "run_order"]).reset_index()
x = np.arange(len(d))
fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
b0 = np.where(d.experiment == NEW_EXP)[0][:3]
for a, col, lim, lab in ((axs[0], "t2", model["t2lim"], "Hotelling T²"), (axs[1], "q", model["qlim"], "Q (SPE)")):
    a.set_yscale("log")
    for r_, c in ROLE_COLOR.items():
        k = d.role.values == r_
        a.scatter(x[k], d[col].values[k], color=c, s=26, zorder=3)
    a.axhline(lim, color=INK2, ls="--", lw=1.2)
    a.set_ylabel(lab, fontsize=16)
    a.axvspan(b0[0] - 0.5, b0[-1] + 0.5, color="#d9d9d9", alpha=0.6, lw=0, zorder=0)
for e_, g in d.groupby("experiment"):
    for a in axs:
        a.axvline(g.index.min() - 0.5, color=GREY, lw=1)
    axs[0].text(g.index.min() + 1, axs[0].get_ylim()[1] * 0.55, f"exp {e_}", color=INK2, fontsize=14, va="top")
axs[1].text(b0.mean(), 3e3, "start", color=INK2, fontsize=11, ha="center", va="center", rotation=90)
axs[1].set_xlabel("Wafers in run order, grouped by experiment", fontsize=16)
h = [Line2D([], [], marker="o", ls="", color=c, label=ROLE_LABEL[r_], ms=8) for r_, c in ROLE_COLOR.items()]
h.append(Line2D([], [], color=INK2, ls="--", label="99% limit"))
axs[1].legend(handles=h, loc="upper right", fontsize=13, frameon=True, framealpha=0.9, edgecolor="#cccccc")
axs[0].set_title("On-the-fly monitoring: exp 33 standardised from its own earlier wafers, no labels", fontsize=14)
fig.tight_layout(); fig.savefig(FIG / "otf_monitor_machine.png", bbox_inches="tight"); plt.close(fig)

# running reference of each exp-33 wafer (as used by run_online with update='all')
def ref_of(i):
    return [b for b in E33[:3] if b != E33[i]] if i < 3 else E33[:i]

# 4b. strip plot: raw -> minus centre -> standardised
feats = ["Pressure_mean_s4", "Vat Valve_mean_s5", "Endpt A_mean_s5"]
EC = {29: BLUE, 31: PURPLE, 33: AQUA}
rng = np.random.default_rng(1)
cen, zed = {}, {}
for w in old:
    cen[w] = X.loc[w, feats] - model["mu"][exp[w]][feats]
    zed[w] = cen[w] / model["sd"][exp[w]][feats]
for i, w in enumerate(E33):
    R = X.loc[ref_of(i), feats]
    cen[w] = X.loc[w, feats] - R.mean()
    zed[w] = self_start(X.loc[w, feats], R, model["sd_pool"])
fig, axs = plt.subplots(3, 3, figsize=(13, 9.5))
for i, f in enumerate(feats):
    for j, lab in enumerate(("(a) raw value", "(b) minus centre", "(c) standardised (z)")):
        ax = axs[i, j]
        for e_ in (29, 31, 33):
            W_ = [w for w in X.index if exp[w] == e_]
            v = np.array([X.loc[w, f] if j == 0 else (cen[w][f] if j == 1 else zed[w][f]) for w in W_], float)
            isf = np.array([role[w] == "fault" for w in W_])
            out = (np.abs(v) > 10) if j == 2 else np.zeros(len(v), bool)
            if j == 2:
                v = np.clip(v, -10, 10)
            jit = e_ + rng.uniform(-0.5, 0.5, len(v))
            for mask, col, sz in ((~isf, EC[e_], 16), (isf, ORANGE, 22)):
                ax.scatter(jit[mask & ~out], v[mask & ~out], s=sz, color=col, zorder=3 if col == ORANGE else 2)
                ax.scatter(jit[mask & out], v[mask & out], s=sz + 14, color=col, marker="^", zorder=3)
        ax.set_xticks([29, 31, 33])
        if j == 2:
            ax.axhline(3, color=ORANGE, ls="--", lw=1); ax.axhline(-3, color=ORANGE, ls="--", lw=1); ax.set_ylim(-11, 11)
        if i == 0:
            ax.set_title(lab, fontsize=14)
        if j == 0:
            ax.set_ylabel(f, fontsize=13)
fig.text(0.5, -0.01, "exps 29, 31: own Phase I mean and sd.   exp 33: mean and sd of its earlier wafers, t-corrected (no labels).   "
         "Orange = fault (colouring only).   Triangles: beyond ±10.", ha="center", fontsize=11, color=INK2)
fig.tight_layout(); fig.savefig(FIG / "otf_scaling_strip.png", bbox_inches="tight"); plt.close(fig)

# 4c. how the exp-33 reference is learned: running mean and the band that maps to z = ±3
feats_t = ["Vat Valve_mean_s4", "Pressure_mean_s4", "Endpt A_mean_s5"]
z3 = stats.norm.cdf(3)
nor33 = [w for w in E33 if role[w] in NORMAL]
ro = meta.loc[E33, "run_order"].values
isf = np.array([role[w] == "fault" for w in E33])
fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, f in zip(axs, feats_t):
    c, half = [], []
    for i in range(len(E33)):
        R = X.loc[ref_of(i), f]; n = len(R); s_ = R.std(ddof=1) if R.std(ddof=1) > 0 else model["sd_pool"][f]
        c.append(R.mean()); half.append(stats.t.ppf(z3, n - 1) * s_ * np.sqrt(1 + 1 / n))
    c, half = np.array(c), np.array(half)
    ax.scatter(ro[~isf], X33[f].values[~isf], s=22, color=AQUA, label="normal (unknown to the monitor)", zorder=3)
    ax.scatter(ro[isf], X33[f].values[isf], s=30, color=ORANGE, label="fault (unknown to the monitor)", zorder=3)
    ax.plot(ro, c, color=INK2, lw=1.8, label="running mean")
    ax.fill_between(ro, c - half, c + half, color=GREY, alpha=0.2, lw=0, label="band that maps to z = ±3")
    ax.axhline(X.loc[nor33, f].mean(), color=BLUE, ls=":", lw=1.6, label="mean of true normals (hindsight)")
    ax.axvspan(0.5, 3.5, color="#d9d9d9", alpha=0.6, lw=0, zorder=0)
    lo, hi = np.percentile(X33[f], [1, 99]); pad = (hi - lo) * 0.6
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_title(f, fontsize=14); ax.set_xlabel("run order in exp 33")
axs[0].legend(fontsize=10, loc="upper right")
fig.tight_layout(); fig.savefig(FIG / "otf_baseline_tracking_exp33.png", bbox_inches="tight"); plt.close(fig)

# 4d. method comparison
names = list(variants)
fig, axs = plt.subplots(1, 2, figsize=(15, 5.2), sharey=True)
yy = np.arange(len(names))[::-1]
for ax, (col, tot, ttl) in zip(axs, (("faults_detected", "faults", "Faults detected"), ("false_alarms", "normals", "False alarms on normal wafers"))):
    for held, off, colr in ((33, 0.27, AQUA), (29, 0, BLUE), (31, -0.27, PURPLE)):
        cc = comp[comp.new_experiment == held].set_index("method").loc[names]
        frac = cc[col] / cc[tot]
        ax.barh(yy + off, frac, height=0.25, color=colr, label=f"exp {held} as new experiment ({cc[tot].iloc[0]} {tot})")
        for yv, fr, n_ in zip(yy + off, frac, cc[col]):
            ax.text(fr + 0.01, yv, f"{n_}", va="center", fontsize=10)
    ax.set_title(ttl, fontsize=15); ax.set_xlim(0, 1.12); ax.set_xlabel("fraction")
axs[0].set_yticks(yy); axs[0].set_yticklabels(names, fontsize=12)
axs[1].legend(fontsize=11, loc="lower right", frameon=True, framealpha=0.9, edgecolor="#cccccc")
fig.tight_layout(); fig.savefig(FIG / "otf_method_comparison.png", bbox_inches="tight"); plt.close(fig)

# 4e. root cause: Q contributions of every exp-33 alarm
al = list(sc33.index[sc33.alarm])
nc = 4; nr = int(np.ceil(len(al) / nc))
fig, axs = plt.subplots(nr, nc, figsize=(18, 3.3 * nr), squeeze=False)
for ax, w in zip(axs.flat, al):
    e2 = E_33.loc[w] ** 2; top = e2.sort_values(ascending=False).head(5)
    yv = np.arange(5)[::-1]
    ax.barh(yv, 100 * top.values / e2.sum(), color=BLUE)
    ax.set_yticks(yv); ax.set_yticklabels(top.index, fontsize=10.5)
    lab = meta.fault[w] if meta.fault[w] != "normal" else "normal (false alarm)"
    ax.set_title(f"{w}  {lab}", fontsize=13); ax.set_xlabel("% of Q", fontsize=11)
for ax in list(axs.flat)[len(al):]:
    ax.axis("off")
fig.tight_layout(); fig.savefig(FIG / "otf_q_contributions_exp33.png", bbox_inches="tight"); plt.close(fig)
print("done")
