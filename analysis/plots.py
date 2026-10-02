import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy import stats
from .config import FIGURES, BLUE, ORANGE, AQUA, GREY

INK, INK2 = "#0b0b0b", "#52514e"
ROLE_COLOR = {"phase1": BLUE, "phase2": AQUA, "fault": ORANGE}
ROLE_LABEL = {"phase1": "Phase I normal", "phase2": "Phase II normal", "fault": "Fault"}
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": GREY,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "font.size": 9, "legend.frameon": False, "savefig.dpi": 140,
})


def _save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def _role_legend(ax, extra=()):
    h = [Line2D([], [], marker="o", ls="", color=c, label=ROLE_LABEL[r]) for r, c in ROLE_COLOR.items()]
    ax.legend(handles=h + list(extra), loc="best")


def clear_time(df):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for e, g in df.groupby("experiment"):
        g = g.sort_values("run_order")
        ax.plot(g.run_order, g.clear_time, color=GREY, lw=1, zorder=1)
        for r, c in ROLE_COLOR.items():
            s = g[g.role == r]
            ax.scatter(s.run_order, s.clear_time, color=c, s=22, zorder=2)
        ax.text(g.run_order.max(), g.clear_time.max() + 0.3, f"exp {e}", color=INK2, ha="right")
    ax.set_xlabel("run order within experiment")
    ax.set_ylabel("clear time (s)")
    ax.set_title("Clear time by run order, per experiment")
    _role_legend(ax)
    _save(fig, "eda_clear_time")


def corr_heatmap(X, name, title):
    c = X.corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(c, cmap="coolwarm", vmin=-1, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(len(c)), c.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(c)), c.columns, fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.7, label="correlation")
    ax.set_title(title)
    _save(fig, name)


def imr_chart(res, var):
    exps = sorted(res.experiment.unique())
    fig, axs = plt.subplots(2, len(exps), figsize=(4.2 * len(exps), 5), sharex="col")
    for j, e in enumerate(exps):
        g = res[res.experiment == e]
        c, s, mrbar = g.center.iloc[0], g.sigma.iloc[0], g.mr_ucl.iloc[0] / 3.267
        a, b = axs[0, j], axs[1, j]
        a.plot(g.run_order, g.x, color=GREY, lw=1)
        for r, col in ROLE_COLOR.items():
            k = g[g.role == r]
            a.scatter(k.run_order, k.x, color=col, s=20, zorder=3)
        a.scatter(g.run_order[g.r1], g.x[g.r1], facecolors="none", edgecolors=INK, s=70, zorder=4)
        for v, ls in ((c, "-"), (c + 3 * s, "--"), (c - 3 * s, "--")):
            a.axhline(v, color=INK2, ls=ls, lw=0.8)
        a.set_title(f"Exp {e}: individuals")
        b.plot(g.run_order, g.mr, color=GREY, lw=1, marker="o", ms=3)
        b.axhline(mrbar, color=INK2, lw=0.8)
        b.axhline(g.mr_ucl.iloc[0], color=INK2, ls="--", lw=0.8)
        b.set_title("moving range")
        b.set_xlabel("run order")
    axs[0, 0].set_ylabel(var)
    h = [Line2D([], [], marker="o", ls="", color=c, label=ROLE_LABEL[r]) for r, c in ROLE_COLOR.items()]
    h.append(Line2D([], [], marker="o", ls="", mfc="none", mec=INK, label="beyond 3σ"))
    fig.legend(handles=h, loc="upper right", ncol=4)
    fig.suptitle(f"I-MR: {var}", x=0.01, ha="left", fontweight="bold")
    _save(fig, f"imr_{var.replace(' ', '_')}")


def xbar_r(sub, var, limits, title):
    """sub: {wafer: (xbar, range)}; limits: (lo, hi, rhi)."""
    fig, axs = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    for (w, (a, r)), col in zip(sub.items(), (BLUE, ORANGE)):
        axs[0].plot(a, marker="o", ms=3, color=col, label=f"wafer {w}")
        axs[1].plot(r, marker="o", ms=3, color=col)
    lo, hi, rhi = limits
    for v in (lo, hi):
        axs[0].axhline(v, color=INK2, ls="--", lw=0.8)
    axs[1].axhline(rhi, color=INK2, ls="--", lw=0.8)
    axs[0].set_ylabel(f"{var} subgroup mean")
    axs[1].set_ylabel("subgroup range")
    axs[1].set_xlabel("subgroup (5 consecutive samples)")
    axs[0].set_title(title)
    axs[0].legend()
    _save(fig, f"xbar_r_{var.replace(' ', '_')}")


def scree(mon, name):
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))
    k = np.arange(1, len(mon.press) + 1)
    axs[0].plot(k, mon.press, marker="o", color=BLUE)
    axs[0].axvline(mon.k, color=ORANGE, ls="--", lw=1, label=f"chosen k = {mon.k}")
    axs[0].set_xlabel("components")
    axs[0].set_ylabel("PRESS (cross-validated)")
    axs[0].legend()
    axs[0].set_title("Component choice")
    cv = mon.cum_var()[:15]
    axs[1].plot(range(1, len(cv) + 1), cv, marker="o", color=BLUE)
    axs[0].set_xticks(k)
    axs[1].set_xticks(range(1, len(cv) + 1, 2))
    axs[1].set_xlabel("components")
    axs[1].set_ylabel("cumulative variance")
    axs[1].set_title("Variance explained")
    fig.suptitle(name, x=0.01, ha="left", fontweight="bold")
    _save(fig, f"pca_scree_{name}")


def monitor_chart(mon, sc, meta, name):
    d = sc.join(meta[["experiment", "run_order", "role"]]).sort_values(["experiment", "run_order"])
    x = np.arange(len(d))
    fig, axs = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    for a, col, lim, lab in ((axs[0], "t2", mon.t2lim, "Hotelling T²"), (axs[1], "q", mon.qlim, "Q (SPE)")):
        a.set_yscale("log")
        for r, c in ROLE_COLOR.items():
            k = d.role.values == r
            a.scatter(x[k], d[col].values[k], color=c, s=18, zorder=3)
        a.axhline(lim, color=INK2, ls="--", lw=0.8)
        a.set_ylabel(lab)
    for e, g in d.reset_index(drop=True).groupby("experiment"):
        for a in axs:
            a.axvline(g.index.min() - 0.5, color=GREY, lw=0.6)
        axs[0].text(g.index.min(), axs[0].get_ylim()[1], f" exp {e}", color=INK2, va="top")
    axs[1].set_xlabel("wafers in run order, grouped by experiment")
    _role_legend(axs[0])
    axs[0].set_title(f"Monitoring: {name} (dashed = {int(mon.alpha * 100)}% limit)")
    _save(fig, f"pca_monitor_{name}")


def score_plot(mon, T, meta, name):
    if mon.k < 2:
        return
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for r, c in ROLE_COLOR.items():
        k = meta.loc[T.index, "role"] == r
        ax.scatter(T[k][0], T[k][1], color=c, s=22, zorder=3)
    s = np.sqrt(mon.lam[:2] * mon.t2lim)
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(s[0] * np.cos(th), s[1] * np.sin(th), color=INK2, ls="--", lw=0.8)
    ax.set_xlabel("PC1 score")
    ax.set_ylabel("PC2 score")
    ax.set_title(f"Scores: {name} (ellipse = T² limit)")
    _role_legend(ax)
    _save(fig, f"pca_scores_{name}")


def contribution_grid(report_top, name):
    """report_top: {wafer: (label, Series of top contributions as share)}."""
    n = len(report_top)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(4.6 * cols, 2.3 * rows))
    axs = np.atleast_1d(axs).ravel()
    for a, (w, (lab, s)) in zip(axs, report_top.items()):
        a.barh(range(len(s))[::-1], s.values, color=ORANGE, height=0.6)
        a.set_yticks(range(len(s))[::-1], [t[:34] for t in s.index], fontsize=6.5)
        a.set_title(f"{w}  {lab}", fontsize=8)
        a.grid(False)
    for a in axs[n:]:
        a.axis("off")
    fig.suptitle(f"Top contributors per fault wafer (RBC share): {name}", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    _save(fig, f"rca_contrib_{name}")
