import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from .config import HINTS, MECHANISM


def hint(name):
    n = name.split(":")[-1]
    for k, v in HINTS.items():
        if k[0] == "S" and k[1].isdigit():
            if n.startswith(k):
                return v
        elif k in n:
            return v
    return ""


def contributions(mon, Z, T):
    """Classical and reconstruction-based (RBC) contributions to T2 and Q."""
    P, lam = mon.P, mon.lam
    Zv, Tv = Z.values, T.values
    E = Zv - Tv @ P.T
    ct2 = Zv * ((Tv / lam) @ P.T)
    rbc_t2 = ((Tv / lam) @ P.T) ** 2 / (P ** 2 / lam).sum(1)
    rbc_q = E ** 2 / np.clip(1 - (P ** 2).sum(1), 1e-9, None)
    idx, cols = Z.index, Z.columns
    f = lambda a: pd.DataFrame(a, index=idx, columns=cols)
    return {"t2": f(ct2), "q": f(E ** 2), "rbc_t2": f(rbc_t2), "rbc_q": f(rbc_q),
            "rbc": f(rbc_t2 / mon.t2lim + rbc_q / mon.qlim)}


def fault_report(mon, scores, Z, contrib, meta, n_top=5):
    rows = []
    for w in scores.index[meta.loc[scores.index, "role"] == "fault"]:
        top = contrib["rbc"].loc[w].nlargest(n_top)
        share = top / contrib["rbc"].loc[w].sum()
        row = {"wafer": w, "fault": meta.fault[w], "alarm": scores.alarm[w],
               "t2_ratio": scores.t2_ratio[w], "q_ratio": scores.q_ratio[w]}
        for i, (f, s) in enumerate(share.items(), 1):
            row[f"top{i}"] = f"{f} ({Z.loc[w, f] / mon.w[f]:+.1f} sd)"
            row[f"top{i}_share"] = round(float(s), 3)
        row["reading_top1"] = hint(share.index[0])
        rows.append(row)
    return pd.DataFrame(rows)


def _represent(Z, rep):
    c = np.sign(Z) * np.log1p(Z.abs())
    return c.abs() if rep == "abs" else c


def classify(Z, meta, rep="abs", clf="centroid", target="family", by="experiment"):
    """Leave-one-group-out fault classifier. rep: 'abs' (size of deviation) or 'signed'; clf: 'centroid' (cosine) or 'lda'."""
    fw = Z.index[meta.loc[Z.index, "role"] == "fault"]
    S = _represent(Z.loc[fw], rep)
    y = meta.loc[fw, "family"]
    y = y.map(MECHANISM) if target == "mechanism" else y
    g = meta.loc[fw, "experiment"] if by == "experiment" else pd.Series(fw, index=fw)
    pred = {}
    for w in fw:
        tr = fw[g[fw] != g[w]]
        if clf == "centroid":
            cen = S.loc[tr].groupby(y[tr]).mean()
            cos = cen.values @ S.loc[w].values / (np.linalg.norm(cen.values, axis=1) * np.linalg.norm(S.loc[w]) + 1e-12)
            pred[w] = cen.index[int(np.argmax(cos))]
        elif y[tr].nunique() > 1:
            pred[w] = LDA(solver="lsqr", shrinkage="auto").fit(S.loc[tr], y[tr]).predict(S.loc[[w]])[0]
        else:
            pred[w] = None
    out = pd.DataFrame({"fault": meta.loc[fw, "fault"], "true": y, "predicted": pd.Series(pred)})
    out["correct"] = out.true == out.predicted
    return out


def clear_time_link(scores, imr_ct):
    z = imr_ct.set_index("wafer").z
    d = scores.join(z.rename("clear_time_z"), how="inner")
    d["worst_ratio"] = d[["t2_ratio", "q_ratio"]].max(axis=1)
    return d, d[["worst_ratio", "clear_time_z"]].assign(a=lambda x: x.clear_time_z.abs()).corr("spearman").loc["worst_ratio", "a"]
