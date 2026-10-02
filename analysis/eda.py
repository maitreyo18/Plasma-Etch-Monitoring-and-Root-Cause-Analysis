import pandas as pd


def eta_squared(X, exp):
    g = X.groupby(exp.values)
    n = g.size()
    ssb = (g.mean().sub(X.mean()) ** 2).mul(n, axis=0).sum()
    sst = ((X - X.mean()) ** 2).sum()
    return (ssb / sst).dropna().sort_values(ascending=False)


def clear_time_trend(df):
    rows = []
    for e, g in df[df.role.isin(["phase1", "phase2"])].groupby("experiment"):
        s = g.sort_values("run_order")
        slope = ((s.run_order - s.run_order.mean()) * (s.clear_time - s.clear_time.mean())).sum() / \
                ((s.run_order - s.run_order.mean()) ** 2).sum()
        rows.append({"experiment": e, "n": len(s), "mean": s.clear_time.mean(), "std": s.clear_time.std(),
                     "slope_s_per_wafer": slope, "corr_with_run_order": s.clear_time.corr(s.run_order),
                     "first_wafer_minus_median": s.clear_time.iloc[0] - s.clear_time.median()})
    return pd.DataFrame(rows)
