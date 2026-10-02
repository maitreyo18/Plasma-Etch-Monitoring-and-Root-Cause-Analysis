import numpy as np
import pandas as pd
from scipy import stats
from .config import ALPHA, KMAX, CV_FOLDS

NORMAL = ["phase1", "phase2"]


def press(Z, kmax, folds, seed=0):
    """Row-wise cross-validated PRESS: each variable is predicted from the others."""
    n, p = Z.shape
    fold = np.random.default_rng(seed).permutation(n) % folds
    out = np.zeros(kmax)
    for f in range(folds):
        tr, te = Z[fold != f], Z[fold == f]
        mu = tr.mean(0)
        Vt = np.linalg.svd(tr - mu, full_matrices=False)[2]
        te = te - mu
        for k in range(1, kmax + 1):
            P = Vt[:k].T
            for j in range(p):
                Pj = np.delete(P, j, axis=0)
                t = np.delete(te, j, axis=1) @ np.linalg.pinv(Pj).T
                out[k - 1] += ((te[:, j] - t @ P[j]) ** 2).sum()
    return out


class PCAMonitor:
    """PCA fitted on Phase I normals; T2 and Q statistics with F / Jackson-Mudholkar limits."""

    def __init__(self, center="experiment", alpha=ALPHA, blocks=None):
        self.center, self.alpha, self.blocks = center, alpha, blocks

    def _centre(self, X, exp):
        mu = pd.DataFrame({e: self.mu[e] for e in exp.unique()}).T
        return X - mu.loc[exp.values].set_axis(X.index)

    def fit(self, X, meta, train, k=None):
        X = X.replace([np.inf, -np.inf], np.nan)
        exp, role = meta.loc[X.index, "experiment"], meta.loc[X.index, "role"]
        if self.center == "experiment":
            self.mu = {e: X[(exp == e) & role.isin(NORMAL)].mean() for e in exp.unique()}
        else:
            g = X.loc[train].mean()
            self.mu = {e: g for e in exp.unique()}
        Xc = self._centre(X, exp)
        sd = Xc.loc[train].std(ddof=1)
        self.cols = sd.index[(sd > 1e-12) & Xc.loc[train].notna().all()]
        self.sd = sd[self.cols]
        w = pd.Series(1.0, index=self.cols)
        if self.blocks is not None:
            b = self.blocks[self.cols]
            w = 1 / np.sqrt(b.map(b.value_counts()))
        self.w = w
        Zt = self._prep(Xc).loc[train].values
        n = len(Zt)
        _, S, Vt = np.linalg.svd(Zt, full_matrices=False)
        self.lam_all = S ** 2 / (n - 1)
        if k is None:
            self.press = press(Zt, min(KMAX, n // 3), CV_FOLDS)
            k = int(np.argmin(self.press)) + 1
        self.k = k
        self.P, self.lam = Vt[:self.k].T, self.lam_all[:self.k]
        self.n = n
        k = self.k
        self.t2lim = self.t2lim_f = k * (n ** 2 - 1) / (n * (n - k)) * stats.f.ppf(self.alpha, k, n - k)
        self.qlim = self.qlim_jm = self._q_limit(self.lam_all[k:n - 1])
        return self

    def calibrate_loo(self, X, meta, train):
        """Replace the parametric limits by the alpha-quantile of leave-one-out (out-of-sample) T2 and Q."""
        stat = []
        for w in train:
            m = PCAMonitor(self.center, self.alpha, self.blocks).fit(X.drop(index=w), meta, train.drop(w), k=self.k)
            s = m.score(X.loc[[w]], meta)[0]
            stat.append((s.t2.iloc[0], s.q.iloc[0]))
        stat = np.array(stat)
        self.loo = pd.DataFrame(stat, index=train, columns=["t2", "q"])
        self.t2lim, self.qlim = np.quantile(stat, self.alpha, axis=0)
        return self

    def _prep(self, Xc):
        return (Xc[self.cols] / self.sd * self.w).fillna(0)

    def _q_limit(self, lam):
        th1, th2, th3 = [(lam ** i).sum() for i in (1, 2, 3)]
        h0 = 1 - 2 * th1 * th3 / (3 * th2 ** 2)
        ca = stats.norm.ppf(self.alpha)
        return th1 * (ca * np.sqrt(2 * th2 * h0 ** 2) / th1 + 1 + th2 * h0 * (h0 - 1) / th1 ** 2) ** (1 / h0)

    def score(self, X, meta):
        X = X.replace([np.inf, -np.inf], np.nan)
        exp = meta.loc[X.index, "experiment"]
        Z = self._prep(self._centre(X, exp))
        T = pd.DataFrame(Z.values @ self.P, index=Z.index)
        E = Z - T.values @ self.P.T
        out = pd.DataFrame({"t2": ((T ** 2) / self.lam).sum(axis=1), "q": (E ** 2).sum(axis=1)})
        out["t2_ratio"], out["q_ratio"] = out.t2 / self.t2lim, out.q / self.qlim
        out["alarm"] = (out.t2_ratio > 1) | (out.q_ratio > 1)
        return out, Z, T, E

    def cum_var(self):
        return np.cumsum(self.lam_all) / self.lam_all.sum()
