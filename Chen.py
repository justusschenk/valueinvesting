"""chen_factor_model.py  — v1.3

* Eliminated stray duplicated `__all__` block that caused a `SyntaxError`.
* Self‑contained Fama–MacBeth (`_fama_macbeth_manual`) retained.
* `_load_any` helper restored so the CLI works.

Usage (notebook):
-----------------
```python
from Chen import estimate  # or chen_factor_model depending on filename
factors, betas, (lam, t, path) = estimate(returns_df, macro_df)
```
"""

from __future__ import annotations
import argparse, pathlib, textwrap
from typing import Mapping

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

__all__ = [
    "compute_factors",
    "rolling_betas",
    "estimate",
]

# ---------------------------------------------------------------------------
# 0. Simple *manual* Fama‑MacBeth -------------------------------------------
# ---------------------------------------------------------------------------

def _fama_macbeth_manual(R: pd.DataFrame, B: pd.DataFrame, add_const: bool = True):
    """Return mean risk premia and t‑stats using the two‑pass Fama–MacBeth."""
    if not isinstance(B.columns, pd.MultiIndex):
        raise ValueError("Betas DataFrame must have MultiIndex columns (factor, asset)")

    factors = B.columns.get_level_values(0).unique()
    cols = list(factors)
    if add_const:
        cols = ["const"] + cols

    lambdas = []
    for t in R.index:
        r_t = R.loc[t].to_numpy()
        X_t = np.column_stack([B[f].loc[t].to_numpy() for f in factors])
        if add_const:
            X_t = np.column_stack([np.ones_like(r_t), X_t])
        beta_hat, *_ = np.linalg.lstsq(X_t, r_t, rcond=None)
        lambdas.append(beta_hat)

    lambdas = np.asarray(lambdas)
    lambdas_ts = pd.DataFrame(lambdas, index=R.index, columns=cols)
    lam_mean = lambdas_ts.mean()
    lam_std = lambdas_ts.std(ddof=1)
    T = len(lambdas_ts)
    tstats = lam_mean / (lam_std / np.sqrt(T))
    return lam_mean, tstats, lambdas_ts

# ---------------------------------------------------------------------------
# helpers -------------------------------------------------------------------
# ---------------------------------------------------------------------------

def _to_period_month(idx: pd.Index) -> pd.PeriodIndex:
    if isinstance(idx, pd.PeriodIndex):
        return idx.asfreq("M") if idx.freqstr != "M" else idx
    if isinstance(idx, pd.DatetimeIndex):
        return idx.to_period("M")
    raise TypeError("Index must be DatetimeIndex or PeriodIndex.")

# ---------------------------------------------------------------------------
# 1. Factor construction -----------------------------------------------------
# ---------------------------------------------------------------------------

def compute_factors(macro: pd.DataFrame, colmap: Mapping[str, str] | None = None) -> pd.DataFrame:
    if colmap:
        macro = macro.rename(columns={v: k for k, v in colmap.items()})
    req = ["INDPRO", "CPI", "BAA", "DGS10", "TB3MS"]
    miss = [c for c in req if c not in macro.columns]
    if miss:
        raise ValueError(f"Missing columns: {miss}")

    m = macro.copy().sort_index()
    m.index = _to_period_month(m.index)

    out = pd.DataFrame(index=m.index)
    out["MP"] = np.log(m["INDPRO"]).diff()
    inf = np.log(m["CPI"]).diff()
    exp_inf = inf.rolling(120, min_periods=24).apply(
        lambda x: sm.tsa.ARIMA(x, order=(12, 0, 0)).fit().forecast()[0]
    )
    out["UI"] = inf - exp_inf
    out["DEI"] = exp_inf.diff()
    out["UPR"] = (m["BAA"] - m["DGS10"]).diff()
    out["UTS"] = (m["DGS10"] - m["TB3MS"]).diff()
    return out.dropna()

# ---------------------------------------------------------------------------
# 2. Rolling betas -----------------------------------------------------------
# ---------------------------------------------------------------------------

def rolling_betas(returns: pd.DataFrame, factors: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    aligned = returns.join(factors, how="inner")
    X = sm.add_constant(aligned[factors.columns])
    betas = {}
    for col in returns.columns:
        res = RollingOLS(aligned[col], X, window=window).fit()
        betas[col] = res.params.drop(columns="const")
    return (
        pd.concat(betas, axis=1).swaplevel(axis=1).sort_index(axis=1).dropna()
    )

# ---------------------------------------------------------------------------
from dataclasses import dataclass

@dataclass
class FMResult:
    """Minimal container mimicking a .summary() like linearmodels."""
    lambda_mean: pd.Series
    tstats: pd.Series
    lambda_ts: pd.DataFrame

    def summary(self):
        return pd.DataFrame({"lambda": self.lambda_mean, "t": self.tstats})


def estimate(
    returns: pd.DataFrame,
    macro: pd.DataFrame,
    *,
    window: int = 60,
    colmap: Mapping[str, str] | None = None,
):
    factors = compute_factors(macro, colmap=colmap)
    r = returns.copy()
    r.index = _to_period_month(r.index)
    common = r.index.intersection(factors.index)
    if common.empty:
        raise KeyError("DataFrames have no overlapping monthly dates.")
    r, f = r.loc[common], factors.loc[common]
    betas = rolling_betas(r, f, window=window)
    R, B = r.loc[betas.index], betas
    lam_mean, tstats, lam_ts = _fama_macbeth_manual(R, B)
    fm_res = FMResult(lam_mean, tstats, lam_ts)
    return f, betas, fm_res

# ---------------------------------------------------------------------------
# 4. CLI helpers -------------------------------------------------------------
# ---------------------------------------------------------------------------

def _load_any(path: str | pathlib.Path) -> pd.DataFrame:
    path = pathlib.Path(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    elif path.suffix.lower() in {".pkl", ".pickle"}:
        df = pd.read_pickle(path)
    else:
        raise ValueError("Unsupported file type. Use CSV or pickle.")
    df.index = _to_period_month(df.index)
    return df


def main(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description="CLI – run FMB with local CSV/pickle files")
    p.add_argument("--macro", required=True)
    p.add_argument("--returns", required=True)
    p.add_argument("--window", type=int, default=60)
    args = p.parse_args(argv)

    macro = _load_any(args.macro)
    rets = _load_any(args.returns)
    fac, betas, (lam, t, _) = estimate(rets, macro, window=args.window)
    fac.to_csv("factors.csv"); betas.to_csv("betas.csv")
    summ = pd.DataFrame({"lambda": lam, "t": t})
    summ.to_csv("fama_macbeth_summary.txt", sep="\t")
    print("Fama–MacBeth risk premia:\n", summ)


if __name__ == "__main__":
    main()
