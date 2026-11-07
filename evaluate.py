"""Metrics, prediction intervals, paired tests and error breakdowns for data/backtest.csv."""
import numpy as np
import pandas as pd

CALIB = 12  # weeks of past backtest errors needed before an interval is issued
BASELINE = "Seasonal naive"


def abs_err(bt: pd.DataFrame) -> pd.Series:
    return (bt.pred - bt.actual).abs()


def metrics(bt: pd.DataFrame) -> pd.DataFrame:
    """One row per model, sorted by MAE. MAE % is MAE over the mean actual use; Improvement % is the MAE drop vs the baseline."""
    e = bt.pred - bt.actual
    g = pd.DataFrame({"MAE": e.abs(), "RMSE": e**2, "sMAPE %": 200 * e.abs() / (bt.pred.abs() + bt.actual.abs()),
                      "Bias": e, "model": bt.model}).groupby("model").mean()
    g["RMSE"] **= 0.5
    g["MAE %"] = 100 * g.MAE / bt.actual.mean()
    g["Improvement %"] = 100 * (1 - g.MAE / g.MAE[BASELINE])
    return g[["MAE", "RMSE", "MAE %", "sMAPE %", "Improvement %", "Bias"]].sort_values("MAE").round(2)


def add_intervals(bt: pd.DataFrame, level: float = 0.8, calib: int = CALIB) -> pd.DataFrame:
    """Band of pred +/- the `level` quantile of |error| over all earlier weeks of the same model, pooled over the 7 horizons.

    Each band uses only errors known at forecast time. The first `calib` weeks get NaN.
    """
    out = []
    for _, g in bt.assign(err=abs_err(bt)).groupby("model"):
        origins = sorted(g.origin.unique())
        for i, o in enumerate(origins):
            w = g[g.origin == o].copy()
            past = g[g.origin < o].err
            q = past.quantile(level) if i >= calib else np.nan
            w["lo"], w["hi"] = w.pred - q, w.pred + q
            out.append(w)
    return pd.concat(out).drop(columns="err")


def coverage(bt_int: pd.DataFrame) -> pd.DataFrame:
    """Share of actuals inside the band and the mean band width, per model (weeks with a band only)."""
    d = bt_int.dropna(subset=["lo"])
    inside = (d.actual >= d.lo) & (d.actual <= d.hi)
    return pd.DataFrame({"coverage": inside.groupby(d.model).mean(), "width": (d.hi - d.lo).groupby(d.model).mean(),
                         "weeks": d.groupby("model").origin.nunique()})


def weekly_mae(bt: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute error of each 7-day window: rows = origin, columns = model."""
    return bt.assign(err=abs_err(bt)).pivot_table(index="origin", columns="model", values="err", aggfunc="mean")


def paired_tests(bt: pd.DataFrame, pairs: list[tuple[str, str]], seed: int = 0) -> pd.DataFrame:
    """For each (a, b): is b better than a? Paired over weekly windows, assumed independent.

    diff = MAE_a - MAE_b per week, so positive means b is better. Returns the Wilcoxon signed-rank p, a bootstrap 95% CI
    of the mean diff, the share of weeks b wins, and the Holm-adjusted p across all pairs.
    """
    from scipy.stats import wilcoxon
    from statsmodels.stats.multitest import multipletests

    w = weekly_mae(bt)
    rng = np.random.default_rng(seed)
    rows = []
    for a, b in pairs:
        d = (w[a] - w[b]).dropna().to_numpy()
        boot = rng.choice(d, (5000, len(d))).mean(axis=1)
        rows.append({"a": a, "b": b, "mean_diff": d.mean(), "ci_lo": np.quantile(boot, .025), "ci_hi": np.quantile(boot, .975),
                     "b_wins": (d > 0).mean(), "p": wilcoxon(d).pvalue, "weeks": len(d)})
    r = pd.DataFrame(rows)
    r["p_holm"] = multipletests(r.p, method="holm")[1]
    return r


def breakdown(bt: pd.DataFrame, daily: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Mean absolute error by horizon, weekday, month, holiday and imputed-day flag (rows = group, columns = model)."""
    import holidays

    fr = holidays.country_holidays("FR", years=range(2006, 2012))
    d = bt.assign(err=abs_err(bt))
    d["horizon"] = (d.date - d.origin).dt.days
    d["weekday"] = d.date.dt.dayofweek
    d["month"] = d.date.dt.month
    d["holiday"] = d.date.map(lambda x: x in fr)
    d["imputed"] = d.date.map(daily.imputed)
    return {k: d.pivot_table(index=k, columns="model", values="err", aggfunc="mean") for k in
            ["horizon", "weekday", "month", "holiday", "imputed"]}
