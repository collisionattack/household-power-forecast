"""Reliability analysis on top of data/backtest.csv: prediction intervals, paired significance tests, error breakdowns."""
import holidays
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

CALIB = 12  # weeks of past backtest errors needed before an interval is issued


BASELINE = "Seasonal naive"

# What each number means, shown in the app and the notebook. Units: kWh per day unless stated.
METRIC_HELP = {
    "MAE": "Ortalama mutlak hata: tahminin gerçek günlük tüketimden ortalama kaç kWh saptığı. Yönü (fazla/eksik) önemsemez. Ana metrik budur.",
    "RMSE": "Hataların karesinin ortalamasının karekökü. MAE gibi kWh cinsindendir ama büyük hatalara daha çok ceza verir; RMSE, MAE'den ne kadar büyükse hatalar o kadar uçlara yayılmıştır.",
    "MAE %": "MAE'nin ortalama günlük tüketime oranı. Ölçekten bağımsız okumak içindir: %18, tahminin tipik olarak ortalama günün %18'i kadar şaştığı demektir.",
    "sMAPE %": "Simetrik yüzde hata: 200 × |hata| / (|tahmin| + |gerçek|). Düşük tüketimli günlerde (yaz, tatil) şişebilir, bu yüzden yardımcı metrik.",
    "İyileşme %": "Baseline'a (geçen haftanın aynısı) göre MAE'deki azalma: 100 × (1 − MAE / MAE_baseline). Modelin gerçek katkısı budur.",
    "Yanlılık": "Ortalama işaretli hata (tahmin − gerçek). 0'a yakınsa model sistematik olarak fazla ya da eksik tahmin etmiyor; eksi değer eksik tahmin demektir.",
    "Kapsama": "%80 tahmin aralığının, gerçek değeri içine alma oranı. Hedef %80: çok düşükse aralık fazla dar, çok yüksekse gereksiz geniştir.",
    "Aralık genişliği": "Tahmin aralığının ortalama genişliği (kWh). Aynı kapsamayı daha dar aralıkla veren model daha keskindir.",
}


def abs_err(bt: pd.DataFrame) -> pd.Series:
    return (bt.pred - bt.actual).abs()


def metrics(bt: pd.DataFrame) -> pd.DataFrame:
    """One row per model, sorted by MAE: MAE, RMSE, MAE as % of mean use, sMAPE, skill vs the baseline, bias."""
    e = bt.pred - bt.actual
    g = pd.DataFrame({"MAE": e.abs(), "RMSE": e**2, "sMAPE %": 200 * e.abs() / (bt.pred.abs() + bt.actual.abs()),
                      "Yanlılık": e, "model": bt.model}).groupby("model").mean()
    g["RMSE"] **= 0.5
    g["MAE %"] = 100 * g.MAE / bt.actual.mean()
    g["İyileşme %"] = 100 * (1 - g.MAE / g.MAE[BASELINE])
    return g[["MAE", "RMSE", "MAE %", "sMAPE %", "İyileşme %", "Yanlılık"]].sort_values("MAE").round(2)


def add_intervals(bt: pd.DataFrame, level: float = 0.8, calib: int = CALIB) -> pd.DataFrame:
    """Split-conformal-style band: pred +/- the `level` quantile of |error| over all EARLIER weekly windows of the same model.

    Pooled over the 7 horizons and expanding, so every band uses only errors that were known when the forecast was made.
    Weeks before `calib` get NaN. shortcut: symmetric band, pooled horizons; make it per-horizon if more history is available.
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
    """For each (a, b): is b better than a? Paired over weekly windows (non-overlapping, so ~independent).

    diff = MAE_a - MAE_b per week (positive = b better). Wilcoxon signed-rank p, bootstrap 95% CI of the mean diff,
    share of weeks b wins, Holm-adjusted p across all pairs.
    """
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
    fr = holidays.country_holidays("FR", years=range(2006, 2012))
    d = bt.assign(err=abs_err(bt))
    d["horizon"] = (d.date - d.origin).dt.days
    d["weekday"] = d.date.dt.dayofweek
    d["month"] = d.date.dt.month
    d["holiday"] = d.date.map(lambda x: x in fr)
    d["imputed"] = d.date.map(daily.imputed)
    return {k: d.pivot_table(index=k, columns="model", values="err", aggfunc="mean") for k in
            ["horizon", "weekday", "month", "holiday", "imputed"]}


def worst_weeks(bt: pd.DataFrame, model: str, n: int = 5) -> pd.DataFrame:
    w = weekly_mae(bt)[model].sort_values(ascending=False).head(n)
    return w.rename("MAE").reset_index()


def bias(bt: pd.DataFrame) -> pd.Series:
    """Mean signed error (pred - actual) per model; ~0 means no systematic over- or under-forecast."""
    return (bt.pred - bt.actual).groupby(bt.model).mean()
