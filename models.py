"""Forecasters, one contract: fn(y, h, x, **params) -> Series of the h days after y.

y: daily kWh up to the origin. x: weather for ALL dates (train + future).
Future weather is the observed value: this measures "how much does weather help",
live you would feed in a weather forecast instead (Open-Meteo has a forecast endpoint).
"""
import itertools
import logging
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

WEATHER = ["temp", "sunshine_h"]


def _future(y: pd.Series, h: int) -> pd.DatetimeIndex:
    return pd.date_range(y.index[-1] + pd.Timedelta("1D"), periods=h, freq="D")


def naive_seasonal(y, h=7, x=None):
    """Repeat the last observed week."""
    return pd.Series(y.iloc[-7:].to_numpy()[:h], index=_future(y, h))


def _sarimax(y, h, x, order, seasonal_order, use_weather):
    f = _future(y, h)
    ex_tr, ex_f = (x.loc[y.index, WEATHER], x.loc[f, WEATHER]) if use_weather else (None, None)
    fit = SARIMAX(y.asfreq("D"), exog=ex_tr, order=order, seasonal_order=seasonal_order).fit(disp=False, maxiter=50)
    return pd.Series(fit.forecast(h, exog=ex_f).to_numpy(), index=f)


def _prophet(y, h, x, use_weather, **kw):
    from prophet import Prophet

    m = Prophet(daily_seasonality=False, **kw)
    m.add_country_holidays("FR")
    df = pd.DataFrame({"ds": y.index, "y": y.to_numpy()})
    f = pd.DataFrame({"ds": _future(y, h)})
    if use_weather:
        for c in WEATHER:
            m.add_regressor(c)
            df[c], f[c] = x.loc[y.index, c].to_numpy(), x.loc[f.ds, c].to_numpy()
    m.fit(df)
    return pd.Series(m.predict(f).yhat.to_numpy(), index=f.ds)


def _lgbm(y, h, x, use_weather, **kw):
    """Direct multi-horizon GBM: one model over (origin, k) rows, k = days ahead is a feature.

    Features known at the origin: last 7 days, 28-day mean, target weekday/month/holiday (+ weather of the target day).
    """
    import holidays
    import lightgbm as lgb

    fr = holidays.country_holidays("FR", years=range(2006, 2012))
    lags = pd.concat({f"lag{i}": y.shift(i) for i in range(7)}, axis=1).assign(mean28=y.rolling(28).mean())
    n = len(y)

    def rows(k, origins):
        r = lags.iloc[origins].copy()
        date = y.index[origins] + pd.Timedelta(days=k)
        r["k"], r["dow"], r["month"] = k, date.dayofweek, date.month
        r["holiday"] = [int(d in fr) for d in date]
        if use_weather:
            r[WEATHER] = x.loc[date, WEATHER].to_numpy()
        return r

    train = [rows(k, np.arange(27, n - k)).assign(target=y.to_numpy()[27 + k :]) for k in range(1, h + 1)]
    train = pd.concat(train)
    m = lgb.LGBMRegressor(n_estimators=300, verbose=-1, n_jobs=4, random_state=0, **kw)
    m.fit(train.drop(columns="target"), train.target)
    test = pd.concat([rows(k, np.array([n - 1])) for k in range(1, h + 1)])
    return pd.Series(m.predict(test), index=_future(y, h))


# name -> (fn, param grid). Grids are searched in tune.py on data BEFORE the backtest window.
_LGBM_GRID = {"num_leaves": [7, 15], "learning_rate": [0.03, 0.1], "min_child_samples": [20, 50]}
MODELS = {
    "Seasonal naive": (naive_seasonal, {}),
    "SARIMA": (lambda y, h, x, order, seasonal_order: _sarimax(y, h, x, order, seasonal_order, False),
               {"order": [(p, 0, q) for p in (0, 1, 2) for q in (0, 1, 2)],
                "seasonal_order": [(P, 1, Q, 7) for P in (0, 1) for Q in (0, 1)]}),
    "SARIMAX + weather": (lambda y, h, x, order, seasonal_order: _sarimax(y, h, x, order, seasonal_order, True),
                          {"order": [(p, 0, q) for p in (1, 2) for q in (0, 1)],
                           "seasonal_order": [(P, 1, Q, 7) for P in (0, 1) for Q in (0, 1)]}),
    "Prophet": (lambda y, h, x, **kw: _prophet(y, h, x, False, **kw),
                {"changepoint_prior_scale": [0.01, 0.05, 0.2], "seasonality_prior_scale": [1, 10],
                 "seasonality_mode": ["additive", "multiplicative"]}),
    "Prophet + weather": (lambda y, h, x, **kw: _prophet(y, h, x, True, **kw),
                          {"changepoint_prior_scale": [0.01, 0.05, 0.2], "seasonality_prior_scale": [1, 10],
                           "seasonality_mode": ["additive", "multiplicative"]}),
    "LightGBM": (lambda y, h, x, **kw: _lgbm(y, h, x, False, **kw), _LGBM_GRID),
    "LightGBM + weather": (lambda y, h, x, **kw: _lgbm(y, h, x, True, **kw), _LGBM_GRID),
}


def grid(space: dict):
    keys = list(space)
    return [dict(zip(keys, v)) for v in itertools.product(*space.values())] or [{}]
