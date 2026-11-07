import numpy as np
import pandas as pd

import evaluate as ev
from backtest import load_params
from data import to_daily
from models import MODELS, grid


def test_to_daily_ignores_missing_minutes_and_fills_gaps():
    idx = pd.date_range("2020-01-01", "2020-01-04 23:59", freq="min")
    raw = pd.DataFrame({"Global_active_power": 1.0, "Sub_metering_1": 0.0, "Sub_metering_2": 0.0, "Sub_metering_3": 0.0}, index=idx)
    raw.loc["2020-01-02 00:00":"2020-01-02 11:59"] = np.nan  # half a day missing -> still 24 kWh
    raw.loc["2020-01-03"] = np.nan  # whole day missing -> imputed
    raw = pd.concat([raw, raw.iloc[-1:].set_axis([raw.index[-1] + pd.Timedelta("1min")])])  # partial day 5, dropped with day 1
    d = to_daily(raw)
    assert list(d.index.strftime("%d")) == ["02", "03", "04"]
    assert d.kwh.round(6).tolist() == [24.0, 24.0, 24.0]
    assert d.imputed.tolist() == [False, True, False]


def _fake_bt(n_weeks=30):
    rng = np.random.default_rng(0)
    rows = []
    for w in range(n_weeks):
        o = pd.Timestamp("2020-01-02") + pd.Timedelta(weeks=w)
        for m, sd in [("Seasonal naive", 4.0), ("A", 2.0)]:
            a = 25 + rng.normal(0, 3, 7)
            rows.append(pd.DataFrame({"origin": o, "model": m, "date": pd.date_range(o + pd.Timedelta("1D"), periods=7),
                                      "actual": a, "pred": a + rng.normal(0, sd, 7)}))
    return pd.concat(rows, ignore_index=True)


def test_intervals_use_only_past_errors():
    bt = _fake_bt()
    bi = ev.add_intervals(bt)
    assert bi[bi.origin < bi.origin.min() + pd.Timedelta(weeks=ev.CALIB)].lo.isna().all()  # no band without history
    # changing a FUTURE week's actuals must not change an earlier week's band
    bt2 = bt.copy(); bt2.loc[bt2.origin == bt2.origin.max(), "actual"] += 100
    a, b = ev.add_intervals(bt2), bi
    early = b.origin < b.origin.max()
    assert np.allclose(a[early].hi.fillna(0), b[early].hi.fillna(0))
    assert 0.65 < ev.coverage(bi).coverage.min() < 0.95


def test_paired_test_finds_a_real_gap_and_baseline_improvement_is_zero():
    bt = _fake_bt()
    t = ev.paired_tests(bt, [("Seasonal naive", "A")]).iloc[0]
    assert t.mean_diff > 0 and t.p_holm < 0.05 and t.ci_lo > 0
    assert ev.metrics(bt).loc["Seasonal naive", "Improvement %"] == 0


def test_models_return_h_future_days():
    idx = pd.date_range("2020-01-01", periods=407)
    x = pd.DataFrame({"temp": np.cos(np.arange(407) / 58), "sunshine_h": 5.0}, index=idx)
    y = pd.Series(20 + 5 * np.sin(np.arange(407) * 2 * np.pi / 7), index=idx)[:400]
    for name, (fn, space) in MODELS.items():
        p = fn(y, 7, x, **grid(space)[0])
        assert len(p) == 7 and p.index[0] == y.index[-1] + pd.Timedelta("1D") and p.notna().all()


def test_saved_params_load_as_tuples():
    assert isinstance(load_params()["SARIMA"]["order"], tuple)
