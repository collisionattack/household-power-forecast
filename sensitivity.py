"""How much of the weather gain survives an imperfect weather forecast?

The backtest feeds weather models the OBSERVED weather of the 7 forecast days. Real forecasts have errors and archived 2006-2010
forecasts do not exist, so we add Gaussian noise to the forecast-window weather: noise sd = alpha * the series' own daily sd
(temp 6.4 C, sunshine 4.6 h). alpha=1 is a very pessimistic case (error as large as the series' own spread; the noisy "forecast" still correlates ~0.7 with truth).
Noise is independent per day; real forecast errors are autocorrelated.
Training weather stays observed. Output: data/sensitivity.csv (alpha, model, origin, date, actual, pred).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import H, WINDOWS, load, load_params
from models import MODELS, WEATHER

ROOT = Path(__file__).parent
ALPHAS = [0.25, 0.5, 1.0]
NAMES = ["SARIMAX + weather", "Prophet + weather", "LightGBM + weather"]


def run(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    sd = df[WEATHER].std()
    rows = []
    for k in range(WINDOWS):
        end = len(df) - H * (WINDOWS - k)
        train, test = df.kwh.iloc[:end], df.kwh.iloc[end : end + H]
        for alpha in ALPHAS:
            noisy = df.copy()
            rng = np.random.default_rng(1000 * k + int(alpha * 100))  # same noise for every model
            noisy.loc[test.index, WEATHER] += rng.normal(0, 1, (H, len(WEATHER))) * (alpha * sd).to_numpy()
            for name in NAMES:
                pred = MODELS[name][0](train, H, noisy, **params[name])
                rows.append(pd.DataFrame({"alpha": alpha, "model": name, "origin": train.index[-1], "date": test.index,
                                          "actual": test.to_numpy(), "pred": pred.to_numpy()}))
        print(f"window {k + 1}/{WINDOWS}", flush=True)
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    out = run(load(), load_params())
    out.to_csv(ROOT / "data" / "sensitivity.csv", index=False)
    print(out.assign(e=(out.pred - out.actual).abs()).groupby(["model", "alpha"]).e.mean().unstack().round(2))
