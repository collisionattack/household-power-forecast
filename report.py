"""Prints the tables behind the README and the slides, from the committed data/*.csv."""
from pathlib import Path

import pandas as pd

import evaluate as ev

DATA = Path(__file__).parent / "data"
bt = pd.read_csv(DATA / "backtest.csv", parse_dates=["origin", "date"])
daily = pd.read_csv(DATA / "daily.csv", index_col=0, parse_dates=True)
sens = pd.read_csv(DATA / "sensitivity.csv")
models = [m for m in bt.model.unique() if m != ev.BASELINE]
weather_pairs = [("SARIMA", "SARIMAX + weather"), ("Prophet", "Prophet + weather"), ("LightGBM", "LightGBM + weather")]


def show(title, df):
    print(f"\n{title}\n{df.round(2).to_string()}")


show("Metrics", ev.metrics(bt))
show("80% interval: coverage and width", ev.coverage(ev.add_intervals(bt)))
show("Each model vs the baseline", ev.paired_tests(bt, [(ev.BASELINE, m) for m in models]))
show("Weather vs no weather", ev.paired_tests(bt, weather_pairs))
show("MAE by weekday (0 = Monday)", ev.breakdown(bt, daily)["weekday"])
show("MAE with noisy weather (alpha = noise size)", sens.assign(e=(sens.pred - sens.actual).abs()).groupby(["model", "alpha"]).e.mean().unstack())
