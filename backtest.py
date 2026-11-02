"""Rolling-origin backtest: last 52 weeks, a new 7-day forecast every 7 days, each model refit on all data before the origin.
Hyperparameters come from tune.py (searched only on earlier data)."""
import json
import sys
from pathlib import Path

import pandas as pd

from evaluate import metrics as summarize
from models import MODELS

ROOT = Path(__file__).parent
H, WINDOWS = 7, 52


def load() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "daily.csv", index_col=0, parse_dates=True).asfreq("D")


def load_params() -> dict:
    """JSON turns tuples into lists; statsmodels wants tuples back."""
    raw = json.loads((ROOT / "data" / "params.json").read_text())
    return {m: {k: tuple(v) if isinstance(v, list) else v for k, v in d["params"].items()} for m, d in raw.items()}


def run(df: pd.DataFrame, params: dict, names: list[str]) -> pd.DataFrame:
    rows = []
    for k in range(WINDOWS):
        end = len(df) - H * (WINDOWS - k)  # train = df[:end], test = next H days
        train, test = df.kwh.iloc[:end], df.kwh.iloc[end : end + H]
        for name in names:
            fn = MODELS[name][0]
            pred = fn(train, H, df, **params[name])
            rows.append(pd.DataFrame({"origin": train.index[-1], "model": name, "date": test.index,
                                      "actual": test.to_numpy(), "pred": pred.to_numpy()}))
        print(f"window {k + 1}/{WINDOWS}", flush=True)
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    # `python backtest.py [model ...]` reruns only the named models and merges into backtest.csv
    out = ROOT / "data" / "backtest.csv"
    names = sys.argv[1:] or list(MODELS)
    new = run(load(), load_params(), names)
    old = pd.read_csv(out, parse_dates=["origin", "date"]) if out.exists() else new.iloc[:0]
    bt = pd.concat([old[~old.model.isin(names)], new], ignore_index=True)
    bt.to_csv(out, index=False)
    print(summarize(bt))
