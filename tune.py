"""Grid search per model on the weeks just before the backtest window; writes data/params.json."""
import json
import sys
from pathlib import Path

import pandas as pd

from backtest import H, WINDOWS, load
from models import MODELS, grid

ROOT = Path(__file__).parent
CV_ORIGINS = 4  # weekly forecast origins used for tuning


def tune(df: pd.DataFrame, names: list[str]) -> dict:
    y, x = df.kwh, df
    cut = len(y) - H * WINDOWS  # backtest starts here; never look past it
    best = {}
    for name in names:
        fn, space = MODELS[name]
        scores = []
        for p in grid(space):
            errs = []
            for k in range(CV_ORIGINS):
                end = cut - H * (CV_ORIGINS - k)
                pred = fn(y.iloc[:end], H, x, **p)
                errs.append((pred - y.iloc[end : end + H].to_numpy()).abs().mean())
            scores.append((sum(errs) / len(errs), p))
        score, p = min(scores, key=lambda s: s[0])
        best[name] = {"params": p, "cv_mae": round(score, 3)}
        print(f"{name}: cv_mae={score:.2f} {p} ({len(scores)} combos)", flush=True)
    return best


if __name__ == "__main__":
    # `python tune.py [model ...]` tunes only the named models and merges into params.json
    out = ROOT / "data" / "params.json"
    names = sys.argv[1:] or list(MODELS)
    old = json.loads(out.read_text()) if out.exists() else {}
    out.write_text(json.dumps({**old, **tune(load(), names)}, indent=1, default=list))
