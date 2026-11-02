# Household Power Forecast

7-day-ahead forecast of daily electricity use for one French household (UCI dataset, 2006–2010). It has an EDA notebook, a rolling backtest of 7 models, a reliability analysis in `evaluate.py` and `sensitivity.py` (prediction intervals, significance tests, weather sensitivity, error breakdown) and a small Streamlit demo with a Turkish UI.

## Run
```bash
pip install -r requirements.txt        # app only; requirements-dev.txt adds prophet, lightgbm, jupyter, pytest
streamlit run app.py          # reads the committed data/*.csv, no raw file needed
```

## Notebook (Turkish, outputs included)
- `notebooks/eda.ipynb`: missingness, seasonality (STL, ACF/PACF), stationarity, weather link, sub-meters. Section 1 needs the raw file.

## Rebuild from scratch
```bash
# put household_power_consumption.txt in raw/ (https://archive.ics.uci.edu/dataset/235)
python data.py            # minute data -> daily kWh + Open-Meteo weather -> data/daily.csv
python tune.py            # grid search, rolling CV on data before the backtest window -> data/params.json
python backtest.py        # 52 weekly rolling-origin 7-day forecasts -> data/backtest.csv   (~15 min)
python sensitivity.py     # weather-forecast noise sensitivity -> data/sensitivity.csv      (~35 min)
pytest
```
`tune.py` and `backtest.py` accept model names to rerun only those, e.g. `python backtest.py LightGBM`.

## Method
- Target: daily kWh (mean of available minutes × 24; 9 fully missing days interpolated).
- Models: seasonal naive (baseline), SARIMA, SARIMAX + weather, Prophet, Prophet + weather, LightGBM, LightGBM + weather. Weather: temperature, sunshine hours.
- Evaluation: refit before each of the last 52 weekly origins; metrics defined in `evaluate.METRIC_HELP`.
- Intervals: pred ± the 80% quantile of the model's own earlier absolute errors (no look-ahead).
- Significance: paired over weekly windows, Wilcoxon + bootstrap CI, Holm-corrected within each family of comparisons.

## Results (52 × 7 days)
| model | MAE kWh/day | vs baseline |
|---|---|---|
| LightGBM | 4.66 | −26% |
| LightGBM + weather | 4.67 | −26% |
| Prophet + weather | 4.67 | −26% |
| Prophet | 4.68 | −25% |
| SARIMA | 4.81 | −23% |
| SARIMAX + weather | 4.87 | −23% |
| Seasonal naive | 6.28 | 0 |

- All six models beat the baseline (Holm p < 0.001).
- Weather adds nothing detectable (0 of 3 pairs significant) and the models cannot be ranked against each other. "Not significant" is not "equal": the CIs are about ±0.3 kWh wide.
- The 80% interval covers 80–86% of actuals overall (Fri 98%, Sat–Sun 72–75%) and is wide (≈ 15 kWh).

## Caveats
- One household, one test year.
- The backtest always forecasts from a Thursday, so horizon and weekday are confounded.
- Weather models get observed weather for the forecast days; see the sensitivity analysis for noisy weather.
