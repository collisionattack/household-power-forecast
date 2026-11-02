"""Raw 1-minute UCI file -> data/daily.csv (daily kWh)."""
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).parent
RAW = ROOT / "raw" / "household_power_consumption.txt"
OUT = ROOT / "data" / "daily.csv"


def to_daily(raw: pd.DataFrame) -> pd.DataFrame:
    """Minute rows -> daily kWh. Mean of present minutes * 1440, so missing minutes don't shrink a day."""
    m = raw[["Global_active_power", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]].resample("D").mean()
    d = pd.DataFrame({
        "kwh": m.Global_active_power * 24,  # mean kW * 24 h
        "kitchen": m.Sub_metering_1 * 1.44,  # mean Wh/min * 1440 min / 1000
        "laundry": m.Sub_metering_2 * 1.44,
        "heater": m.Sub_metering_3 * 1.44,
    })
    d["imputed"] = d.kwh.isna()
    d = d.interpolate("time")
    return d.iloc[1:-1]  # first (starts 17:24) and last (ends 21:02) days are partial


def fetch_weather(start: str, end: str) -> pd.DataFrame:
    """Open-Meteo archive (free, no key) for Sceaux, France (the household's town)."""
    url = ("https://archive-api.open-meteo.com/v1/archive?latitude=48.78&longitude=2.29"
           f"&start_date={start}&end_date={end}&timezone=Europe%2FParis"
           "&daily=temperature_2m_mean,sunshine_duration,precipitation_sum")
    w = pd.DataFrame(requests.get(url, timeout=60).json()["daily"]).set_index("time")
    w.index = pd.to_datetime(w.index)
    w.columns = ["temp", "sunshine_h", "precip_mm"]
    w["sunshine_h"] /= 3600
    return w


def main():
    raw = pd.read_csv(RAW, sep=";", na_values="?", low_memory=False)
    raw.index = pd.to_datetime(raw.pop("Date") + " " + raw.pop("Time"), format="%d/%m/%Y %H:%M:%S")
    daily = to_daily(raw)
    daily = daily.join(fetch_weather(str(daily.index[0].date()), str(daily.index[-1].date())), how="left")
    assert not daily.isna().any().any(), "weather join left gaps"
    OUT.parent.mkdir(exist_ok=True)
    daily.rename_axis("date").round(3).to_csv(OUT)
    print(f"{len(daily)} days {daily.index[0].date()}..{daily.index[-1].date()}, imputed={int(daily.imputed.sum())}")


if __name__ == "__main__":
    main()
