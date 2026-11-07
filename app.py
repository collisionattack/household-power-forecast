from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import evaluate as ev

ROOT = Path(__file__).parent

st.set_page_config(page_title="Hane elektrik tüketimi tahmini", layout="wide")


@st.cache_data
def load():
    daily = pd.read_csv(ROOT / "data" / "daily.csv", index_col=0, parse_dates=True)
    bt = pd.read_csv(ROOT / "data" / "backtest.csv", parse_dates=["origin", "date"])
    return daily, bt, ev.add_intervals(bt), ev.metrics(bt)


daily, bt, bt_int, M = load()
best = M.index[0]

st.title("Hane elektrik tüketimi tahmini")
st.write(f"Önümüzdeki 7 günün günlük tüketimini ortalama **{M.MAE[best]:.1f} kWh** hatayla tahmin ediyoruz. "
         f"Geçen haftayı tekrar etmek (baseline) {M.MAE[ev.BASELINE]:.1f} kWh şaşıyor.")

st.subheader("Tahmin")
origins = sorted(bt.origin.unique())
left, right = st.columns([1, 3], gap="large")
origin = left.select_slider("Tahminin yapıldığı gün", origins, value=origins[-1], format_func=lambda t: pd.Timestamp(t).strftime("%d.%m.%Y"))
models = bt.model.unique().tolist()
model = left.selectbox("Model", models, index=models.index(best))
left.caption("Model yalnızca bu günden öncesini gördü. Aralık, modelin önceki haftalardaki hatalarından hesaplanır "
             f"(%80, ilk {ev.CALIB} haftada yok). Hava modellerine tahmin günlerinin gerçekleşen havası verildi.")

w = bt_int[(bt_int.origin == origin) & (bt_int.model == model)]
base = bt[(bt.origin == origin) & (bt.model == ev.BASELINE)]
hist = daily.loc[pd.Timestamp(origin) - pd.Timedelta("45D"): pd.Timestamp(origin) + pd.Timedelta("7D")]

fig = go.Figure()
if w.lo.notna().all():
    fig.add_trace(go.Scatter(x=list(w.date) + list(w.date[::-1]), y=list(w.hi) + list(w.lo[::-1]), fill="toself", mode="none",
                             fillcolor="rgba(68,114,196,0.25)", name="%80 aralık", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=hist.index, y=hist.kwh, name="gerçek", line=dict(color="black", width=2)))
if model != ev.BASELINE:
    fig.add_trace(go.Scatter(x=base.date, y=base.pred, name="baseline", line=dict(color="gray", dash="dot")))
fig.add_trace(go.Scatter(x=w.date, y=w.pred, name=model, mode="lines+markers", line=dict(color="crimson", width=2.5)))
fig.add_vline(x=pd.Timestamp(origin), line_color="gray", line_dash="dash")
fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified", legend=dict(orientation="h", y=1.1, x=0))
fig.update_xaxes(tickformat="%d.%m").update_yaxes(title="kWh/gün")
right.plotly_chart(fig, width="stretch")

err = (w.pred - w.actual).abs().mean()
right.write(f"Bu hafta {model} ortalama {err:.1f} kWh şaştı (baseline: {(base.pred - base.actual).abs().mean():.1f} kWh).")

st.subheader(f"Modeller, son {bt.origin.nunique()} hafta")
st.table(M.rename(columns={"Improvement %": "İyileşme %", "Bias": "Yanlılık"}).rename_axis("Model").style.format(
    {"MAE": "{:.2f}", "RMSE": "{:.2f}", "MAE %": "{:.1f}", "sMAPE %": "{:.1f}", "İyileşme %": "{:.1f}", "Yanlılık": "{:+.2f}"}))
st.caption("MAE, RMSE ve yanlılık kWh/gün cinsindendir.")
