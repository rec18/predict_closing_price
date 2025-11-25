
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go
st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("📈 Stock Price Prediction ")
# ---------------------------
# SIDEBAR INPUT
# ---------------------------
st.sidebar.header("Stock Input")
symbol = st.sidebar.text_input("Stock Symbol (e.g., RELIANCE.NS)", "RELIANCE.NS")
start_date = st.sidebar.text_input("Start Date", "2010-01-01")
end_date = st.sidebar.text_input("End Date", "2025-11-07")
# ---------------------------
# FETCH DATA
# ---------------------------
st.subheader("Stock Data")
df = yf.download(symbol, start=start_date, end=end_date)
if df.empty:
    st.error("No data found. Check symbol or date range.")
    st.stop()
# Fix multi-index from yfinance
df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
# Ensure Close column exists
if "Close" not in df.columns:
    st.error("Downloaded data has no 'Close' column. Cannot proceed.")
    st.stop()
st.dataframe(df.head(20))
# ---------------------------
# CREATE FEATURES
# ---------------------------
df["SMA10"] = df["Close"].rolling(10).mean()
df["SMA50"] = df["Close"].rolling(50).mean()
df["Daily_Return"] = df["Close"].pct_change()
# ---------------------------
# CHARTS SECTION
# ---------------------------
# ========== CANDLESTICK ==========
st.subheader("📊 Candlestick Chart with SMA")
fig = go.Figure(data=[go.Candlestick(
    x=df.index,
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    name="Candlestick"
)])
fig.add_trace(go.Scatter(x=df.index, y=df["SMA10"], name="SMA10", line=dict(color='yellow')))
fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA50", line=dict(color='cyan')))

fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)
# ========== VOLUME CHART ==========
st.subheader("📈 Volume Chart")
vol_fig = go.Figure()
vol_fig.add_trace(go.Bar(
    x=df.index,
    y=df["Volume"],
    name="Volume"
))

vol_fig.update_layout(template="plotly_dark", height=250)
st.plotly_chart(vol_fig, use_container_width=True)
# ========== DAILY RETURN HISTOGRAM ==========
st.subheader("📉 Daily Return Distribution")
hist_fig = go.Figure()
hist_fig.add_trace(go.Histogram(
    x=df["Daily_Return"].dropna(),
    nbinsx=100,
    name="Daily Returns"
))
hist_fig.update_layout(template="plotly_dark", height=300)
st.plotly_chart(hist_fig, use_container_width=True)
# ---------------------------
# MODEL DATA PREP
# ---------------------------
model_df = df.dropna(subset=["SMA10", "SMA50"]).copy()
if len(model_df) < 50:
    st.warning("Not enough SMA data to train model (need ≥ 50 rows). Increase date range.")
    st.stop()

X = model_df[["SMA10", "SMA50"]]
y = model_df["Close"]

# Train model
model = LinearRegression()
model.fit(X, y)

# ---------------------------
# PREDICT NEXT DAY
# ---------------------------
last_row = df.iloc[-1]
last_sma10 = last_row["SMA10"]
last_sma50 = last_row["SMA50"]

next_day_price = model.predict([[last_sma10, last_sma50]])[0]
# ---------------------------
# OUTPUT
# ---------------------------
st.subheader("🎯 Next Day Price Prediction")
st.success(f" Predicted Close Price for Next Day: **₹{next_day_price:.2f}**")

st.write("Model used features:")
st.write(f"- SMA10: {last_sma10:.2f}")
st.write(f"- SMA50: {last_sma50:.2f}")

