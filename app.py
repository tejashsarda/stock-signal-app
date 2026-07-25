import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Indian Stock Signal Engine", page_icon="📈", layout="wide")

st.title("🇮🇳 Indian & Global Stock Indicator Engine")

# Common Indian Stock Name Mappings
INDIAN_TICKER_MAP = {
    "TATA STEEL": "TATASTEEL.NS",
    "TATASTEEL": "TATASTEEL.NS",
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "INFYS": "INFY.NS",
    "INFY": "INFY.NS",
    "HDFC BANK": "HDFCBANK.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "ICICI BANK": "ICICIBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "SBIN": "SBIN.NS",
    "SBI": "SBIN.NS",
    "BHARTI AIRTEL": "BHARTIARTL.NS",
    "BHARTIARTL": "BHARTIARTL.NS",
    "ITC": "ITC.NS",
    "L&T": "LT.NS",
    "LT": "LT.NS",
    "TATAMOTORS": "TATAMOTORS.NS",
    "TATA MOTORS": "TATAMOTORS.NS"
}

def resolve_ticker(user_input):
    cleaned = user_input.upper().strip()
    
    # 1. Check direct map
    if cleaned in INDIAN_TICKER_MAP:
        return INDIAN_TICKER_MAP[cleaned]
    
    # 2. If it already ends with .NS or .BO, leave it
    if cleaned.endswith(".NS") or cleaned.endswith(".BO"):
        return cleaned
    
    # 3. If it looks like a standard symbol, try NSE first by appending .NS
    # Note: If searching US stocks (e.g. AAPL), user can just pass AAPL
    return cleaned

with st.sidebar.form(key="stock_form"):
    user_symbol = st.text_input("Enter Stock Ticker or Name", value="Tata Steel")
    exchange_pref = st.radio("Market Suffix (if not found automatically)", ["NSE (.NS)", "BSE (.BO)", "US / Direct"], index=0)
    timeframe = st.selectbox("Timeframe", options=["3m", "6m", "1y", "2y"], index=2)
    submit_button = st.form_submit_button(label="Analyze Stock")

# Format Ticker
raw_ticker = resolve_ticker(user_symbol)
if exchange_pref == "NSE (.NS)" and not raw_ticker.endswith(".NS") and not raw_ticker.endswith(".BO"):
    final_ticker = f"{raw_ticker}.NS"
elif exchange_pref == "BSE (.BO)" and not raw_ticker.endswith(".NS") and not raw_ticker.endswith(".BO"):
    final_ticker = f"{raw_ticker}.BO"
else:
    final_ticker = raw_ticker

@st.cache_data(ttl=900)
def load_and_calculate(symbol, period):
    df = yf.Ticker(symbol).history(period=period)
    if df.empty or len(df) < 30:
        return None, None
    
    df.columns = [c.lower() for c in df.columns]
    
    # Technical Calculations
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(200).mean()
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    df['bb_mid'] = df['close'].rolling(20).mean()
    std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + (std * 2)
    df['bb_lower'] = df['bb_mid'] - (std * 2)
    
    # Signals
    latest = df.iloc[-1]
    signals = []
    
    if not pd.isna(latest['sma_20']):
        signals.append(('Trend', 'SMA 20', 'BUY' if latest['close'] > latest['sma_20'] else 'SELL'))
    if not pd.isna(latest['sma_50']):
        signals.append(('Trend', 'SMA 50', 'BUY' if latest['close'] > latest['sma_50'] else 'SELL'))
    if not pd.isna(latest['sma_200']):
        signals.append(('Trend', 'SMA 200', 'BUY' if latest['close'] > latest['sma_200'] else 'SELL'))
    signals.append(('Trend', 'EMA Cross (12/26)', 'BUY' if latest['ema_12'] > latest['ema_26'] else 'SELL'))
    
    if not pd.isna(latest['rsi']):
        rsi_val = latest['rsi']
        sig = 'BUY (Oversold)' if rsi_val < 30 else ('SELL (Overbought)' if rsi_val > 70 else 'NEUTRAL')
        signals.append(('Momentum', f'RSI ({round(rsi_val,1)})', sig))
        
    signals.append(('Momentum', 'MACD Histogram', 'BUY' if latest['macd_hist'] > 0 else 'SELL'))
    
    if latest['close'] <= latest['bb_lower']:
        signals.append(('Volatility', 'Bollinger Bands', 'BUY (At Lower Band)'))
    elif latest['close'] >= latest['bb_upper']:
        signals.append(('Volatility', 'Bollinger Bands', 'SELL (At Upper Band)'))
    else:
        signals.append(('Volatility', 'Bollinger Bands', 'NEUTRAL'))

    return df, pd.DataFrame(signals, columns=['Category', 'Indicator', 'Signal'])

if user_symbol:
    with st.spinner(f"Fetching market data for `{final_ticker}`..."):
        df, signals_df = load_and_calculate(final_ticker, timeframe)
        
    if df is None:
        st.error(f"Could not load ticker `{final_ticker}`. Try using the exact Yahoo ticker (e.g. `TATASTEEL.NS` or `RELIANCE.NS`).")
    else:
        buy_count = len(signals_df[signals_df['Signal'].str.contains('BUY')])
        sell_count = len(signals_df[signals_df['Signal'].str.contains('SELL')])
        total = len(signals_df)
        
        score_pct = round(((buy_count - sell_count) / total) * 100, 1)
        
        if score_pct >= 25: overall, color = "STRONG BUY", "green"
        elif 5 <= score_pct < 25: overall, color = "BUY", "#2e7d32"
        elif -5 < score_pct < 5: overall, color = "NEUTRAL / HOLD", "gray"
        elif -25 < score_pct <= -5: overall, color = "SELL", "#c62828"
        else: overall, color = "STRONG SELL", "red"

        # Banner
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Resolved Ticker", final_ticker)
        col2.metric("Latest Price", f"₹{df.iloc[-1]['close']:.2f}")
        col3.metric("Buy Signals", f"{buy_count} / {total}")
        col4.markdown(f"### Final Signal:\n<span style='color:{color}; font-size:24px; font-weight:bold;'>{overall}</span>", unsafe_allow_html=True)

        st.divider()

        # Plots
        fig = make_subplots(
            rows=3, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.03, 
            subplot_titles=('Price (INR) & Moving Averages', 'RSI (14)', 'MACD'),
            row_heights=[0.6, 0.2, 0.2]
        )

        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='OHLC'), row=1, col=1)
        if 'sma_20' in df: fig.add_trace(go.Scatter(x=df.index, y=df['sma_20'], name='SMA 20', line=dict(color='orange', width=1)), row=1, col=1)
        if 'sma_50' in df: fig.add_trace(go.Scatter(x=df.index, y=df['sma_50'], name='SMA 50', line=dict(color='blue', width=1)), row=1, col=1)
        if 'sma_200' in df: fig.add_trace(go.Scatter(x=df.index, y=df['sma_200'], name='SMA 200', line=dict(color='purple', width=1)), row=1, col=1)

        fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], name='RSI', line=dict(color='purple', width=1.5)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        fig.add_trace(go.Scatter(x=df.index, y=df['macd'], name='MACD', line=dict(color='blue', width=1)), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['macd_signal'], name='Signal Line', line=dict(color='orange', width=1)), row=3, col=1)
        hist_colors = ['green' if val > 0 else 'red' for val in df['macd_hist']]
        fig.add_trace(go.Bar(x=df.index, y=df['macd_hist'], name='Histogram', marker_color=hist_colors), row=3, col=1)

        fig.update_layout(height=700, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📋 Indicator Breakdown")
        st.dataframe(signals_df, use_container_width=True)
