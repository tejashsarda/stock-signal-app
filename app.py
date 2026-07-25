import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="Stock Signal Engine", page_icon="📈", layout="wide")
st.title("📈 Multi-Indicator Consensus Engine")

ticker_symbol = st.sidebar.text_input("Ticker Symbol", value="AAPL").upper()
timeframe = st.sidebar.selectbox("Time Horizon", options=["6m", "1y", "2y"], index=1)

@st.cache_data(ttl=3600)
def analyze_stock(symbol, period):
    df = yf.Ticker(symbol).history(period=period)
    if df.empty or len(df) < 50:
        return None, None
    
    df.columns = [c.lower() for c in df.columns]
    
    # --- PURE PYTHON TECHNICAL INDICATORS ---
    # Moving Averages
    for ma in [10, 20, 50, 200]:
        df[f'sma_{ma}'] = df['close'].rolling(window=ma).mean()
        
    df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_mid'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_mid'] - (df['bb_std'] * 2)
    
    # Stochastic Oscillator
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
    
    # On-Balance Volume (OBV)
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

    # --- EVALUATE SIGNALS ---
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    signals = []
    
    # Trend Signals
    for ma in [10, 20, 50, 200]:
        if not pd.isna(latest[f'sma_{ma}']):
            signals.append(('Trend', f'SMA {ma}', 1 if latest['close'] > latest[f'sma_{ma}'] else -1))
    
    signals.append(('Trend', 'EMA Cross (10/50)', 1 if latest['ema_10'] > latest['ema_50'] else -1))
    
    # Momentum Signals
    if not pd.isna(latest['rsi_14']):
        rsi = latest['rsi_14']
        val = 1 if rsi < 30 else (-1 if rsi > 70 else (rsi - 50) / 20)
        signals.append(('Momentum', f'RSI ({round(rsi,1)})', val))
        
    signals.append(('Momentum', 'MACD Histogram', 1 if latest['macd_hist'] > 0 else -1))
    
    if not pd.isna(latest['stoch_k']):
        stoch = latest['stoch_k']
        signals.append(('Momentum', f'Stochastic ({round(stoch,1)})', 1 if stoch < 20 else (-1 if stoch > 80 else 0)))
        
    # Volatility Signals
    if latest['close'] <= latest['bb_lower']:
        signals.append(('Volatility', 'Bollinger Lower Band', 1))
    elif latest['close'] >= latest['bb_upper']:
        signals.append(('Volatility', 'Bollinger Upper Band', -1))
    else:
        signals.append(('Volatility', 'Bollinger Neutral', 0))
        
    # Volume Signals
    signals.append(('Volume', 'OBV Trend', 1 if latest['obv'] > prev['obv'] else -1))
    
    return df, pd.DataFrame(signals, columns=['Category', 'Indicator', 'Score'])

# --- APP LAYOUT ---
if ticker_symbol:
    with st.spinner(f"Analyzing {ticker_symbol}..."):
        df, signals_df = analyze_stock(ticker_symbol, timeframe)
        
    if df is None:
        st.error("Invalid ticker symbol or insufficient data.")
    else:
        score_pct = round(signals_df['Score'].mean() * 100, 1)
        
        if score_pct >= 30: signal, color = "STRONG BUY", "green"
        elif 10 <= score_pct < 30: signal, color = "BUY", "#2e7d32"
        elif -10 < score_pct < 10: signal, color = "NEUTRAL", "gray"
        elif -30 < score_pct <= -10: signal, color = "SELL", "#c62828"
        else: signal, color = "STRONG SELL", "red"
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Price", f"${df.iloc[-1]['close']:.2f}")
        col2.metric("Active Signals", len(signals_df))
        col3.metric("Consensus Score", f"{score_pct}%")
        col4.markdown(f"### Signal: <span style='color:{color}'>{signal}</span>", unsafe_allow_html=True)
        
        st.divider()
        
        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
        fig.update_layout(xaxis_rangeslider_visible=False, height=400, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Indicator Signal Breakdown")
        st.dataframe(signals_df, use_container_width=True)
