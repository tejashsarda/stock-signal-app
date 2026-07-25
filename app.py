import streamlit as st
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="100+ Indicator Stock Signal Engine",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Multi-Indicator Consensus Engine")
st.markdown("Analyzes **100+ technical indicators & patterns** to generate a unified quantitative signal.")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("User Parameters")
ticker_symbol = st.sidebar.text_input("Ticker Symbol", value="AAPL").upper()
timeframe = st.sidebar.selectbox("Time Horizon", options=["6m", "1y", "2y", "5y"], index=1)

# --- QUANT ENGINE FUNCTION ---
@st.cache_data(ttl=3600)  # Cache data for 1 hour to optimize performance
def run_quant_analysis(symbol, period):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)
    
    if df.empty or len(df) < 50:
        return None, None, None

    df.columns = [c.lower() for c in df.columns]

    # Calculate 100+ technical indicators via pandas-ta
    df.ta.strategy("All")
    
    latest_bar = df.iloc[-1]
    prev_bar = df.iloc[-2]
    
    signals = []

    # 1. TREND
    for ma in [10, 20, 50, 200]:
        col = f"SMA_{ma}"
        if col in df.columns and not pd.isna(latest_bar[col]):
            signals.append(('Trend', 'SMA Cross', 1 if latest_bar['close'] > latest_bar[col] else -1))
            
    if 'EMA_10' in df.columns and 'EMA_50' in df.columns:
        signals.append(('Trend', 'EMA Cross', 1 if latest_bar['EMA_10'] > latest_bar['EMA_50'] else -1))
        
    if 'SUPERT_7_3.0' in df.columns:
        signals.append(('Trend', 'Supertrend', 1 if latest_bar['close'] > latest_bar['SUPERT_7_3.0'] else -1))

    # 2. MOMENTUM
    if 'RSI_14' in df.columns and not pd.isna(latest_bar['RSI_14']):
        rsi = latest_bar['RSI_14']
        val = 1 if rsi < 30 else (-1 if rsi > 70 else (rsi - 50) / 20)
        signals.append(('Momentum', f'RSI ({round(rsi,1)})', val))

    if 'MACDh_12_26_9' in df.columns and not pd.isna(latest_bar['MACDh_12_26_9']):
        signals.append(('Momentum', 'MACD Hist', 1 if latest_bar['MACDh_12_26_9'] > 0 else -1))

    if 'STOCHk_14_3_3' in df.columns and not pd.isna(latest_bar['STOCHk_14_3_3']):
        stoch = latest_bar['STOCHk_14_3_3']
        signals.append(('Momentum', f'Stochastic ({round(stoch,1)})', 1 if stoch < 20 else (-1 if stoch > 80 else 0)))

    # 3. VOLATILITY
    if 'BBL_50_2.0' in df.columns and 'BBU_50_2.0' in df.columns:
        if latest_bar['close'] <= latest_bar['BBL_50_2.0']:
            signals.append(('Volatility', 'Bollinger Lower', 1))
        elif latest_bar['close'] >= latest_bar['BBU_50_2.0']:
            signals.append(('Volatility', 'Bollinger Upper', -1))

    # 4. VOLUME
    if 'OBV' in df.columns and not pd.isna(latest_bar['OBV']):
        signals.append(('Volume', 'OBV Trend', 1 if latest_bar['OBV'] > prev_bar['OBV'] else -1))
        
    if 'CMF_20' in df.columns and not pd.isna(latest_bar['CMF_20']):
        signals.append(('Volume', 'Chaikin Money Flow', 1 if latest_bar['CMF_20'] > 0 else -1))

    # 5. CANDLESTICK PATTERNS
    cdl_cols = [c for c in df.columns if c.startswith('CDL_')]
    for cdl in cdl_cols:
        val = latest_bar[cdl]
        if not pd.isna(val) and val != 0:
            signals.append(('Candlesticks', cdl.replace('CDL_', ''), 1 if val > 0 else -1))

    signals_df = pd.DataFrame(signals, columns=['Category', 'Indicator', 'Score'])
    return df, signals_df, latest_bar

# --- MAIN EXECUTION ---
if ticker_symbol:
    with st.spinner(f"Computing indicators for {ticker_symbol}..."):
        df, signals_df, latest_bar = run_quant_analysis(ticker_symbol, timeframe)

    if df is None:
        st.error(f"Could not retrieve enough data for symbol: `{ticker_symbol}`. Check the ticker and try again.")
    else:
        # Calculate Scores
        score_percentage = round(signals_df['Score'].mean() * 100, 1)
        
        if score_percentage >= 35:
            signal_text, signal_color = "STRONG BUY", "green"
        elif 10 <= score_percentage < 35:
            signal_text, signal_color = "BUY", "#2e7d32"
        elif -10 < score_percentage < 10:
            signal_text, signal_color = "NEUTRAL / HOLD", "gray"
        elif -35 < score_percentage <= -10:
            signal_text, signal_color = "SELL", "#c62828"
        else:
            signal_text, signal_color = "STRONG SELL", "red"

        # --- TOP METRICS DISPLAY ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Latest Price", f"${latest_bar['close']:.2f}")
        col2.metric("Evaluated Signals", len(signals_df))
        col3.metric("Consensus Score", f"{score_percentage}%")
        col4.markdown(f"### Signal: <span style='color:{signal_color}'>{signal_text}</span>", unsafe_allow_html=True)

        st.divider()

        # --- INTERACTIVE CHART ---
        st.subheader(f"{ticker_symbol} Price & Volume Chart")
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'], high=df['high'],
            low=df['low'], close=df['close'],
            name="OHLC"
        ))
        
        # Add SMA 50 and 200 overlays if available
        if 'SMA_50' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', name='SMA 50', line=dict(width=1.5)))
        if 'SMA_200' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], mode='lines', name='SMA 200', line=dict(width=1.5)))

        fig.update_layout(xaxis_rangeslider_visible=False, height=450, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)

        # --- CATEGORY BREAKDOWN ---
        st.subheader("Category Breakdown")
        cat_summary = signals_df.groupby('Category')['Score'].agg(['count', 'mean']).reset_index()
        cat_summary.columns = ['Category', 'Indicators Active', 'Sentiment Score (-1.0 to +1.0)']
        
        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.dataframe(cat_summary, use_container_width=True)
        with col_right:
            st.caption("Active Signal Breakdown")
            st.dataframe(signals_df[['Category', 'Indicator', 'Score']], use_container_width=True)
