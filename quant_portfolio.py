import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import platform
import os

# --- 1. LINUX/CLOUD FIX ---
# This trick forces yfinance to use a temp folder if running on Streamlit Cloud
if platform.system() == "Linux":
    os.environ["YFINANCE_CACHE_DIR"] = "/tmp/yfinance"

# --- PAGE CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Quantitative Strategy Dashboard")

# --- SIDEBAR INPUTS ---
st.sidebar.header("User Input")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, NVDA)", "AAPL").upper()
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# --- FUNCTION: FETCH DATA ---
@st.cache_data(ttl=3600)
def get_data(ticker, start, end):
    try:
        # auto_adjust=False often helps with data alignment issues
        data = yf.download(ticker, start=start, end=end, auto_adjust=False)
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        if data.empty:
            return pd.DataFrame()
            
        return data
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()

# --- FUNCTION: FETCH FUNDAMENTALS (WITH FALLBACK) ---
@st.cache_data(ttl=86400)
def get_fundamentals(ticker):
    # Default dictionary to ensure tabs ALWAYS render
    default_data = {
        "Market Cap": "N/A", "P/E Ratio": "N/A", "Forward P/E": "N/A", "EV/EBITDA": "N/A",
        "EPS": "N/A", "ROE": "N/A", "Profit Margin": "N/A",
        "Debt/Equity": "N/A", "Current Ratio": "N/A", "Quick Ratio": "N/A",
        "Sector": "N/A", "Beta": "N/A",
        "Status": "Blocked" # Flag to show user
    }
    
    try:
        stock = yf.Ticker(ticker)
        # We try to fetch info. If it crashes (common on Cloud), we catch it.
        info = stock.info
        
        if not info or len(info) < 2:
            return default_data

        def get_safe(key):
            return info.get(key, "N/A")

        return {
            "Market Cap": get_safe("marketCap"),
            "P/E Ratio": get_safe("trailingPE"),
            "Forward P/E": get_safe("forwardPE"),
            "EV/EBITDA": get_safe("enterpriseToEbitda"),
            "EPS": get_safe("trailingEps"),
            "ROE": get_safe("returnOnEquity"),
            "Profit Margin": get_safe("profitMargins"),
            "Debt/Equity": get_safe("debtToEquity"),
            "Current Ratio": get_safe("currentRatio"),
            "Quick Ratio": get_safe("quickRatio"),
            "Sector": get_safe("sector"),
            "Beta": get_safe("beta"),
            "Status": "OK"
        }
    except Exception:
        return default_data

# --- MAIN APP ---
st.title(f"Quantitative Stock Analysis: {ticker}")

data = get_data(ticker, start_date, end_date)

if not data.empty:
    
    # --- FUNDAMENTALS SECTION (Now Guaranteed to Show) ---
    fund_data = get_fundamentals(ticker)
    
    st.subheader("Fundamental Analysis")
    
    # If Status is Blocked, show a warning but STILL show the tabs
    if fund_data["Status"] == "Blocked":
        st.warning("⚠️ Yahoo Finance blocked the 'Profile' data request on the Cloud server. Showing 'N/A' placeholders.")

    tab1, tab2, tab3 = st.tabs(["💲 Valuation", "📈 Profitability", "⚖️ Financial Health"])
    
    def fmt(val, is_pct=False, is_large=False):
        if val == "N/A" or val is None: return "N/A"
        if is_pct: return f"{val * 100:.2f}%"
        if is_large:
            if val >= 1e12: return f"${val/1e12:.2f}T"
            if val >= 1e9: return f"${val/1e9:.2f}B"
            return f"${val/1e6:.2f}M"
        return f"{val:.2f}"

    with tab1:
        col_v1, col_v2, col_v3, col_v4 = st.columns(4)
        col_v1.metric("Market Cap", fmt(fund_data["Market Cap"], is_large=True))
        col_v2.metric("Trailing P/E", fmt(fund_data["P/E Ratio"]))
        col_v3.metric("Forward P/E", fmt(fund_data["Forward P/E"]))
        col_v4.metric("EV/EBITDA", fmt(fund_data["EV/EBITDA"]))

    with tab2:
        col_p1, col_p2, col_p3 = st.columns(3)
        col_p1.metric("EPS (Trailing)", f"${fmt(fund_data['EPS'])}")
        col_p2.metric("Return on Equity (ROE)", fmt(fund_data["ROE"], is_pct=True))
        col_p3.metric("Profit Margin", fmt(fund_data["Profit Margin"], is_pct=True))

    with tab3:
        col_h1, col_h2, col_h3 = st.columns(3)
        col_h1.metric("Debt-to-Equity", fmt(fund_data["Debt/Equity"]))
        col_h2.metric("Current Ratio", fmt(fund_data["Current Ratio"]))
        col_h3.metric("Quick Ratio", fmt(fund_data["Quick Ratio"]))

    st.markdown("---")

    # --- TECHNICALS (Should work fine) ---
    # Quick fix for 'Close' column mismatch
    if 'Close' not in data.columns and 'Adj Close' in data.columns:
        data['Close'] = data['Adj Close']

    data['Daily Return'] = data['Close'].pct_change()
    
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_100'] = data['Close'].rolling(window=100).mean()
    
    def calculate_rsi(data, window=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    data['RSI'] = calculate_rsi(data['Close'])
    data['BB_Upper'] = data['SMA_20'] + (data['Close'].rolling(window=20).std() * 2)
    data['BB_Lower'] = data['SMA_20'] - (data['Close'].rolling(window=20).std() * 2)
    
    annual_volatility = data['Daily Return'].std() * np.sqrt(252)
    risk_free_rate = 0.042
    sharpe_ratio = (data['Daily Return'].mean() * 252 - risk_free_rate) / annual_volatility

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", f"${data['Close'].iloc[-1]:.2f}")
    col2.metric("RSI (14-Day)", f"{data['RSI'].iloc[-1]:.0f}")
    col3.metric("Volatility", f"{annual_volatility:.2%}")
    col4.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index,
                    open=data['Open'], high=data['High'],
                    low=data['Low'], close=data['Close'], name='Price'))
    fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], line=dict(color='gray', width=1), name='Upper Band', showlegend=False))
    fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], line=dict(color='gray', width=1), name='Lower Band', fill='tonexty', fillcolor='rgba(128, 128, 128, 0.1)', showlegend=False))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], line=dict(color='blue', width=1.5), name='50-Day SMA'))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_100'], line=dict(color='red', width=1.5), name='100-Day SMA'))
    fig.update_layout(height=600, title=f"{ticker} Technical Chart")
    st.plotly_chart(fig, use_container_width=True)

    # --- MONTE CARLO (Session State Fixed) ---
    st.markdown("---")
    st.subheader("Monte Carlo Simulation")
    
    sim_type = st.radio("Select Simulation Type:", ("Weighted Momentum", "Long-Term Avg", "Risk-Neutral"))
    simulations = st.sidebar.slider("Number of Simulations", 200, 1000, 200)
    time_horizon = st.sidebar.slider("Time Horizon (Days)", 30, 365, 252)

    # Session State Logic
    if "run_mc" not in st.session_state:
        st.session_state.run_mc = False

    if st.button("Run Monte Carlo Simulation"):
        st.session_state.run_mc = True

    if st.session_state.run_mc:
        with st.spinner("Running Simulation..."):
            try:
                mc_data = yf.download(ticker, period="5y", auto_adjust=False)
                if isinstance(mc_data.columns, pd.MultiIndex):
                    mc_data.columns = mc_data.columns.get_level_values(0)

                mc_data['Daily Return'] = mc_data['Close'].pct_change()
                last_price = mc_data['Close'].iloc[-1]
                daily_vol = mc_data['Daily Return'].std()
                
                if sim_type == "Weighted Momentum":
                    drift = mc_data['Daily Return'].ewm(span=252).mean().iloc[-1] - (0.5 * daily_vol ** 2)
                elif sim_type == "Long-Term Avg":
                    drift = mc_data['Daily Return'].mean() - (0.5 * daily_vol ** 2)
                else:
                    drift = (0.042 / 252) - (0.5 * daily_vol ** 2)
                
                simulation_df = pd.DataFrame()
                for i in range(simulations):
                    daily_shocks = np.random.normal(drift, daily_vol, time_horizon)
                    price_series = [last_price]
                    for shock in daily_shocks:
                        price_series.append(price_series[-1] * np.exp(shock))
                    simulation_df[f"Sim_{i}"] = price_series

                fig_mc = go.Figure()
                for col in simulation_df.columns[:50]:
                    fig_mc.add_trace(go.Scatter(y=simulation_df[col], mode='lines', line=dict(color='rgba(100, 100, 255, 0.05)'), hoverinfo='skip', showlegend=False))
                fig_mc.add_trace(go.Scatter(y=simulation_df.mean(axis=1), mode='lines', line=dict(color='red', width=3), name='Average Path'))
                fig_mc.update_layout(title="Monte Carlo Simulation", xaxis_title="Days", yaxis_title="Price ($)")
                st.plotly_chart(fig_mc, use_container_width=True)
            except Exception as e:
                st.error(f"Simulation failed: {e}")

else:
    st.error(f"No data found for {ticker}. The market might be closed or Yahoo is blocking the request.")
