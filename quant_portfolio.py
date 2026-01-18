import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Quantitative Strategy Dashboard")

# --- SIDEBAR INPUTS ---
st.sidebar.header("User Input")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, NVDA)", "AAPL").upper()
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# --- FUNCTION: FETCH DATA (Cached to save bandwidth) ---
@st.cache_data(ttl=3600)  # Cache data for 1 hour to prevent Yahoo blocking
def get_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end)
        # Flatten MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# --- FUNCTION: FETCH FUNDAMENTALS (With Fallback) ---
@st.cache_data(ttl=86400) # Cache for 24 hours
def get_fundamentals(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # If Yahoo blocks us, info might be empty or None
        if not info or len(info) < 5: 
            return None

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
            "Beta": get_safe("beta")
        }
    except Exception:
        return None

# --- MAIN APP ---
st.title(f"Quantitative Stock Analysis: {ticker}")

# 1. Load Data
data = get_data(ticker, start_date, end_date)

if not data.empty:
    
    # 2. Display Fundamental Data
    fund_data = get_fundamentals(ticker)
    
    if fund_data:
        st.subheader("Fundamental Analysis")
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
            
    else:
        st.warning(f"⚠️ Yahoo Finance temporarily blocked the Fundamental Data request for {ticker} (Common on Cloud servers). The Technical Charts below will still work.")

    st.markdown("---")

    # 3. Financial Calculations
    data['Daily Return'] = data['Close'].pct_change()
    
    # Moving Averages
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_100'] = data['Close'].rolling(window=100).mean()
    
    # RSI
    def calculate_rsi(data, window=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    data['RSI'] = calculate_rsi(data['Close'])
    
    # Bollinger Bands
    data['BB_Upper'] = data['SMA_20'] + (data['Close'].rolling(window=20).std() * 2)
    data['BB_Lower'] = data['SMA_20'] - (data['Close'].rolling(window=20).std() * 2)
    
    # Risk Metrics
    annual_volatility = data['Daily Return'].std() * np.sqrt(252)
    risk_free_rate = 0.042
    sharpe_ratio = (data['Daily Return'].mean() * 252 - risk_free_rate) / annual_volatility

    # 4. Display Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", f"${data['Close'].iloc[-1]:.2f}")
    col2.metric("RSI (14-Day)", f"{data['RSI'].iloc[-1]:.0f}")
    col3.metric("Volatility", f"{annual_volatility:.2%}")
    col4.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

    # 5. Plotting
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index,
                    open=data['Open'], high=data['High'],
                    low=data['Low'], close=data['Close'],
                    name='Price'))
    
    fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], line=dict(color='gray', width=1), name='Upper Band', showlegend=False))
    fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], line=dict(color='gray', width=1), name='Lower Band', fill='tonexty', fillcolor='rgba(128, 128, 128, 0.1)', showlegend=False))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], line=dict(color='blue', width=1.5), name='50-Day SMA'))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_100'], line=dict(color='red', width=1.5), name='100-Day SMA'))

    fig.update_layout(height=600, title=f"{ticker} Technical Chart (Price + Volatility Bands)")
    st.plotly_chart(fig, use_container_width=True)

    # 6. Algorithmic Trading Signal
    st.subheader("Algorithmic Trading Signal (Trend + Momentum)")
    
    last_price = data['Close'].iloc[-1]
    sma_50 = data['SMA_50'].iloc[-1]
    sma_100 = data['SMA_100'].iloc[-1]
    rsi = data['RSI'].iloc[-1]
    
    trend = "BULLISH" if sma_50 > sma_100 else "BEARISH"
    
    if trend == "BULLISH" and rsi < 70:
        st.success(f"Signal: BUY 🟢 (Trend is Up, RSI is Healthy at {rsi:.0f})")
    elif trend == "BULLISH" and rsi > 70:
        st.warning(f"Signal: HOLD 🟡 (Trend is Up, but RSI is Overbought at {rsi:.0f})")
    elif trend == "BEARISH" and rsi > 30:
        st.error(f"Signal: SELL 🔴 (Trend is Down)")
    elif trend == "BEARISH" and rsi < 30:
        st.warning(f"Signal: WATCH 🟠 (Trend is Down, but RSI is Oversold at {rsi:.0f})")

    with st.expander("See Strategy Logic"):
        st.write("Strategy combines SMA Trend Following with RSI Mean Reversion.")

    # 7. Risk Management (VaR)
    st.markdown("---")
    st.subheader("Risk Management: Value at Risk (VaR)")

    portfolio_size = st.sidebar.number_input("Enter Portfolio Value ($)", min_value=10000, value=1000000, step=100000)
    
    daily_vol = data['Daily Return'].std()
    var_95 = portfolio_size * daily_vol * 1.65
    
    col_risk1, col_risk2 = st.columns(2)
    with col_risk1:
        st.info(f"If you invested **${portfolio_size:,.0f}** in {ticker}:")
        st.metric("Estimated Daily VaR (95%)", f"-${var_95:,.2f}")
    
    with col_risk2:
        st.caption(f"With 95% confidence, you are unlikely to lose more than **${var_95:,.0f}** in a single day.")
        
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=data['Daily Return'].dropna(), nbinsx=50, name='Returns', marker_color='#1f77b4'))
        var_cutoff = -1.65 * daily_vol
        fig_hist.add_vline(x=var_cutoff, line_width=3, line_dash="dash", line_color="red")
        fig_hist.update_layout(title="Distribution of Daily Returns", showlegend=False, height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_hist, use_container_width=True)

    # 8. Monte Carlo Simulation (FIXED WITH SESSION STATE)
    st.markdown("---")
    st.subheader("Monte Carlo Simulation (Future Price Projection)")
    
    sim_type = st.radio("Select Simulation Type:", 
                        ("Weighted Momentum (Recent Bias)", 
                         "Long-Term Avg (5-Year)", 
                         "Risk-Neutral (Conservative)"))
    
    simulations = st.sidebar.slider("Number of Simulations", 200, 1000, 200)
    time_horizon = st.sidebar.slider("Time Horizon (Days)", 30, 365, 252)

    # Initialize Session State
    if "run_mc" not in st.session_state:
        st.session_state.run_mc = False

    if st.button("Run Monte Carlo Simulation"):
        st.session_state.run_mc = True

    if st.session_state.run_mc:
        with st.spinner("Running Simulation..."):
            # Fetch 5 Years history separately
            mc_data = yf.download(ticker, period="5y")
            
            if not mc_data.empty:
                if isinstance(mc_data.columns, pd.MultiIndex):
                    mc_data.columns = mc_data.columns.get_level_values(0)
                
                mc_data['Daily Return'] = mc_data['Close'].pct_change()
                last_price = mc_data['Close'].iloc[-1]
                daily_vol = mc_data['Daily Return'].std()
                
                if sim_type == "Weighted Momentum (Recent Bias)":
                    avg_daily_return = mc_data['Daily Return'].ewm(span=252).mean().iloc[-1]
                    drift = avg_daily_return - (0.5 * daily_vol ** 2)
                elif sim_type == "Long-Term Avg (5-Year)":
                    avg_daily_return = mc_data['Daily Return'].mean()
                    drift = avg_daily_return - (0.5 * daily_vol ** 2)
                else:
                    drift = (0.042 / 252) - (0.5 * daily_vol ** 2)
                
                simulation_df = pd.DataFrame()
                for i in range(simulations):
                    daily_shocks = np.random.normal(drift, daily_vol, time_horizon)
                    price_series = [last_price]
                    for shock in daily_shocks:
                        price_series.append(price_series[-1] * np.exp(shock))
                    simulation_df[f"Sim_{i}"] = price_series

                # Plot
                fig_mc = go.Figure()
                for col in simulation_df.columns[:50]: # Limit lines for performance
                    fig_mc.add_trace(go.Scatter(y=simulation_df[col], mode='lines', line=dict(color='rgba(100, 100, 255, 0.05)'), hoverinfo='skip', showlegend=False))
                
                fig_mc.add_trace(go.Scatter(y=simulation_df.mean(axis=1), mode='lines', line=dict(color='red', width=3), name='Average Path'))
                fig_mc.update_layout(title=f"Monte Carlo: {simulations} Scenarios", xaxis_title="Days", yaxis_title="Price ($)")
                st.plotly_chart(fig_mc, use_container_width=True)
                
                final_prices = simulation_df.iloc[-1]
                col_mc1, col_mc2, col_mc3 = st.columns(3)
                col_mc2.metric("Expected Price", f"${final_prices.mean():.2f}")
                col_mc1.metric("Bearish Case (5%)", f"${np.percentile(final_prices, 5):.2f}")
                col_mc3.metric("Bullish Case (95%)", f"${np.percentile(final_prices, 95):.2f}")
            else:
                st.error("Could not fetch data for simulation.")
