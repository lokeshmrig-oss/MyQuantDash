import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_ta as ta
import praw
from datetime import datetime, timedelta
import yfinance as yf

# Optional transformers (for FinBERT sentiment - large package)
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except (ImportError, Exception):
    TRANSFORMERS_AVAILABLE = False
    pipeline = None

# Optional OpenBB (may not work on Streamlit Cloud due to filesystem restrictions)
try:
    from openbb import obb
    obb.user.preferences.output_type = "dataframe"
    OBB_AVAILABLE = True
except (ImportError, PermissionError, Exception):
    OBB_AVAILABLE = False
    obb = None

# Optional FRED API for economic indicators
try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Pro Macro Quant Workstation")

# --- BLOOMBERG-STYLE CUSTOM CSS ---
st.markdown("""
<style>
    /* Bloomberg Terminal Dark Theme Enhancements */
    .stApp {
        background-color: #0A0E27;
    }

    /* Header styling */
    h1, h2, h3 {
        color: #00D9FF !important;
        font-family: 'Courier New', monospace !important;
        letter-spacing: 1px;
    }

    /* Metric cards */
    [data-testid="stMetricValue"] {
        color: #00FF00 !important;
        font-weight: bold;
        font-family: 'Courier New', monospace;
    }

    [data-testid="stMetricDelta"] {
        font-family: 'Courier New', monospace;
    }

    /* Tables */
    .dataframe {
        font-family: 'Courier New', monospace !important;
        font-size: 13px !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #1A1F3A;
        color: #00D9FF;
        border-radius: 4px;
        font-family: 'Courier New', monospace;
        font-weight: bold;
    }

    .stTabs [aria-selected="true"] {
        background-color: #00D9FF;
        color: #0A0E27;
    }

    /* Buttons */
    .stButton > button {
        background-color: #1A1F3A;
        color: #00D9FF;
        border: 1px solid #00D9FF;
        font-family: 'Courier New', monospace;
        font-weight: bold;
    }

    .stButton > button:hover {
        background-color: #00D9FF;
        color: #0A0E27;
        border: 1px solid #00D9FF;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0F1419;
    }

    /* Text inputs */
    .stTextInput > div > div > input {
        background-color: #1A1F3A;
        color: #E0E6ED;
        border: 1px solid #00D9FF;
        font-family: 'Courier New', monospace;
    }

    /* Success/Warning/Error messages */
    .stSuccess {
        background-color: #1A3A1A;
        color: #00FF00;
    }

    .stWarning {
        background-color: #3A3A1A;
        color: #FFFF00;
    }

    .stError {
        background-color: #3A1A1A;
        color: #FF4444;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Settings")
    debug_mode = st.checkbox("Show Debug Logs", value=False)
    st.caption("Check this if you see blank screens to view raw data/errors.")

    st.markdown("---")
    st.markdown("### 🔄 Data Refresh")
    cache_duration = st.selectbox(
        "Cache Duration",
        options=[300, 900, 1800, 3600],
        index=2,
        format_func=lambda x: f"{x//60} minutes"
    )

    if st.button("🔄 Clear All Cache"):
        st.cache_data.clear()
        st.success("Cache cleared!")

# --- HELPER: ANNUALIZED RETURNS (CAGR) ---
def calculate_perf_table(df):
    """Calculate performance metrics across multiple time periods"""
    df.index = pd.to_datetime(df.index)
    latest_date = df.index[-1]

    periods = {
        "1 Day": (df.index[-2], 1),
        "1 Week": (latest_date - pd.DateOffset(weeks=1), 1),
        "MTD": (df[df.index < latest_date.replace(day=1)].index[-1] if len(df[df.index < latest_date.replace(day=1)]) > 0 else df.index[0], 1),
        "YTD": (df[df.index < pd.Timestamp(year=latest_date.year, month=1, day=1)].index[-1] if len(df[df.index < pd.Timestamp(year=latest_date.year, month=1, day=1)]) > 0 else df.index[0], 1),
        "1 Year": (latest_date - pd.DateOffset(years=1), 1),
        "3 Years": (latest_date - pd.DateOffset(years=3), 3),
        "5 Years": (latest_date - pd.DateOffset(years=5), 5)
    }

    metrics = {}
    current_prices = df.iloc[-1]

    for label, (target_date, n_years) in periods.items():
        try:
            idx = df.index.get_indexer([target_date], method='pad')[0]
            if idx != -1 and idx < len(df):
                past_prices = df.iloc[idx]
                if n_years > 1:  # Annualized (CAGR)
                    val = ((current_prices / past_prices) ** (1 / n_years) - 1) * 100
                else:  # Cumulative
                    val = ((current_prices - past_prices) / past_prices) * 100
                metrics[label] = val
            else:
                metrics[label] = pd.Series(index=df.columns, data=None)
        except:
            metrics[label] = pd.Series(index=df.columns, data=None)

    return pd.DataFrame(metrics)

# --- HELPER: COMPREHENSIVE RISK METRICS ---
def calculate_risk_metrics(returns_series, risk_free_rate=0.04):
    """Calculate comprehensive risk analytics"""
    if len(returns_series) < 2:
        return {}

    # Annualized metrics
    annual_ret = returns_series.mean() * 252
    annual_vol = returns_series.std() * np.sqrt(252)

    # Sharpe Ratio
    sharpe = (annual_ret - risk_free_rate) / annual_vol if annual_vol != 0 else 0

    # Sortino Ratio (downside deviation)
    downside_returns = returns_series[returns_series < 0]
    downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino = (annual_ret - risk_free_rate) / downside_std if downside_std != 0 else 0

    # Maximum Drawdown
    cum_returns = (1 + returns_series).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    max_dd = drawdown.min()

    # Calmar Ratio
    calmar = annual_ret / abs(max_dd) if max_dd != 0 else 0

    # Value at Risk (95% and 99%)
    var_95 = returns_series.quantile(0.05)
    var_99 = returns_series.quantile(0.01)

    # Expected Shortfall (CVaR)
    cvar_95 = returns_series[returns_series <= var_95].mean()

    # Skewness and Kurtosis
    skew = returns_series.skew()
    kurt = returns_series.kurtosis()

    # Win rate
    win_rate = (returns_series > 0).sum() / len(returns_series)

    return {
        'Annual Return': annual_ret,
        'Annual Volatility': annual_vol,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown': max_dd,
        'Calmar Ratio': calmar,
        'VaR (95%)': var_95,
        'VaR (99%)': var_99,
        'CVaR (95%)': cvar_95,
        'Skewness': skew,
        'Kurtosis': kurt,
        'Win Rate': win_rate
    }

# --- HELPER: VOLATILITY REGIME DETECTION ---
def detect_vol_regime(prices, window=20):
    """Identify low/medium/high volatility regimes"""
    returns = prices.pct_change()
    rolling_vol = returns.rolling(window).std() * np.sqrt(252) * 100

    # Define regime thresholds (33rd and 67th percentiles)
    low_threshold = rolling_vol.quantile(0.33)
    high_threshold = rolling_vol.quantile(0.67)

    regime = pd.Series('Medium', index=rolling_vol.index)
    regime[rolling_vol < low_threshold] = 'Low Vol'
    regime[rolling_vol > high_threshold] = 'High Vol'

    return regime, rolling_vol

# --- HELPER: DATA QUALITY MONITORING ---
def show_data_quality(df, location="sidebar"):
    """Display data quality metrics"""
    if df.empty:
        return

    last_update = df.index[-1].strftime('%Y-%m-%d %H:%M') if hasattr(df.index[-1], 'strftime') else str(df.index[-1])
    data_points = len(df)
    missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100 if df.shape[0] * df.shape[1] > 0 else 0

    if location == "sidebar":
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Data Quality")
        st.sidebar.metric("Last Updated", last_update)
        st.sidebar.metric("Data Points", f"{data_points:,}")
        st.sidebar.metric("Missing Data", f"{missing_pct:.2f}%")
        st.sidebar.caption("Source: TIINGO/Yahoo Finance")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Last Updated", last_update)
        c2.metric("Data Points", f"{data_points:,}")
        c3.metric("Missing Data", f"{missing_pct:.2f}%")

# --- DATA FETCHING (ENHANCED) ---
@st.cache_data(ttl=1800)  # 30 minutes default
def get_price_data(tickers, years=5, provider_preference="tiingo"):
    """Enhanced data fetching with TIINGO preference"""
    start_date = (datetime.now() - timedelta(days=years*365 + 30)).strftime('%Y-%m-%d')
    df = pd.DataFrame()
    errors = []
    data_source = "Unknown"

    # 1. Try Tiingo first (preferred for professional use)
    if OBB_AVAILABLE and "tiingo" in st.secrets and provider_preference == "tiingo":
        try:
            if debug_mode: st.write(f"🔍 Fetching from TIINGO: {len(tickers)} tickers...")
            data = obb.equity.price.historical(tickers, provider="tiingo", start_date=start_date)
            df = data.pivot_table(index="date", columns="symbol", values="close")
            data_source = "TIINGO"
            if debug_mode: st.success(f"✅ TIINGO: Loaded {len(df)} rows")
        except Exception as e:
            errors.append(f"Tiingo Failed: {e}")
            if debug_mode: st.warning(f"⚠️ TIINGO failed: {e}")

    # 2. Fallback to Yahoo via OpenBB
    if df.empty and OBB_AVAILABLE:
        try:
            if debug_mode: st.write("🔍 Falling back to Yahoo Finance (OpenBB)...")
            data = obb.equity.price.historical(tickers, provider="yfinance", start_date=start_date)
            df = data.pivot_table(index="date", columns="symbol", values="close")
            data_source = "Yahoo Finance"
            if debug_mode: st.success(f"✅ Yahoo: Loaded {len(df)} rows")
        except Exception as e:
            errors.append(f"Yahoo Failed: {e}")
            if debug_mode: st.warning(f"⚠️ Yahoo (OpenBB) failed: {e}")

    # 3. Direct yfinance fallback (no OpenBB required)
    if df.empty:
        try:
            if debug_mode: st.write(f"🔍 Using direct yfinance API for {len(tickers)} tickers...")
            dfs = []
            failed_tickers = []

            for ticker in tickers:
                try:
                    if debug_mode: st.write(f"  Downloading {ticker}...")
                    ticker_data = yf.download(ticker, start=start_date, progress=False)

                    if debug_mode: st.write(f"  {ticker} downloaded, type: {type(ticker_data)}")

                    if not ticker_data.empty:
                        # Handle different yfinance return structures
                        if isinstance(ticker_data, pd.DataFrame):
                            if 'Close' in ticker_data.columns:
                                close_data = ticker_data['Close']
                            else:
                                # Take the first column if Close doesn't exist
                                close_data = ticker_data.iloc[:, 0]

                            # Set the name directly instead of using rename
                            close_data.name = ticker
                            dfs.append(close_data)
                        elif isinstance(ticker_data, pd.Series):
                            ticker_data.name = ticker
                            dfs.append(ticker_data)
                        else:
                            failed_tickers.append(f"{ticker}(unknown type)")
                    else:
                        failed_tickers.append(f"{ticker}(empty)")
                except Exception as e:
                    import traceback
                    error_detail = traceback.format_exc()
                    failed_tickers.append(f"{ticker}({str(e)[:30]})")
                    if debug_mode:
                        st.write(f"❌ {ticker}: {str(e)}")
                        st.code(error_detail)

            if dfs:
                df = pd.concat(dfs, axis=1)
                data_source = "Yahoo Finance (Direct)"
                if debug_mode:
                    st.success(f"✅ Direct yfinance: Loaded {len(dfs)}/{len(tickers)} tickers, {len(df)} rows")
                    if failed_tickers:
                        st.warning(f"Failed tickers: {', '.join(failed_tickers[:5])}")
            else:
                errors.append(f"No data from any ticker. Failed: {failed_tickers}")
        except Exception as e:
            errors.append(f"Direct yfinance Failed: {e}")
            if debug_mode: st.error(f"❌ Direct yfinance failed: {e}")

    # 3. Final processing
    if not df.empty:
        df.index = pd.to_datetime(df.index)
        df = df.ffill().bfill()  # Forward fill then backward fill
        if debug_mode:
            st.info(f"📡 Data Source: **{data_source}**")
    else:
        if debug_mode: st.error(f"❌ ALL DATA SOURCES FAILED. Errors: {errors}")

    return df

# --- TIINGO SPECIFIC FUNCTIONS ---
@st.cache_data(ttl=1800)
def get_tiingo_fundamentals(ticker):
    """Fetch TIINGO fundamental metrics"""
    if OBB_AVAILABLE and "tiingo" in st.secrets:
        try:
            meta = obb.equity.profile(ticker, provider="tiingo")
            return meta
        except Exception as e:
            if debug_mode: st.warning(f"TIINGO fundamentals failed for {ticker}: {e}")
            return None
    return None

@st.cache_data(ttl=300)  # 5 min cache for FX (more volatile)
def get_fx_data(pairs=['EURUSD=X', 'USDJPY=X', 'GBPUSD=X'], days=365):
    """Get actual FX pairs - using Yahoo FX tickers with individual fetching"""
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    all_data = {}

    # Try fetching each pair individually for better error handling
    for pair in pairs:
        try:
            if debug_mode: st.write(f"Fetching {pair}...")

            # Try OpenBB first if available
            if OBB_AVAILABLE:
                data = obb.equity.price.historical(pair, provider="yfinance", start_date=start_date)
                if not data.empty and 'close' in data.columns:
                    series = data.set_index('date')['close']
                    # Only include if we have valid data (not all NaN)
                    if series.notna().sum() > 0:
                        all_data[pair] = series
                        if debug_mode: st.success(f"✅ {pair}: {len(data)} rows, {series.notna().sum()} valid")
                        continue

            # Direct yfinance fallback
            ticker_data = yf.download(pair, start=start_date, progress=False)
            if not ticker_data.empty:
                # Handle both DataFrame and Series returns
                if isinstance(ticker_data, pd.DataFrame):
                    if 'Close' in ticker_data.columns:
                        series = ticker_data['Close']
                    else:
                        series = ticker_data.iloc[:, 0]
                else:
                    series = ticker_data

                # Ensure it's a Series before checking
                if isinstance(series, pd.Series):
                    if series.notna().sum() > 0:
                        all_data[pair] = series
                        if debug_mode: st.success(f"✅ {pair} (direct): {len(ticker_data)} rows")
                    else:
                        if debug_mode: st.warning(f"⚠️ {pair}: Data returned but all NaN")
                else:
                    if debug_mode: st.warning(f"⚠️ {pair}: Unexpected data type")
            else:
                if debug_mode: st.warning(f"⚠️ {pair}: No data returned")
        except Exception as e:
            if debug_mode: st.error(f"❌ {pair} failed: {e}")

    if all_data:
        df = pd.DataFrame(all_data)
        df.index = pd.to_datetime(df.index)
        # Forward fill and backward fill to handle missing values
        df = df.ffill().bfill()
        return df
    else:
        if debug_mode: st.error("No FX data could be fetched")
        return pd.DataFrame()

@st.cache_data(ttl=60)  # 1 min cache for crypto (very volatile)
def get_crypto_data(tickers=['BTC-USD', 'ETH-USD', 'SOL-USD'], days=365):
    """Get crypto data with shorter cache"""
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    df = pd.DataFrame()

    # Try OpenBB first if available
    if OBB_AVAILABLE:
        try:
            data = obb.crypto.price.historical(tickers, provider="yfinance", start_date=start_date)
            df = data.pivot_table(index="date", columns="symbol", values="close")
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            if debug_mode: st.warning(f"Crypto data (OpenBB) failed: {e}")

    # Direct yfinance fallback
    try:
        dfs = []
        for ticker in tickers:
            ticker_data = yf.download(ticker, start=start_date, progress=False)
            if not ticker_data.empty:
                # Handle both DataFrame and Series returns
                if isinstance(ticker_data, pd.DataFrame):
                    if 'Close' in ticker_data.columns:
                        close_data = ticker_data['Close']
                    else:
                        close_data = ticker_data.iloc[:, 0]
                else:
                    close_data = ticker_data

                # Set name directly
                close_data.name = ticker
                dfs.append(close_data)
        if dfs:
            df = pd.concat(dfs, axis=1)
            df.index = pd.to_datetime(df.index)
            return df
    except Exception as e:
        if debug_mode: st.warning(f"Crypto data (direct yfinance) failed: {e}")

    return pd.DataFrame()

@st.cache_data(ttl=1800)  # 30 min cache for treasury yields
def get_yield_curve_data(years=2):
    """
    Get US Treasury Yield Curve data with TIINGO preference and Yahoo fallback.
    TIINGO-first approach for consistency with professional data sourcing.
    Returns yields already in percentage format.
    """
    start_date = (datetime.now() - timedelta(days=years*365 + 30)).strftime('%Y-%m-%d')
    df = pd.DataFrame()
    data_source = "Unknown"

    # Treasury yield tickers (Yahoo Finance format)
    yield_tickers = {
        "^IRX": "3M Treasury",  # 3-month
        "^FVX": "5Y Treasury",  # 5-year
        "^TNX": "10Y Treasury", # 10-year
        "^TYX": "30Y Treasury"  # 30-year
    }

    # 1. Try TIINGO first (preferred for professional use)
    # Note: As of now, TIINGO doesn't provide treasury yield data, but we structure
    # the code this way for future compatibility and consistency with the codebase
    if OBB_AVAILABLE and "tiingo" in st.secrets:
        try:
            if debug_mode: st.info("🔍 Attempting TIINGO for Treasury Yields...")
            # TIINGO doesn't currently have treasury yields, but we try anyway
            data = obb.equity.price.historical(
                list(yield_tickers.keys()),
                provider="tiingo",
                start_date=start_date
            )
            if not data.empty:
                df = data.pivot_table(index="date", columns="symbol", values="close")
                data_source = "TIINGO"
                if debug_mode: st.success(f"✅ TIINGO: Loaded {len(df)} rows of yield data")
        except Exception as e:
            if debug_mode: st.info(f"ℹ️ TIINGO treasury yields not available (expected): {str(e)[:100]}")

    # 2. Fallback to Yahoo Finance via OpenBB
    if df.empty and OBB_AVAILABLE:
        try:
            if debug_mode: st.info("🔍 Fetching Treasury Yields from Yahoo Finance (OpenBB)...")
            data = obb.equity.price.historical(
                list(yield_tickers.keys()),
                provider="yfinance",
                start_date=start_date
            )
            df = data.pivot_table(index="date", columns="symbol", values="close")
            data_source = "Yahoo Finance"
            if debug_mode: st.success(f"✅ Yahoo Finance: Loaded {len(df)} rows of yield data")
        except Exception as e:
            if debug_mode: st.warning(f"⚠️ Yahoo Finance (OpenBB) treasury yields failed: {e}")

    # 3. Direct yfinance fallback
    if df.empty:
        try:
            if debug_mode: st.info("🔍 Fetching Treasury Yields via direct yfinance...")
            dfs = []
            for ticker in yield_tickers.keys():
                try:
                    ticker_data = yf.download(ticker, start=start_date, progress=False)
                    if not ticker_data.empty:
                        # Handle both DataFrame and Series returns
                        if isinstance(ticker_data, pd.DataFrame):
                            if 'Close' in ticker_data.columns:
                                close_data = ticker_data['Close']
                            else:
                                close_data = ticker_data.iloc[:, 0]
                        else:
                            close_data = ticker_data

                        # Set name directly
                        close_data.name = ticker
                        dfs.append(close_data)
                except Exception as e:
                    if debug_mode: st.warning(f"⚠️ {ticker}: {str(e)}")

            if dfs:
                df = pd.concat(dfs, axis=1)
                data_source = "Yahoo Finance (Direct)"
                if debug_mode: st.success(f"✅ Direct yfinance: Loaded {len(df)} rows of yield data")
        except Exception as e:
            if debug_mode: st.error(f"❌ Direct yfinance treasury yields failed: {e}")

    # Validation and data quality
    if not df.empty:
        df.index = pd.to_datetime(df.index)
        df = df.ffill().bfill()  # Handle missing values

        # Verify we have reasonable yield data (between 0% and 20%)
        valid_data = True
        for col in df.columns:
            if df[col].notna().sum() == 0:
                if debug_mode: st.warning(f"⚠️ No valid data for {yield_tickers.get(col, col)}")
                valid_data = False
            elif df[col].max() > 50 or df[col].min() < 0:
                if debug_mode: st.warning(f"⚠️ Suspicious yield values for {col}")

        if debug_mode and valid_data:
            st.success(f"📊 Yield Data Source: {data_source} | Data points: {len(df)} | Latest 10Y: {df['^TNX'].iloc[-1]:.2f}%")
    else:
        if debug_mode: st.error("❌ Unable to fetch treasury yield data from any source")

    return df

@st.cache_data(ttl=1800)  # 30 min cache
def get_credit_spreads(years=5):
    """
    Calculate credit spreads (HYG-LQD) as recession/stress indicator - yfinance priority
    """
    tickers = ['HYG', 'LQD']
    data = pd.DataFrame()

    # Try yfinance FIRST
    try:
        if debug_mode:
            st.info("🔍 Fetching HYG/LQD via yfinance...")

        hyg_ticker = yf.Ticker('HYG')
        lqd_ticker = yf.Ticker('LQD')

        hyg_data = hyg_ticker.history(period=f"{years}y")
        lqd_data = lqd_ticker.history(period=f"{years}y")

        if not hyg_data.empty and not lqd_data.empty and 'Close' in hyg_data.columns and 'Close' in lqd_data.columns:
            data = pd.DataFrame({
                'HYG': hyg_data['Close'],
                'LQD': lqd_data['Close']
            })
            if debug_mode:
                st.success(f"✅ HYG/LQD loaded via yfinance: {len(data)} days")
    except Exception as e:
        if debug_mode:
            st.warning(f"⚠️ yfinance failed for HYG/LQD: {e}")

    # Fallback to OpenBB
    if data.empty or not all(t in data.columns for t in tickers):
        try:
            if debug_mode:
                st.info("🔍 Trying OpenBB for HYG/LQD...")
            data = get_price_data(tickers, years=years)
        except Exception as e:
            if debug_mode:
                st.error(f"❌ OpenBB failed for HYG/LQD: {e}")

    # Calculate spread ratio if we have data
    if not data.empty and all(t in data.columns for t in tickers):
        data['Spread_Ratio'] = (data['HYG'] / data['LQD']) * 100
        return data

    return pd.DataFrame()

@st.cache_data(ttl=3600)  # 1 hour cache for economic data
def get_fred_indicators():
    """
    Fetch key economic indicators from FRED API.
    Falls back to proxies if FRED API unavailable.
    """
    indicators = {}

    if FRED_AVAILABLE and "fred" in st.secrets:
        try:
            fred = Fred(api_key=st.secrets["fred"]["api_key"])

            # Key macro indicators
            fred_series = {
                'GDP': 'GDP',                          # GDP (Quarterly)
                'UNRATE': 'Unemployment Rate',         # Unemployment %
                'CPIAUCSL': 'CPI',                     # Consumer Price Index
                'FEDFUNDS': 'Fed Funds Rate',          # Fed policy rate
                'T10Y2Y': '10Y-2Y Spread',            # Yield curve spread
                'DEXCHUS': 'USD/CNY',                  # China FX rate
                'UMCSENT': 'Consumer Sentiment'        # U Michigan sentiment
            }

            for series_id, name in fred_series.items():
                try:
                    data = fred.get_series(series_id, observation_start='2020-01-01')
                    indicators[name] = data
                except:
                    pass

            if indicators:
                df = pd.DataFrame(indicators)
                df.index = pd.to_datetime(df.index)
                return df
        except Exception as e:
            if debug_mode: st.warning(f"FRED API fetch failed: {e}")

    # Return empty DataFrame if FRED not available
    return pd.DataFrame()

@st.cache_data(ttl=1800)  # 30 min cache
def get_inflation_breakeven(years=5):
    """
    Calculate inflation breakeven using TIP (TIPS) vs TLT (Treasuries) - yfinance priority
    """
    tickers = ['TIP', 'TLT']
    data = pd.DataFrame()

    # Try yfinance FIRST
    try:
        if debug_mode:
            st.info("🔍 Fetching TIP/TLT via yfinance...")

        tip_ticker = yf.Ticker('TIP')
        tlt_ticker = yf.Ticker('TLT')

        tip_data = tip_ticker.history(period=f"{years}y")
        tlt_data = tlt_ticker.history(period=f"{years}y")

        if not tip_data.empty and not tlt_data.empty and 'Close' in tip_data.columns and 'Close' in tlt_data.columns:
            data = pd.DataFrame({
                'TIP': tip_data['Close'],
                'TLT': tlt_data['Close']
            })
            if debug_mode:
                st.success(f"✅ TIP/TLT loaded via yfinance: {len(data)} days")
    except Exception as e:
        if debug_mode:
            st.warning(f"⚠️ yfinance failed for TIP/TLT: {e}")

    # Fallback to OpenBB
    if data.empty or not all(t in data.columns for t in tickers):
        try:
            if debug_mode:
                st.info("🔍 Trying OpenBB for TIP/TLT...")
            data = get_price_data(tickers, years=years)
        except Exception as e:
            if debug_mode:
                st.error(f"❌ OpenBB failed for TIP/TLT: {e}")

    # Calculate breakeven ratio if we have data
    if not data.empty and all(t in data.columns for t in tickers):
        tip_norm = (data['TIP'] / data['TIP'].iloc[0]) * 100
        tlt_norm = (data['TLT'] / data['TLT'].iloc[0]) * 100
        data['Breakeven_Ratio'] = tip_norm / tlt_norm
        return data

    return pd.DataFrame()

@st.cache_data(ttl=900)  # 15 min cache
def get_dollar_index(years=2):
    """Get US Dollar Index (DXY) strength indicator - yfinance priority"""
    data = pd.DataFrame()

    # Try yfinance FIRST (most reliable for ETFs)
    try:
        if debug_mode:
            st.info("🔍 Fetching UUP via yfinance...")

        uup_ticker = yf.Ticker('UUP')
        uup_data = uup_ticker.history(period=f"{years}y")

        if not uup_data.empty and 'Close' in uup_data.columns:
            data = pd.DataFrame({'UUP': uup_data['Close']})
            if debug_mode:
                st.success(f"✅ UUP loaded via yfinance: {len(data)} days")
            return data
    except Exception as e:
        if debug_mode:
            st.warning(f"⚠️ yfinance failed for UUP: {e}")

    # Fallback to OpenBB if yfinance failed
    if data.empty:
        try:
            if debug_mode:
                st.info("🔍 Trying OpenBB for UUP...")
            data = get_price_data(['UUP'], years=years)
            if not data.empty and 'UUP' in data.columns:
                if debug_mode:
                    st.success(f"✅ UUP loaded via OpenBB: {len(data)} days")
        except Exception as e:
            if debug_mode:
                st.error(f"❌ OpenBB also failed for UUP: {e}")

    return data

@st.cache_data(ttl=1800)  # 30 min cache
def calculate_market_breadth(sector_data):
    """
    Calculate market breadth indicators from sector data.
    Returns percentage of sectors above their moving averages.
    """
    try:
        if sector_data.empty:
            return pd.DataFrame()

        # Calculate 50-day MA for each sector
        ma_50 = sector_data.rolling(window=50).mean()

        # Count sectors above MA
        above_ma = (sector_data > ma_50).sum(axis=1)
        total_sectors = len(sector_data.columns)
        breadth_pct = (above_ma / total_sectors) * 100

        return pd.DataFrame({'Breadth': breadth_pct})
    except Exception as e:
        if debug_mode: st.error(f"Market breadth calculation failed: {e}")
        return pd.DataFrame()

# --- RESOURCES ---
@st.cache_resource
def load_finbert():
    if TRANSFORMERS_AVAILABLE:
        return pipeline("sentiment-analysis", model="ProsusAI/finbert")
    return None

@st.cache_resource
def init_reddit():
    if "reddit" in st.secrets:
        return praw.Reddit(
            client_id=st.secrets["reddit"]["client_id"],
            client_secret=st.secrets["reddit"]["client_secret"],
            user_agent=st.secrets["reddit"]["user_agent"]
        )
    return None

# --- MAIN DASHBOARD ---
st.title("🦅 Professional Macro-Quant Workstation")
st.caption("Powered by TIINGO API • Advanced Analytics for Macro Traders")

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = st.tabs([
    "🌍 Global View",
    "📈 Macro Dashboard",
    "🌎 Global Equity Markets",
    "💱 Currency Dashboard",
    "📦 Commodities Dashboard",
    "🔗 Cross-Asset Correlations",
    "📊 Sector Rotation",
    "🎯 Factor Analysis",
    "💎 Stock Fundamentals",
    "🧠 AI Sentiment",
    "⚠️ Risk Analytics",
    "⚖️ Portfolio Analytics"
])

# ==============================================================================
# TAB 1: ENHANCED GLOBAL VIEW
# ==============================================================================
with tab1:
    st.subheader("🌍 Global Markets Snapshot")
    st.caption("Quick overview of major market benchmarks • Live analysis below")

    # Simplified to 14 major benchmarks - comprehensive coverage in Tabs 3-5
    asset_map = {
        # Major Equity Indices (6)
        "SPY": "🇺🇸 S&P 500",
        "QQQ": "🇺🇸 Nasdaq 100",
        "DIA": "🇺🇸 Dow Jones",
        "EEM": "🌏 Emerging Markets",
        "VGK": "🇪🇺 Europe",
        "EWJ": "🇯🇵 Japan",
        # Fixed Income (4)
        "TLT": "📜 US 20Y Treasury",
        "LQD": "🏢 Investment Grade Bonds",
        "HYG": "⚡ High Yield Bonds",
        "TIP": "🔥 TIPS (Inflation-Protected)",
        # Commodities (2)
        "GLD": "🥇 Gold",
        "USO": "🛢️ Crude Oil",
        # FX (1)
        "UUP": "💵 US Dollar Index",
        # Crypto (1)
        "BTC-USD": "₿ Bitcoin"
    }
    categories = {
        # Equities
        "SPY": "Equities", "QQQ": "Equities", "DIA": "Equities",
        "EEM": "Equities", "VGK": "Equities", "EWJ": "Equities",
        # Fixed Income
        "TLT": "Fixed Income", "LQD": "Fixed Income",
        "HYG": "Fixed Income", "TIP": "Fixed Income",
        # Commodities
        "GLD": "Commodities", "USO": "Commodities",
        # FX
        "UUP": "FX",
        # Crypto
        "BTC-USD": "Crypto"
    }

    df = get_price_data(list(asset_map.keys()), years=5)

    if not df.empty:
        # Performance table
        perf = calculate_perf_table(df)
        perf["Ticker"] = perf.index  # Preserve ticker symbol
        perf["Asset Name"] = [asset_map.get(t, t) for t in perf.index]
        perf["Category"] = [categories.get(t, "Other") for t in perf.index]
        perf = perf.reset_index(drop=True)

        # DYNAMIC MARKET COMMENTARY - Multi-Timeframe Analysis
        st.markdown("### 📊 Live Market Analysis (Multi-Timeframe)")

        # Helper function to get returns across timeframes
        def get_ret(ticker, period):
            if period in perf.columns and ticker in perf["Ticker"].values:
                return perf[perf["Ticker"] == ticker][period].values[0]
            return 0

        # Get multi-period returns
        spy_1w, spy_ytd, spy_1y = get_ret("SPY", "1 Week"), get_ret("SPY", "YTD"), get_ret("SPY", "1 Year")
        eem_1w, eem_ytd, eem_1y = get_ret("EEM", "1 Week"), get_ret("EEM", "YTD"), get_ret("EEM", "1 Year")
        tlt_1w, tlt_ytd, tlt_1y = get_ret("TLT", "1 Week"), get_ret("TLT", "YTD"), get_ret("TLT", "1 Year")
        gld_1w, gld_ytd, gld_1y = get_ret("GLD", "1 Week"), get_ret("GLD", "YTD"), get_ret("GLD", "1 Year")
        uup_1w, uup_ytd = get_ret("UUP", "1 Week"), get_ret("UUP", "YTD")

        # Multi-timeframe risk scoring
        risk_on_1w = (spy_1w > 1.0) + (eem_1w > spy_1w) + (tlt_1w < -0.5) + (uup_1w < -0.3)
        risk_on_ytd = (spy_ytd > 5.0) + (eem_ytd > 3.0) + (spy_ytd > tlt_ytd)
        risk_on_1y = (spy_1y > 10.0) + (eem_1y > 5.0)

        risk_off_1w = (spy_1w < -1.0) + (tlt_1w > 1.0) + (gld_1w > 1.0) + (uup_1w > 0.5)
        risk_off_ytd = (spy_ytd < 0) + (tlt_ytd > 5.0) + (gld_ytd > spy_ytd)

        # Generate multi-timeframe commentary
        commentary = []

        # Display key asset performance across timeframes
        commentary.append(f"**SPY:** 1W {spy_1w:+.1f}% | YTD {spy_ytd:+.1f}% | 1Y {spy_1y:+.1f}%")
        commentary.append(f"**EEM:** 1W {eem_1w:+.1f}% | YTD {eem_ytd:+.1f}% | 1Y {eem_1y:+.1f}%")
        commentary.append(f"**TLT:** 1W {tlt_1w:+.1f}% | YTD {tlt_ytd:+.1f}% | **GLD:** 1W {gld_1w:+.1f}% | YTD {gld_ytd:+.1f}%")
        commentary.append("")

        # Regime determination with trend consistency check
        if risk_on_1w >= 2 and (risk_on_ytd >= 1 or risk_on_1y >= 1):
            st.success("🟢 **RISK-ON Environment**\n\n" + "\n".join(commentary))

            # Trend consistency analysis
            if spy_1w > 0 and spy_ytd > 0 and spy_1y > 0:
                st.info("✅ **Consistent Bull Market:** Positive across all timeframes (1W, YTD, 1Y)\n\n"
                       "→ **Action:** Maximize equity exposure (100%+), favor growth/momentum, long cyclicals, reduce defensive positioning")
            elif spy_1w > 0 and spy_ytd > 0 and spy_1y < 0:
                st.warning("⚠️ **Recovery Phase:** Strong YTD but 1Y negative\n\n"
                          "→ **Action:** Increase equities (80-90%), monitor for sustainability, still in recovery from prior drawdown")
            else:
                st.info("→ **Action:** Favor equities (75-85%), EM if leading, cyclicals, growth sectors")

            # EM leadership check
            if eem_1w > spy_1w and eem_ytd > spy_ytd:
                st.success(f"🌏 **EM LEADING:** EEM outperforming SPY on both 1W and YTD → Global growth, weak USD, commodity supportive\n\n"
                          f"→ **Action:** Rotate to EM equities, materials/XLB, international diversification")

        elif risk_off_1w >= 2 or (risk_off_ytd >= 2):
            st.error("🔴 **RISK-OFF Environment**\n\n" + "\n".join(commentary))

            # Trend deterioration analysis
            if spy_1w < 0 and spy_ytd < 0:
                st.error("🚨 **Deteriorating Trend:** Negative across multiple timeframes\n\n"
                        "→ **Action:** Reduce equities (40-50%), raise cash (20%+), long TLT/GLD, defensive sectors, hedge with puts")
            elif spy_1w < -2 and spy_ytd > 5:
                st.warning("⚠️ **Pullback in Uptrend:** YTD still strong despite recent weakness\n\n"
                          "→ **Action:** Moderate reduction (70% equities), potential dip-buying opportunity if macro supports")
            else:
                st.info("→ **Action:** Defensive positioning (50-60% equities), increase bonds/TLT, raise cash, quality over growth")

            # Safe haven confirmation
            if (tlt_ytd > 5 or gld_ytd > 5):
                st.info(f"💎 **Safe Haven Confirmation:** TLT YTD {tlt_ytd:+.1f}%, GLD YTD {gld_ytd:+.1f}% → Flight to safety confirmed")

        else:
            st.warning("🟡 **MIXED / TRANSITIONAL Market**\n\n" + "\n".join(commentary))

            # Divergence analysis
            if spy_ytd > 5 and spy_1w < -1.5:
                st.info("💡 **Dip in Uptrend:** Recent weakness but YTD remains strong\n\n"
                       "→ **Action:** Maintain 70-75% equity exposure, potential buying opportunity, monitor for trend breakdown")
            elif spy_ytd < 0 and spy_1w > 1.5:
                st.warning("⚠️ **Bounce in Downtrend:** Short-term strength but YTD negative\n\n"
                          "→ **Action:** Caution warranted (60% equities), likely bear market rally, wait for YTD to turn positive")
            else:
                st.info("→ **Action:** Neutral positioning (70%), wait for multi-timeframe alignment, selective opportunities")

        st.markdown("---")

        # Reorder columns to show: Asset Name, Ticker, Category, then performance metrics
        cols = ["Asset Name", "Ticker", "Category", "1 Day", "1 Week", "MTD", "YTD", "1 Year", "3 Years", "5 Years"]
        avail = [c for c in cols if c in perf.columns]
        short_term = [c for c in ["1 Day", "1 Week", "MTD", "YTD"] if c in avail]
        long_term = [c for c in ["1 Year", "3 Years", "5 Years"] if c in avail]

        st.dataframe(
            perf[avail].sort_values("YTD", ascending=False).style
            .format("{:+.2f}%", subset=[c for c in avail if c not in ["Asset Name", "Ticker", "Category"]])
            .background_gradient(cmap="RdYlGn", vmin=-10, vmax=10, subset=short_term)
            .background_gradient(cmap="RdYlGn", vmin=-15, vmax=15, subset=long_term),
            use_container_width=True, height=380, hide_index=True
        )

        # Show data quality
        show_data_quality(df, location="inline")

        st.markdown("---")

        # Navigation hints
        st.markdown("### 🔍 Need More Detail?")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.info("""
            **📊 Tab 3: Global Equity Markets**
            • 30+ countries
            • Regional breakdowns
            • Country rotation signals
            """)

        with col2:
            st.info("""
            **💱 Tab 4: Currency Dashboard**
            • Major FX pairs
            • Technical analysis
            • Trend indicators
            """)

        with col3:
            st.info("""
            **📦 Tab 5: Commodities Dashboard**
            • 17+ commodities
            • Precious metals, energy, agriculture
            • Category performance
            """)
    else:
        st.error("⚠️ Global Data could not be loaded. Check your internet or API limits.")

# ==============================================================================
# TAB 8: ENHANCED PORTFOLIO ANALYTICS
# ==============================================================================
with tab12:
    st.subheader("Advanced Portfolio Analytics & Backtesting")

    with st.expander("🏗️ Build Your Portfolio", expanded=True):
        num_assets = st.number_input("Number of Assets", min_value=2, max_value=10, value=3)
        cols = st.columns(num_assets)
        user_portfolio = {}

        for i in range(num_assets):
            with cols[i]:
                defaults = ["SPY", "TLT", "GLD", "NVDA", "BTC-USD", "AAPL", "MSFT", "GOOG", "AMZN", "META"]
                def_ticker = defaults[i] if i < len(defaults) else "SPY"
                t_ticker = st.text_input(f"Asset {i+1}", def_ticker, key=f"t_{i}").upper()
                t_weight = st.number_input(f"Weight %", min_value=0, max_value=100, value=30, key=f"w_{i}")
                user_portfolio[t_ticker] = t_weight

    if st.button("🚀 Run Comprehensive Backtest"):
        total_raw_weight = sum(user_portfolio.values())
        if total_raw_weight == 0:
            st.error("Total weight cannot be zero.")
        else:
            final_weights = {k: v/total_raw_weight for k, v in user_portfolio.items()}

            if total_raw_weight != 100:
                st.info(f"⚙️ Weights normalized from {total_raw_weight}% to 100%")

            tickers_to_fetch = list(final_weights.keys())
            if "SPY" not in tickers_to_fetch:
                tickers_to_fetch.append("SPY")

            sim_df = get_price_data(tickers_to_fetch, years=5)

            if not sim_df.empty:
                returns = sim_df.pct_change().dropna()

                # Calculate portfolio daily returns
                port_daily_ret = pd.Series(0, index=returns.index)
                for t, w in final_weights.items():
                    if t in returns.columns:
                        port_daily_ret += returns[t] * w

                # Cumulative returns
                port_cum = (1 + port_daily_ret).cumprod() * 10000
                spy_cum = (1 + returns["SPY"]).cumprod() * 10000

                # Basic metrics
                cagr = ((port_cum.iloc[-1] / port_cum.iloc[0]) ** (1/5) - 1) * 100
                spy_cagr = ((spy_cum.iloc[-1] / spy_cum.iloc[0]) ** (1/5) - 1) * 100

                # COMPREHENSIVE RISK METRICS
                risk_metrics = calculate_risk_metrics(port_daily_ret)
                spy_risk_metrics = calculate_risk_metrics(returns["SPY"])

                # Display key metrics
                st.markdown("### 📊 Performance Overview")
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Portfolio CAGR", f"{cagr:.2f}%")
                c2.metric("Benchmark (SPY)", f"{spy_cagr:.2f}%", delta=f"{cagr-spy_cagr:.2f}%")
                c3.metric("Sharpe Ratio", f"{risk_metrics['Sharpe Ratio']:.2f}")
                c4.metric("Max Drawdown", f"{risk_metrics['Max Drawdown']*100:.2f}%")
                c5.metric("Win Rate", f"{risk_metrics['Win Rate']*100:.1f}%")

                # Growth chart
                st.markdown("### 📈 Growth of $10,000")
                fig_growth = go.Figure()
                fig_growth.add_trace(go.Scatter(
                    x=port_cum.index, y=port_cum,
                    name="Your Portfolio",
                    line=dict(color="#00CC96", width=2.5)
                ))
                fig_growth.add_trace(go.Scatter(
                    x=spy_cum.index, y=spy_cum,
                    name="S&P 500",
                    line=dict(color="gray", dash="dot", width=2)
                ))
                fig_growth.update_layout(
                    title="Portfolio Value Over Time (5 Years)",
                    yaxis_title="Portfolio Value ($)",
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig_growth, use_container_width=True)

                # UNDERWATER PLOT (Drawdown)
                st.markdown("### 🌊 Underwater Plot (Drawdown Analysis)")
                cum_rets = (1 + port_daily_ret).cumprod()
                running_max = cum_rets.expanding().max()
                underwater = (cum_rets - running_max) / running_max * 100

                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(
                    x=underwater.index,
                    y=underwater,
                    fill='tozeroy',
                    name='Drawdown',
                    line=dict(color='red')
                ))
                fig_dd.update_layout(
                    title="Drawdown from Peak (%)",
                    yaxis_title="Drawdown (%)",
                    height=350
                )
                fig_dd.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig_dd, use_container_width=True)

                # MONTHLY RETURNS HEATMAP
                st.markdown("### 🗓️ Monthly Returns Heatmap")
                monthly_rets = port_daily_ret.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100

                if len(monthly_rets) > 0:
                    monthly_table = monthly_rets.to_frame('Return')
                    monthly_table['Year'] = monthly_table.index.year
                    monthly_table['Month'] = monthly_table.index.month

                    # Create pivot table
                    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    pivot = monthly_table.pivot(index='Year', columns='Month', values='Return')
                    pivot.columns = [month_names[int(m)-1] for m in pivot.columns]

                    # Add annual returns
                    annual_rets = port_daily_ret.resample('Y').apply(lambda x: (1 + x).prod() - 1) * 100
                    pivot['Annual'] = annual_rets.values

                    fig_heatmap = px.imshow(
                        pivot,
                        color_continuous_scale='RdYlGn',
                        color_continuous_midpoint=0,
                        title="Monthly Returns (%)",
                        aspect='auto',
                        labels=dict(color="Return (%)")
                    )
                    fig_heatmap.update_layout(height=400)
                    st.plotly_chart(fig_heatmap, use_container_width=True)

                # DETAILED RISK COMPARISON TABLE
                st.markdown("### ⚖️ Detailed Risk Metrics Comparison")
                comparison_df = pd.DataFrame({
                    'Your Portfolio': risk_metrics,
                    'S&P 500': spy_risk_metrics
                })
                comparison_df['Difference'] = comparison_df['Your Portfolio'] - comparison_df['S&P 500']

                # Format the dataframe
                formatted_comparison = comparison_df.style.format({
                    'Your Portfolio': '{:.4f}',
                    'S&P 500': '{:.4f}',
                    'Difference': '{:+.4f}'
                }).background_gradient(cmap='RdYlGn', subset=['Difference'])

                st.dataframe(formatted_comparison, use_container_width=True)

                # ROLLING SHARPE RATIO
                st.markdown("### 📉 Rolling 252-Day Sharpe Ratio")
                rolling_sharpe_port = (
                    port_daily_ret.rolling(252).mean() * 252 /
                    (port_daily_ret.rolling(252).std() * np.sqrt(252))
                )
                rolling_sharpe_spy = (
                    returns["SPY"].rolling(252).mean() * 252 /
                    (returns["SPY"].rolling(252).std() * np.sqrt(252))
                )

                fig_rolling = go.Figure()
                fig_rolling.add_trace(go.Scatter(
                    x=rolling_sharpe_port.index,
                    y=rolling_sharpe_port,
                    name='Portfolio',
                    line=dict(color='#00CC96')
                ))
                fig_rolling.add_trace(go.Scatter(
                    x=rolling_sharpe_spy.index,
                    y=rolling_sharpe_spy,
                    name='S&P 500',
                    line=dict(color='gray', dash='dot')
                ))
                fig_rolling.add_hline(y=0, line_dash="dash", line_color="red")
                fig_rolling.update_layout(
                    title="Rolling Sharpe Ratio (252 Days)",
                    yaxis_title="Sharpe Ratio",
                    height=350
                )
                st.plotly_chart(fig_rolling, use_container_width=True)

            else:
                st.error("Could not fetch data for backtest.")

# ==============================================================================
# TAB 7: SECTOR ROTATION
# ==============================================================================
with tab7:
    st.subheader("📊 US Sector Rotation Dashboard")

    sec_map = {
        "XLE": "Energy", "XLK": "Tech", "XLF": "Financials",
        "XLV": "Health", "XLI": "Industrials", "XLP": "Staples",
        "XLU": "Utilities", "XLY": "Discretionary", "XLB": "Materials",
        "XLRE": "Real Estate", "XLC": "Comms"
    }

    df_sectors = get_price_data(list(sec_map.keys()), years=5)

    if not df_sectors.empty:
        sec_matrix = calculate_perf_table(df_sectors)
        sec_matrix["Ticker"] = sec_matrix.index  # Preserve ticker symbol
        sec_matrix["Sector"] = [sec_map.get(t, t) for t in sec_matrix.index]
        sec_matrix = sec_matrix.reset_index(drop=True)

        # LIVE SECTOR ROTATION ANALYSIS - Multi-Timeframe
        st.markdown("### 📊 Live Sector Leadership Analysis (Multi-Timeframe)")

        if "1 Week" in sec_matrix.columns and "YTD" in sec_matrix.columns:
            # Get top sectors across timeframes
            top_3_1w = sec_matrix.nlargest(3, "1 Week")[["Sector", "1 Week"]]
            top_3_ytd = sec_matrix.nlargest(3, "YTD")[["Sector", "YTD"]]

            top_1w = top_3_1w["Sector"].tolist()
            top_ytd = top_3_ytd["Sector"].tolist()

            # Economic cycle indicators
            cyclical_sectors = ["Tech", "Discretionary", "Financials", "Industrials"]
            defensive_sectors = ["Utilities", "Staples", "Health"]
            late_cycle_sectors = ["Energy", "Materials"]

            # Multi-timeframe scoring
            cyclical_1w = sum(1 for s in top_1w if s in cyclical_sectors)
            cyclical_ytd = sum(1 for s in top_ytd if s in cyclical_sectors)
            defensive_1w = sum(1 for s in top_1w if s in defensive_sectors)
            defensive_ytd = sum(1 for s in top_ytd if s in defensive_sectors)
            late_1w = sum(1 for s in top_1w if s in late_cycle_sectors)
            late_ytd = sum(1 for s in top_ytd if s in late_cycle_sectors)

            # Format for display
            top_1w_str = ', '.join([f"{row['Sector']} ({row['1 Week']:+.1f}%)" for _, row in top_3_1w.iterrows()])
            top_ytd_str = ', '.join([f"{row['Sector']} ({row['YTD']:+.1f}%)" for _, row in top_3_ytd.iterrows()])

            commentary = []
            commentary.append(f"**Leading 1W:** {top_1w_str}")
            commentary.append(f"**Leading YTD:** {top_ytd_str}")
            commentary.append("")

            # Sustained cyclical leadership
            if cyclical_1w >= 2 and cyclical_ytd >= 2:
                st.success("🟢 **SUSTAINED EARLY-MID CYCLE EXPANSION**\n\n" + "\n".join(commentary))
                st.success("✅ **Consistent cyclical leadership across 1W and YTD** → Economic expansion confirmed\n\n"
                          "→ **Action:** MAXIMUM risk-on positioning. Overweight XLK (tech), XLY (discretionary), XLF (financials), XLI (industrials). Increase beta to 1.2+. Minimize defensives.")

            # Emerging cyclical leadership
            elif cyclical_1w >= 2 and cyclical_ytd < 2:
                st.info("🟢 **EMERGING CYCLICAL ROTATION**\n\n" + "\n".join(commentary))
                st.info("⚠️ **Recent cyclical strength (1W) not yet confirmed YTD** → Potential cycle turn\n\n"
                       "→ **Action:** Increase cyclicals (60-70% of equity allocation), monitor for YTD confirmation. Early rotation opportunity if sustained.")

            # Sustained defensive leadership
            elif defensive_1w >= 2 and defensive_ytd >= 2:
                st.error("🔴 **SUSTAINED DEFENSIVE ROTATION**\n\n" + "\n".join(commentary))
                st.error("🚨 **Persistent defensive leadership across timeframes** → Recession risk elevated\n\n"
                        "→ **Action:** DEFENSIVE positioning. Overweight XLU (utilities), XLP (staples), XLV (healthcare). Reduce beta to 0.7 or less. Raise cash (20%+).")

            # Defensive rotation emerging
            elif defensive_1w >= 2 and defensive_ytd < 2:
                st.warning("🟠 **EMERGING DEFENSIVE ROTATION**\n\n" + "\n".join(commentary))
                st.warning("⚠️ **Recent defensive strength signals risk-off shift** → Monitor closely\n\n"
                          "→ **Action:** Begin reducing cyclicals (to 50%), increase defensives (30%), prepare for potential recession. Watch credit spreads/VIX.")

            # Late cycle / inflation
            elif late_1w >= 1 and late_ytd >= 1 and ("Energy" in top_1w or "Energy" in top_ytd):
                st.warning("🟠 **LATE CYCLE / INFLATION REGIME**\n\n" + "\n".join(commentary))
                st.info("⚠️ **Energy/Materials persistent leadership** → Inflation pressures, late-cycle dynamics\n\n"
                       "→ **Action:** Overweight XLE (energy), XLB (materials), GLD (gold), TIPS/TIP. Reduce duration. Monitor Fed policy.")

            # Rotation transition
            else:
                st.info("🟡 **SECTOR ROTATION TRANSITION**\n\n" + "\n".join(commentary))

                # Check for divergence
                if cyclical_1w >= 2 and defensive_ytd >= 2:
                    st.warning("⚠️ **DIVERGENT SIGNALS:** Recent cyclical strength vs YTD defensive leadership\n\n"
                              "→ **Action:** Balanced positioning (60% equities), wait for multi-timeframe alignment. Could be false start or genuine turn.")
                elif defensive_1w >= 2 and cyclical_ytd >= 2:
                    st.warning("⚠️ **DIVERGENT SIGNALS:** Recent defensive shift vs YTD cyclical strength\n\n"
                              "→ **Action:** Begin de-risking (reduce to 70% equities), increase defensives. Potential cycle peak.")
                else:
                    st.info("→ **Action:** Balanced sector allocation, monitor for clear rotation pattern. Stay diversified (25% each: cyclical/defensive/materials/financials).")

        st.markdown("---")

        # Reorder columns: Sector, Ticker, then performance metrics
        cols = ["Sector", "Ticker", "1 Day", "1 Week", "MTD", "YTD", "1 Year", "3 Years", "5 Years"]
        avail = [c for c in cols if c in sec_matrix.columns]
        short_term = [c for c in ["1 Day", "1 Week", "MTD", "YTD"] if c in avail]
        long_term = [c for c in ["1 Year", "3 Years", "5 Years"] if c in avail]

        st.dataframe(
            sec_matrix[avail].sort_values("YTD", ascending=False).style
            .format("{:+.2f}%", subset=[c for c in avail if c not in ["Sector", "Ticker"]])
            .background_gradient(cmap="RdYlGn", vmin=-10, vmax=10, subset=short_term)
            .background_gradient(cmap="RdYlGn", vmin=-20, vmax=20, subset=long_term),
            use_container_width=True, height=450, hide_index=True
        )

        # Risk-Return Scatter for Sectors
        st.markdown("---")
        st.markdown("### 📊 Sector Risk vs Reward Analysis (1 Year)")

        # Calculate volatility using ticker symbols as keys
        daily_ret = df_sectors.pct_change().tail(252)
        vol_series = daily_ret.std() * (252**0.5) * 100

        # Create scatter data with proper alignment
        scatter_data = []
        for ticker in sec_matrix["Ticker"]:
            if ticker in df_sectors.columns:
                # Get the row for this ticker
                row = sec_matrix[sec_matrix["Ticker"] == ticker].iloc[0]
                if "1 Year" in row and ticker in vol_series.index:
                    scatter_data.append({
                        "Sector": row["Sector"],
                        "Ticker": ticker,
                        "Return": row["1 Year"],
                        "Volatility": vol_series[ticker]
                    })

        if scatter_data:
            scatter_df = pd.DataFrame(scatter_data)

            fig_risk = px.scatter(
                scatter_df,
                x="Volatility",
                y="Return",
                text="Sector",
                color="Return",
                color_continuous_scale="RdYlGn",
                title="Sector Risk-Return Profile (Past Year)",
                labels={"Return": "1-Year Return (%)", "Volatility": "Annual Volatility (%)"},
                hover_data={"Ticker": True, "Return": ":.2f", "Volatility": ":.2f"}
            )
            fig_risk.update_traces(textposition='top center', marker=dict(size=15))
            fig_risk.update_layout(height=500)
            st.plotly_chart(fig_risk, use_container_width=True)
        else:
            st.warning("Insufficient data for risk-return chart.")
    else:
        st.error("⚠️ Sector data unavailable. Check debug mode.")

# ==============================================================================
# TAB 8: FACTOR ANALYSIS
# ==============================================================================
with tab8:
    st.subheader("🎯 Smart Beta Factor Analysis")

    fac_map = {
        "MTUM": "Momentum", "VLUE": "Value", "USMV": "Low Volatility",
        "QUAL": "Quality", "SIZE": "Size (Small Cap)", "IWM": "Russell 2000 (Small Cap)"
    }

    df_factors = get_price_data(list(fac_map.keys()), years=5)

    if not df_factors.empty:
        fac_matrix = calculate_perf_table(df_factors)
        fac_matrix["Ticker"] = fac_matrix.index
        fac_matrix["Factor"] = [fac_map.get(t, t) for t in fac_matrix.index]
        fac_matrix = fac_matrix.reset_index(drop=True)

        # LIVE FACTOR ANALYSIS - Multi-Timeframe
        st.markdown("### 📊 Live Factor Performance Analysis (Multi-Timeframe)")

        # Get SPY for relative comparison
        spy_data = get_price_data(["SPY"], years=5)
        if not spy_data.empty:
            spy_perf = calculate_perf_table(spy_data)
            spy_1w = spy_perf.loc["SPY", "1 Week"] if "1 Week" in spy_perf.columns else 0
            spy_ytd = spy_perf.loc["SPY", "YTD"] if "YTD" in spy_perf.columns else 0
        else:
            spy_1w, spy_ytd = 0, 0

        if "1 Week" in fac_matrix.columns and "YTD" in fac_matrix.columns:
            # Identify top factors across timeframes
            top_1w = fac_matrix.nlargest(1, "1 Week").iloc[0]
            top_ytd = fac_matrix.nlargest(1, "YTD").iloc[0]

            top_1w_name = top_1w["Factor"]
            top_1w_perf = top_1w["1 Week"]
            top_ytd_name = top_ytd["Factor"]
            top_ytd_perf = top_ytd["YTD"]

            # Display multi-timeframe leadership
            commentary = []
            commentary.append(f"**Leading 1W:** {top_1w_name} ({top_1w_perf:+.1f}%) vs SPY ({spy_1w:+.1f}%)")
            commentary.append(f"**Leading YTD:** {top_ytd_name} ({top_ytd_perf:+.1f}%) vs SPY ({spy_ytd:+.1f}%)")
            commentary.append("")

            # Sustained Momentum leadership
            if "Momentum" in top_1w_name and "Momentum" in top_ytd_name:
                st.success("🟢 **SUSTAINED MOMENTUM REGIME**\n\n" + "\n".join(commentary))
                st.success("✅ **Momentum dominant across all timeframes** → Persistent risk-on, trend-following environment\n\n"
                          "→ **Action:** MAXIMUM momentum allocation (40%+ of equity). Long MTUM, ride winners with trailing stops, cut losers at -8%. Pure trend-following strategy.")

            # Sustained Low-Vol leadership (risk-off)
            elif "Low Volatility" in top_1w_name and "Low Volatility" in top_ytd_name:
                st.error("🔴 **SUSTAINED DEFENSIVE REGIME**\n\n" + "\n".join(commentary))
                st.error("🚨 **Low-Vol persistent leadership** → Sustained risk-off, bear market conditions\n\n"
                        "→ **Action:** DEFENSIVE positioning (50% USMV, 30% cash, 20% bonds). Minimize beta. Avoid momentum/growth. Wait for regime change.")

            # Momentum emerging (regime change)
            elif "Momentum" in top_1w_name and "Low Volatility" in top_ytd_name:
                st.info("🟢 **EMERGING MOMENTUM REGIME**\n\n" + "\n".join(commentary))
                st.warning("⚠️ **Recent momentum strength vs YTD low-vol** → Potential bull market resumption\n\n"
                          "→ **Action:** Begin rotating to momentum (20-30%). Wait for YTD confirmation before full allocation. Early-stage bull signal.")

            # Momentum fading (regime change)
            elif "Low Volatility" in top_1w_name and "Momentum" in top_ytd_name:
                st.warning("🔴 **MOMENTUM REGIME FADING**\n\n" + "\n".join(commentary))
                st.error("⚠️ **Recent low-vol strength vs YTD momentum** → Potential bull market peak\n\n"
                        "→ **Action:** Reduce momentum exposure (to 20%), increase defensives/USMV (30%), raise cash. Prepare for risk-off transition.")

            # Value leadership (late cycle or bear market recovery)
            elif "Value" in top_1w_name or "Value" in top_ytd_name:
                st.warning("🟠 **VALUE FACTOR LEADERSHIP**\n\n" + "\n".join(commentary))

                if "Value" in top_ytd_name and top_ytd_perf > spy_ytd + 5:
                    st.info("✅ **Strong YTD value outperformance** → Late-cycle rotation or bear market\n\n"
                           "→ **Action:** Overweight VLUE (30%), value sectors (financials/energy), contrarian plays. Mean reversion active.")
                else:
                    st.info("→ **Action:** Tactical value allocation (20%), monitor for persistence. Could be temporary rotation.")

            # Small-cap leadership (risk-on)
            elif ("Size" in top_1w_name or "Russell" in top_1w_name) and top_1w_perf > spy_1w + 1:
                st.success("🟢 **SMALL-CAP LEADERSHIP**\n\n" + "\n".join(commentary))

                if ("Size" in top_ytd_name or "Russell" in top_ytd_name) and top_ytd_perf > spy_ytd:
                    st.success("✅ **Sustained small-cap outperformance** → Broad-based risk appetite\n\n"
                              "→ **Action:** Overweight IWM/small-caps (25%), domestic focus. Economic expansion confirmed.")
                else:
                    st.info("→ **Action:** Increase small-caps (15-20%), monitor for YTD confirmation of risk-on regime.")

            # Quality leadership (uncertainty)
            elif "Quality" in top_1w_name or "Quality" in top_ytd_name:
                st.info("⚖️ **QUALITY FACTOR LEADERSHIP**\n\n" + "\n".join(commentary))
                st.info("💡 **Flight to quality** → Uncertain market, favor fundamentals\n\n"
                       "→ **Action:** Overweight QUAL (30%), high-ROE companies, strong balance sheets. Avoid speculation.")

            else:
                st.info("🟡 **FACTOR ROTATION TRANSITION**\n\n" + "\n".join(commentary))
                st.info("→ **Action:** Balanced multi-factor approach (20% each: Momentum/Value/Quality/Low-Vol/Size). Wait for clearer factor regime.")

        st.markdown("---")

        # Reorder columns: Factor, Ticker, then performance metrics
        cols = ["Factor", "Ticker", "1 Day", "1 Week", "MTD", "YTD", "1 Year", "3 Years", "5 Years"]
        avail = [c for c in cols if c in fac_matrix.columns]
        short_term = [c for c in ["1 Day", "1 Week", "MTD", "YTD"] if c in avail]
        long_term = [c for c in ["1 Year", "3 Years", "5 Years"] if c in avail]

        st.dataframe(
            fac_matrix[avail].sort_values("YTD", ascending=False).style
            .format("{:+.2f}%", subset=[c for c in avail if c not in ["Factor", "Ticker"]])
            .background_gradient(cmap="RdYlGn", vmin=-10, vmax=10, subset=short_term)
            .background_gradient(cmap="RdYlGn", vmin=-20, vmax=20, subset=long_term),
            use_container_width=True, height=350, hide_index=True
        )

        # Risk-Return Scatter for Factors
        st.markdown("---")
        st.markdown("### 📊 Factor Risk vs Reward Analysis (1 Year)")

        # Calculate volatility using ticker symbols as keys
        daily_ret = df_factors.pct_change().tail(252)
        vol_series = daily_ret.std() * (252**0.5) * 100

        # Create scatter data with proper alignment
        scatter_data = []
        for ticker in fac_matrix["Ticker"]:
            if ticker in df_factors.columns:
                # Get the row for this ticker
                row = fac_matrix[fac_matrix["Ticker"] == ticker].iloc[0]
                if "1 Year" in row and ticker in vol_series.index:
                    scatter_data.append({
                        "Factor": row["Factor"],
                        "Ticker": ticker,
                        "Return": row["1 Year"],
                        "Volatility": vol_series[ticker]
                    })

        if scatter_data:
            scatter_df = pd.DataFrame(scatter_data)

            fig_risk = px.scatter(
                scatter_df,
                x="Volatility",
                y="Return",
                text="Factor",
                color="Return",
                color_continuous_scale="RdYlGn",
                title="Factor Risk-Return Profile (Past Year)",
                labels={"Return": "1-Year Return (%)", "Volatility": "Annual Volatility (%)"},
                hover_data={"Ticker": True, "Return": ":.2f", "Volatility": ":.2f"}
            )
            fig_risk.update_traces(textposition='top center', marker=dict(size=15))
            fig_risk.update_layout(height=500)
            st.plotly_chart(fig_risk, use_container_width=True)
        else:
            st.warning("Insufficient data for risk-return chart.")

        # Factor comparison vs SPY
        if "SPY" not in df_factors.columns:
            spy_data = get_price_data(["SPY"], years=5)
            if not spy_data.empty:
                df_factors = pd.concat([df_factors, spy_data], axis=1)

        if "SPY" in df_factors.columns:
            st.markdown("---")
            st.markdown("### 📈 Factor Performance vs SPY Benchmark")

            # Normalize all prices to 100
            norm_factors = (df_factors / df_factors.iloc[0]) * 100

            fig_perf = go.Figure()
            for col in norm_factors.columns:
                if col != "SPY":
                    fig_perf.add_trace(go.Scatter(
                        x=norm_factors.index,
                        y=norm_factors[col],
                        name=fac_map.get(col, col),
                        mode='lines'
                    ))

            # Add SPY as benchmark with distinct styling
            fig_perf.add_trace(go.Scatter(
                x=norm_factors.index,
                y=norm_factors["SPY"],
                name="SPY (Benchmark)",
                mode='lines',
                line=dict(color='white', width=3, dash='dash')
            ))

            fig_perf.update_layout(
                title="Factor ETFs vs SPY (Normalized to 100)",
                yaxis_title="Indexed Value",
                height=500,
                hovermode='x unified'
            )
            st.plotly_chart(fig_perf, use_container_width=True)

    else:
        st.error("⚠️ Factor data unavailable. Check debug mode.")

# ==============================================================================
# TAB 5: STOCK FUNDAMENTALS (EXISTING)
# ==============================================================================
with tab9:
    st.subheader("Fundamental Deep Dive")
    t = st.text_input("Ticker:", "NVDA", key="t5").upper()

    if t:
        import yfinance as yf
        stock = yf.Ticker(t)

        col_chart, col_fund = st.columns([2, 1])
        with col_chart:
            try:
                hist = stock.history(period="1y")
                if not hist.empty:
                    hist.index = pd.to_datetime(hist.index).tz_localize(None)
                    hist["SMA_50"] = hist["Close"].rolling(50).mean()
                    hist["SMA_200"] = hist["Close"].rolling(200).mean()

                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=hist.index,
                        open=hist['Open'],
                        high=hist['High'],
                        low=hist['Low'],
                        close=hist['Close'],
                        name='Price'
                    ))
                    fig.add_trace(go.Scatter(
                        x=hist.index, y=hist["SMA_50"],
                        name="50D MA", line=dict(color="orange", width=1.5)
                    ))
                    fig.add_trace(go.Scatter(
                        x=hist.index, y=hist["SMA_200"],
                        name="200D MA", line=dict(color="blue", width=1.5)
                    ))
                    fig.update_layout(title=f"{t} Price Action with Moving Averages", height=450)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"No price history found for {t}")
            except Exception as e:
                st.error(f"Chart Error: {e}")

        with col_fund:
            try:
                info = stock.info
                if len(info) > 1:
                    st.markdown("#### 🔑 Key Ratios")
                    def fmt(val, is_pct=False):
                        return f"{val*100:.2f}%" if is_pct and val else f"{val:.2f}" if val else "N/A"

                    r1, r2 = st.columns(2)
                    r1.metric("P/E (Fwd)", fmt(info.get("forwardPE")))
                    r2.metric("PEG Ratio", fmt(info.get("pegRatio")))

                    r3, r4 = st.columns(2)
                    r3.metric("Margins", fmt(info.get("profitMargins"), True))
                    r4.metric("ROE", fmt(info.get("returnOnEquity"), True))

                    r5, r6 = st.columns(2)
                    r5.metric("Debt/Eq", fmt(info.get("debtToEquity")))
                    r6.metric("Beta", fmt(info.get("beta")))
                else:
                    st.warning("No fundamental data found.")
            except Exception as e:
                st.warning(f"Fundamentals Error: {e}")

        st.markdown("---")
        st.markdown("#### 📊 Financial Health")
        try:
            inc = stock.income_stmt
            if not inc.empty:
                inc = inc.T
                if "Total Revenue" in inc.columns and "Net Income" in inc.columns:
                    inc_display = inc[["Total Revenue", "Net Income"]].sort_index()
                    inc_display["Revenue ($B)"] = inc_display["Total Revenue"] / 1e9
                    inc_display["Net Income ($B)"] = inc_display["Net Income"] / 1e9
                    st.plotly_chart(
                        px.bar(inc_display, y=["Revenue ($B)", "Net Income ($B)"],
                               barmode="group", height=400),
                        use_container_width=True
                    )
            else:
                st.info("Financial statements unavailable for this ticker.")
        except Exception as e:
            st.info(f"Financials not available: {e}")

# ==============================================================================
# TAB 6: PROFESSIONAL SENTIMENT ANALYSIS (TIINGO-POWERED)
# ==============================================================================
with tab10:
    st.subheader("📡 Professional News Sentiment Analysis")
    st.caption("Powered by TIINGO News Feed & FinBERT AI Model")

    # Input controls
    col_input1, col_input2, col_input3 = st.columns([2, 1, 1])
    with col_input1:
        t_s = st.text_input("Enter Ticker Symbol:", "TSLA", key="t6").upper()
    with col_input2:
        news_limit = st.selectbox("Articles to Analyze", [10, 20, 30, 50], index=1)
    with col_input3:
        time_range = st.selectbox("Time Range", ["1 Week", "1 Month", "3 Months", "6 Months"], index=1)

    # Time range mapping
    days_map = {"1 Week": 7, "1 Month": 30, "3 Months": 90, "6 Months": 180}
    days_back = days_map[time_range]

    if st.button("🚀 Run Sentiment Analysis", use_container_width=True):
        with st.spinner(f"Analyzing {news_limit} news articles from TIINGO..."):
            nlp = load_finbert()
            news_df = pd.DataFrame()

            # Primary: TIINGO News API (Most reliable for quant trading)
            if OBB_AVAILABLE and "tiingo" in st.secrets:
                try:
                    if debug_mode: st.info("📡 Fetching from TIINGO News API...")
                    news = obb.news.company(symbol=t_s, provider="tiingo", limit=news_limit)
                    if not news.empty:
                        news_df = news[['date', 'title', 'text' if 'text' in news.columns else 'title']].copy()
                        news_df = news_df.rename(columns={'text': 'description'})
                        if debug_mode: st.success(f"✅ TIINGO: Retrieved {len(news_df)} articles")
                except Exception as e:
                    if debug_mode: st.warning(f"⚠️ TIINGO News error: {e}")

            # Fallback: Yahoo Finance News via OpenBB
            if news_df.empty and OBB_AVAILABLE:
                try:
                    if debug_mode: st.info("📰 Falling back to Yahoo Finance News...")
                    news = obb.news.company(symbol=t_s, provider="yfinance", limit=news_limit)
                    if not news.empty:
                        news_df = news[['date', 'title']].copy()
                        if debug_mode: st.success(f"✅ Yahoo: Retrieved {len(news_df)} articles")
                except Exception as e:
                    if debug_mode: st.warning(f"⚠️ Yahoo News error: {e}")

            # If OpenBB not available, show info message
            if news_df.empty and not OBB_AVAILABLE:
                st.info("📰 News sentiment requires OpenBB package. Using basic Reddit sentiment only.")

            if not news_df.empty:
                # Filter by date range
                news_df['date'] = pd.to_datetime(news_df['date'])
                cutoff_date = datetime.now() - timedelta(days=days_back)
                news_df = news_df[news_df['date'] >= cutoff_date]

                if len(news_df) == 0:
                    st.warning(f"No news found in the last {time_range}. Try a longer time range.")
                elif nlp is None:
                    st.warning("⚠️ FinBERT sentiment analysis requires the transformers package, which is not available. Please install transformers and torch locally to use AI sentiment analysis.")
                else:
                    # Run FinBERT sentiment analysis
                    st.info(f"🤖 Running FinBERT AI analysis on {len(news_df)} articles...")

                    scores, labels, confidence = [], [], []
                    for idx, row in news_df.iterrows():
                        try:
                            # Use title + description if available
                            text = row['title']
                            if 'description' in row and pd.notna(row['description']):
                                text = f"{row['title']}. {row['description']}"

                            # Run FinBERT
                            res = nlp(text[:512])[0]  # Truncate to model max length

                            # Convert to signed sentiment score
                            if res['label'] == 'positive':
                                sc = res['score']
                            elif res['label'] == 'negative':
                                sc = -res['score']
                            else:
                                sc = 0

                            scores.append(sc)
                            labels.append(res['label'])
                            confidence.append(res['score'])
                        except:
                            scores.append(0)
                            labels.append("neutral")
                            confidence.append(0)

                    news_df["Sentiment"] = scores
                    news_df["Label"] = labels
                    news_df["Confidence"] = confidence

                    # === KEY METRICS ===
                    st.markdown("### 📊 Sentiment Overview")
                    avg_sentiment = np.mean(scores)
                    pos_count = (news_df['Label'] == 'positive').sum()
                    neg_count = (news_df['Label'] == 'negative').sum()
                    neu_count = (news_df['Label'] == 'neutral').sum()

                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Average Sentiment", f"{avg_sentiment:.3f}",
                             delta="Bullish" if avg_sentiment > 0.1 else "Bearish" if avg_sentiment < -0.1 else "Neutral")
                    m2.metric("Positive News", f"{pos_count} ({pos_count/len(news_df)*100:.0f}%)")
                    m3.metric("Negative News", f"{neg_count} ({neg_count/len(news_df)*100:.0f}%)")
                    m4.metric("Neutral News", f"{neu_count} ({neu_count/len(news_df)*100:.0f}%)")
                    m5.metric("Avg Confidence", f"{np.mean(confidence):.2%}")

                    st.markdown("---")

                    # === SENTIMENT TREND CHART ===
                    st.markdown("### 📈 Sentiment Trend Over Time")

                    # Create rolling average
                    news_df_sorted = news_df.sort_values('date')
                    news_df_sorted['Sentiment_MA7'] = news_df_sorted['Sentiment'].rolling(window=min(7, len(news_df)), min_periods=1).mean()

                    fig_trend = go.Figure()

                    # Individual sentiment points
                    fig_trend.add_trace(go.Scatter(
                        x=news_df_sorted['date'],
                        y=news_df_sorted['Sentiment'],
                        mode='markers',
                        name='Individual Articles',
                        marker=dict(
                            size=8,
                            color=news_df_sorted['Sentiment'],
                            colorscale='RdYlGn',
                            colorbar=dict(title="Sentiment"),
                            cmin=-1,
                            cmax=1,
                            line=dict(width=1, color='white')
                        ),
                        text=news_df_sorted['title'],
                        hovertemplate='<b>%{text}</b><br>Sentiment: %{y:.3f}<br>Date: %{x}<extra></extra>'
                    ))

                    # 7-day moving average
                    fig_trend.add_trace(go.Scatter(
                        x=news_df_sorted['date'],
                        y=news_df_sorted['Sentiment_MA7'],
                        mode='lines',
                        name='7-Article Moving Avg',
                        line=dict(color='cyan', width=3)
                    ))

                    fig_trend.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
                    fig_trend.update_layout(
                        title=f"{t_s} News Sentiment Trend ({time_range})",
                        xaxis_title="Date",
                        yaxis_title="Sentiment Score",
                        height=450,
                        hovermode='closest'
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)

                    st.markdown("---")

                    # === SENTIMENT DISTRIBUTION ===
                    col_dist1, col_dist2 = st.columns(2)

                    with col_dist1:
                        st.markdown("### 📊 Sentiment Distribution")
                        fig_hist = px.histogram(
                            news_df,
                            x="Sentiment",
                            nbins=20,
                            color_discrete_sequence=['#00D9FF'],
                            title="Sentiment Score Distribution"
                        )
                        fig_hist.add_vline(x=0, line_dash="dash", line_color="white")
                        fig_hist.update_layout(height=350)
                        st.plotly_chart(fig_hist, use_container_width=True)

                    with col_dist2:
                        st.markdown("### 🥧 Label Breakdown")
                        label_counts = news_df['Label'].value_counts()
                        fig_pie = px.pie(
                            values=label_counts.values,
                            names=label_counts.index,
                            color=label_counts.index,
                            color_discrete_map={'positive': '#00FF00', 'negative': '#FF4444', 'neutral': '#888888'},
                            title="News Sentiment Categories"
                        )
                        fig_pie.update_layout(height=350)
                        st.plotly_chart(fig_pie, use_container_width=True)

                    st.markdown("---")

                    # === DETAILED NEWS TABLE ===
                    st.markdown("### 📰 Detailed News Analysis")

                    # Format for display
                    display_df = news_df[['date', 'title', 'Label', 'Sentiment', 'Confidence']].copy()
                    display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d %H:%M')
                    display_df = display_df.sort_values('date', ascending=False)
                    display_df['Sentiment'] = display_df['Sentiment'].round(3)
                    display_df['Confidence'] = display_df['Confidence'].round(3)

                    st.dataframe(
                        display_df.style
                        .background_gradient(cmap='RdYlGn', subset=['Sentiment'], vmin=-1, vmax=1)
                        .background_gradient(cmap='Greens', subset=['Confidence'], vmin=0, vmax=1),
                        use_container_width=True,
                        hide_index=True,
                        height=400
                    )

                    # === DATA SOURCE INFO ===
                    st.info(f"📡 **Data Source:** {'TIINGO News API' if 'tiingo' in st.secrets else 'Yahoo Finance'} | "
                           f"**Articles Analyzed:** {len(news_df)} | **Time Range:** {time_range}")

            else:
                st.error(f"❌ No news articles found for {t_s}. Please check:\n"
                        "- Ticker symbol is correct\n"
                        "- TIINGO API key is configured\n"
                        "- API rate limits not exceeded")

# ==============================================================================
# TAB 2: MACRO DASHBOARD
# ==============================================================================
with tab2:
    st.subheader("📈 Macro Economic Dashboard")
    st.caption("Risk indicators, economic data, and macro regime analysis")

    # YIELD CURVE
    st.markdown("### 🏦 US Treasury Yield Curve")

    curve_tickers = {
        "^IRX": ("3M", 0.25),
        "^FVX": ("5Y", 5),
        "^TNX": ("10Y", 10),
        "^TYX": ("30Y", 30)
    }

    # Fetch yield data using TIINGO-first approach with Yahoo fallback
    curve_data = get_yield_curve_data(years=2)

    if not curve_data.empty:
        # Current yield curve (Yahoo Finance yields are already in %)
        latest_yields = curve_data.iloc[-1]
        prev_month_yields = curve_data.iloc[-22] if len(curve_data) > 22 else latest_yields

        maturities = [curve_tickers[t][1] for t in curve_tickers.keys()]
        labels = [curve_tickers[t][0] for t in curve_tickers.keys()]
        current_yields = [latest_yields[t] for t in curve_tickers.keys()]
        prev_yields = [prev_month_yields[t] for t in curve_tickers.keys()]

        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(
            x=maturities,
            y=current_yields,
            mode='lines+markers',
            name='Current',
            line=dict(color='blue', width=3),
            marker=dict(size=10)
        ))
        fig_curve.add_trace(go.Scatter(
            x=maturities,
            y=prev_yields,
            mode='lines+markers',
            name='1 Month Ago',
            line=dict(color='gray', width=2, dash='dot'),
            marker=dict(size=8)
        ))

        fig_curve.update_layout(
            xaxis_title="Maturity (Years)",
            yaxis_title="Yield (%)",
            title="Treasury Yield Curve Comparison",
            height=400,
            xaxis=dict(type='log')  # Log scale for better visualization
        )
        st.plotly_chart(fig_curve, use_container_width=True)

        # Yield curve metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("3M Yield", f"{current_yields[0]:.2f}%",
                 delta=f"{current_yields[0] - prev_yields[0]:.2f}%")
        c2.metric("5Y Yield", f"{current_yields[1]:.2f}%",
                 delta=f"{current_yields[1] - prev_yields[1]:.2f}%")
        c3.metric("10Y Yield", f"{current_yields[2]:.2f}%",
                 delta=f"{current_yields[2] - prev_yields[2]:.2f}%")
        spread_10y_3m = current_yields[2] - current_yields[0]
        c4.metric("10Y-3M Spread", f"{spread_10y_3m:.2f}%",
                 help="Positive = Normal, Negative = Inverted (recession signal)")

        # LIVE YIELD CURVE INTERPRETATION
        if spread_10y_3m < 0:
            st.error(f"🚨 **INVERTED YIELD CURVE** (10Y-3M = {spread_10y_3m:.2f}%) → **Recession Warning!** Historically predicts recession within 6-18 months. Reduce cyclical exposure, favor defensives.")
        elif spread_10y_3m < 0.5:
            st.warning(f"⚠️ **FLATTENING CURVE** (10Y-3M = {spread_10y_3m:.2f}%) → Late-cycle signal. Economy slowing, Fed may be over-tightening. Monitor for inversion.")
        elif spread_10y_3m < 1.5:
            st.info(f"⚖️ **NORMAL CURVE** (10Y-3M = {spread_10y_3m:.2f}%) → Healthy yield curve. Balanced growth expectations. Mid-cycle environment.")
        else:
            st.success(f"✅ **STEEP CURVE** (10Y-3M = {spread_10y_3m:.2f}%) → Strong expansion signal. Economy accelerating, growth ahead. Favor cyclicals, financials, small-caps.")

        # Historical spread chart
        st.markdown("### 📊 Historical 10Y-3M Spread (Recession Indicator)")
        spread_history = curve_data["^TNX"] - curve_data["^IRX"]  # Already in %

        fig_spread = go.Figure()
        fig_spread.add_trace(go.Scatter(
            x=spread_history.index,
            y=spread_history,
            fill='tozeroy',
            name='Yield Spread',
            line=dict(color='purple')
        ))
        fig_spread.add_hline(y=0, line_dash="dash", line_color="red",
                            annotation_text="Inversion Line")
        fig_spread.update_layout(
            yaxis_title="Spread (%)",
            title="10Y-3M Treasury Spread (Negative = Inverted)",
            height=350
        )
        st.plotly_chart(fig_spread, use_container_width=True)

    st.markdown("---")

    # VIX FEAR INDEX
    st.markdown("### ⚠️ VIX Fear Index")
    st.caption("Market volatility and fear gauge")

    # Direct yfinance fetch for VIX (more reliable than OpenBB for indices)
    vix_data = pd.DataFrame()
    try:
        if debug_mode:
            st.info("🔍 Fetching VIX data via yfinance...")

        vix_ticker = yf.Ticker("^VIX")
        vix_raw = vix_ticker.history(period="2y")

        if not vix_raw.empty and "Close" in vix_raw.columns:
            vix_data = pd.DataFrame(vix_raw["Close"])
            vix_data.columns = ["^VIX"]
            if debug_mode:
                st.success(f"✅ VIX data loaded: {len(vix_data)} days")
        else:
            if debug_mode:
                st.warning("⚠️ yfinance returned empty data, trying OpenBB...")
            vix_data = get_price_data(['^VIX'], years=2)
    except Exception as e:
        if debug_mode:
            st.error(f"❌ VIX fetch error: {type(e).__name__}: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
        # Try fallback
        vix_data = get_price_data(['^VIX'], years=2)

    if not vix_data.empty and '^VIX' in vix_data.columns:
        # VIX chart
        fig_vix = go.Figure()
        fig_vix.add_trace(go.Scatter(
            x=vix_data.index,
            y=vix_data['^VIX'],
            fill='tozeroy',
            name='VIX',
            line=dict(color='red', width=2)
        ))

        # Add reference levels
        fig_vix.add_hline(y=20, line_dash="dash", line_color="yellow", annotation_text="Elevated (20)")
        fig_vix.add_hline(y=30, line_dash="dash", line_color="orange", annotation_text="High (30)")

        fig_vix.update_layout(
            title="VIX (CBOE Volatility Index) - 2 Year History",
            yaxis_title="VIX Level",
            height=350,
            hovermode='x unified'
        )
        st.plotly_chart(fig_vix, use_container_width=True)

        # Current VIX level
        current_vix = vix_data['^VIX'].iloc[-1]
        avg_vix = vix_data['^VIX'].mean()

        c1, c2, c3 = st.columns(3)
        c1.metric("Current VIX", f"{current_vix:.2f}")
        c2.metric("2Y Average", f"{avg_vix:.2f}")

        if current_vix < 15:
            c3.metric("Market Fear", "🟢 Low", help="Complacency zone")
            st.success("✅ **Low Fear:** Market complacent, consider tail-risk hedges")
        elif current_vix < 20:
            c3.metric("Market Fear", "🟡 Normal", help="Healthy market")
            st.info("⚖️ **Normal Fear:** Typical market volatility")
        elif current_vix < 30:
            c3.metric("Market Fear", "🟠 Elevated", help="Caution warranted")
            st.warning("⚠️ **Elevated Fear:** Increased uncertainty, reduce leverage")
        else:
            c3.metric("Market Fear", "🔴 High", help="Crisis mode")
            st.error("🚨 **High Fear:** Panic conditions, wait for stabilization")

    st.markdown("---")

    # VOLATILITY REGIME ANALYSIS
    st.markdown("### 🌡️ Market Volatility Regime (SPY)")
    st.caption("Volatility regime classification for macro risk assessment")

    spy_data = pd.DataFrame()

    try:
        if debug_mode:
            st.info("🔍 Fetching SPY data via yfinance for volatility regime...")

        # Direct yfinance fetch to bypass caching issues
        spy_ticker = yf.Ticker("SPY")
        spy_raw = spy_ticker.history(period="3y")

        if not spy_raw.empty and "Close" in spy_raw.columns:
            # Convert to expected format (single column DataFrame with ticker name)
            spy_data = pd.DataFrame(spy_raw["Close"])
            spy_data.columns = ["SPY"]

            if debug_mode:
                st.success(f"✅ SPY data loaded: {len(spy_data)} days from {spy_data.index[0].date()} to {spy_data.index[-1].date()}")
        else:
            if debug_mode:
                st.warning("⚠️ yfinance returned empty DataFrame or missing Close column")

    except Exception as e:
        spy_data = pd.DataFrame()
        if debug_mode:
            st.error(f"❌ SPY fetch error: {type(e).__name__}: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

    if not spy_data.empty and "SPY" in spy_data.columns:
        regime, vol = detect_vol_regime(spy_data["SPY"], window=20)

        # Create subplot with price and vol regime
        fig_regime = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('SPY Price with Regime Colors', 'Rolling 20-Day Volatility'),
            row_heights=[0.6, 0.4]
        )

        # Price chart with regime coloring
        for reg_type, color in [('Low Vol', 'green'), ('Medium', 'orange'), ('High Vol', 'red')]:
            mask = regime == reg_type
            fig_regime.add_trace(
                go.Scatter(
                    x=spy_data.index[mask],
                    y=spy_data["SPY"][mask],
                    mode='markers',
                    marker=dict(color=color, size=3),
                    name=reg_type,
                    showlegend=True
                ),
                row=1, col=1
            )

        # Volatility chart
        fig_regime.add_trace(
            go.Scatter(x=vol.index, y=vol, name='Volatility', line=dict(color='purple')),
            row=2, col=1
        )

        fig_regime.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig_regime.update_yaxes(title_text="Volatility (%)", row=2, col=1)
        fig_regime.update_layout(height=600, showlegend=True)

        st.plotly_chart(fig_regime, use_container_width=True)

        # Current regime info with interpretation
        current_regime = regime.iloc[-1]
        current_vol = vol.iloc[-1]

        # Regime interpretation for macro traders
        regime_desc = {
            "Low Vol": "📉 **Low Volatility:** Favorable for risk assets, carry trades, and leverage strategies",
            "Medium": "⚖️ **Medium Volatility:** Balanced risk environment, selective positioning recommended",
            "High Vol": "⚠️ **High Volatility:** Risk-off environment, favor defensive assets and hedges"
        }

        st.info(f"📍 **Current Regime:** {current_regime} | **20D Volatility:** {current_vol:.2f}%")
        st.markdown(regime_desc.get(current_regime, ""))
    else:
        st.warning("⚠️ Volatility regime analysis unavailable. Unable to fetch SPY data.")
        if not debug_mode:
            st.info("💡 Enable 'Show Debug Logs' in the sidebar to see detailed error information.")

    st.markdown("---")

    # CREDIT SPREAD ANALYSIS (Recession Indicator)
    st.markdown("### 📊 Credit Spread Analysis (HYG/LQD)")
    st.caption("Widening spreads = increasing credit stress and recession risk")

    credit_data = get_credit_spreads(years=5)

    if not credit_data.empty and 'Spread_Ratio' in credit_data.columns:
        # Create credit spread visualization
        fig_credit = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Credit ETF Prices', 'HYG/LQD Spread Ratio (Higher = More Stress)'),
            row_heights=[0.5, 0.5]
        )

        # Top: HYG and LQD prices
        fig_credit.add_trace(
            go.Scatter(x=credit_data.index, y=credit_data['HYG'],
                      name='HYG (High Yield)', line=dict(color='red')),
            row=1, col=1
        )
        fig_credit.add_trace(
            go.Scatter(x=credit_data.index, y=credit_data['LQD'],
                      name='LQD (Invest Grade)', line=dict(color='green')),
            row=1, col=1
        )

        # Bottom: Spread ratio
        fig_credit.add_trace(
            go.Scatter(x=credit_data.index, y=credit_data['Spread_Ratio'],
                      name='Spread Ratio', fill='tozeroy', line=dict(color='orange')),
            row=2, col=1
        )

        fig_credit.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig_credit.update_yaxes(title_text="Ratio", row=2, col=1)
        fig_credit.update_layout(height=500, showlegend=True)

        st.plotly_chart(fig_credit, use_container_width=True)

        # Current spread metrics
        current_ratio = credit_data['Spread_Ratio'].iloc[-1]
        avg_ratio = credit_data['Spread_Ratio'].mean()
        std_ratio = credit_data['Spread_Ratio'].std()

        c1, c2, c3 = st.columns(3)
        c1.metric("Current Spread Ratio", f"{current_ratio:.2f}")
        c2.metric("5Y Average", f"{avg_ratio:.2f}")
        z_score = (current_ratio - avg_ratio) / std_ratio
        c3.metric("Z-Score", f"{z_score:.2f}",
                 help="Positive = wider than average (more stress)")

        # LIVE CREDIT SPREAD INTERPRETATION
        if z_score > 1.5:
            st.error(f"🚨 **HIGH CREDIT STRESS** (Z-score = {z_score:.2f}) → Spreads {z_score:.1f} standard deviations above normal! **Recession risk elevated.** Corporate credit deteriorating. Favor quality over junk, reduce credit exposure, increase cash.")
        elif z_score > 0.5:
            st.warning(f"⚠️ **MODERATE STRESS** (Z-score = {z_score:.2f}) → Spreads moderately wide. Credit conditions tightening. Monitor closely for further deterioration. Consider reducing high-yield exposure.")
        elif z_score > -0.5:
            st.info(f"⚖️ **NORMAL CONDITIONS** (Z-score = {z_score:.2f}) → Credit spreads in normal range. Healthy corporate credit market. Balanced risk environment.")
        else:
            st.success(f"✅ **LOW STRESS / TIGHT SPREADS** (Z-score = {z_score:.2f}) → Credit conditions excellent! Risk appetite strong. Favorable for corporate bonds, high-yield. Economic expansion likely.")
    else:
        st.warning("⚠️ Credit spread data unavailable")

    st.markdown("---")

    # INFLATION BREAKEVEN ANALYSIS
    st.markdown("### 🔥 Inflation Breakeven Analysis (TIPS vs Treasuries)")
    st.caption("Rising TIP/TLT ratio = rising inflation expectations")

    inflation_data = get_inflation_breakeven(years=5)

    if not inflation_data.empty and 'Breakeven_Ratio' in inflation_data.columns:
        fig_inflation = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('TIP (TIPS) vs TLT (Treasuries) - Normalized', 'Breakeven Ratio (Inflation Expectations)'),
            row_heights=[0.5, 0.5]
        )

        # Top: Normalized prices
        tip_norm = (inflation_data['TIP'] / inflation_data['TIP'].iloc[0]) * 100
        tlt_norm = (inflation_data['TLT'] / inflation_data['TLT'].iloc[0]) * 100

        fig_inflation.add_trace(
            go.Scatter(x=inflation_data.index, y=tip_norm,
                      name='TIP (TIPS)', line=dict(color='red', width=2)),
            row=1, col=1
        )
        fig_inflation.add_trace(
            go.Scatter(x=inflation_data.index, y=tlt_norm,
                      name='TLT (Treasuries)', line=dict(color='blue', width=2)),
            row=1, col=1
        )

        # Bottom: Breakeven ratio
        fig_inflation.add_trace(
            go.Scatter(x=inflation_data.index, y=inflation_data['Breakeven_Ratio'],
                      name='Breakeven Ratio', fill='tozeroy', line=dict(color='purple')),
            row=2, col=1
        )

        fig_inflation.update_yaxes(title_text="Indexed (Base=100)", row=1, col=1)
        fig_inflation.update_yaxes(title_text="Ratio", row=2, col=1)
        fig_inflation.update_layout(height=500, showlegend=True)

        st.plotly_chart(fig_inflation, use_container_width=True)

        # Current metrics
        current_breakeven = inflation_data['Breakeven_Ratio'].iloc[-1]
        prev_breakeven = inflation_data['Breakeven_Ratio'].iloc[-22] if len(inflation_data) > 22 else current_breakeven
        change = current_breakeven - prev_breakeven

        col1, col2, col3 = st.columns(3)
        col1.metric("Current Breakeven", f"{current_breakeven:.3f}")
        col2.metric("1M Change", f"{change:.3f}",
                   delta="Rising" if change > 0 else "Falling")
        col3.metric("Trend", "📈 Inflation Up" if change > 0.01 else "📉 Inflation Down" if change < -0.01 else "➡️ Stable")

    else:
        st.warning("⚠️ Inflation breakeven data unavailable")

    st.markdown("---")

    # DOLLAR STRENGTH INDEX
    st.markdown("### 💵 US Dollar Strength Index (UUP)")
    st.caption("Strong dollar = headwind for commodities and EM; tailwind for importers")

    dollar_data = get_dollar_index(years=2)

    if not dollar_data.empty and 'UUP' in dollar_data.columns:
        # Normalize to 100
        dollar_norm = (dollar_data['UUP'] / dollar_data['UUP'].iloc[0]) * 100

        fig_dollar = go.Figure()
        fig_dollar.add_trace(go.Scatter(
            x=dollar_data.index,
            y=dollar_norm,
            fill='tozeroy',
            name='Dollar Index (UUP)',
            line=dict(color='green', width=2)
        ))

        fig_dollar.update_layout(
            title="US Dollar Strength (Normalized to 100)",
            yaxis_title="Indexed Value",
            height=350,
            hovermode='x unified'
        )
        st.plotly_chart(fig_dollar, use_container_width=True)

        # Calculate trend
        current_level = dollar_norm.iloc[-1]
        ma_50 = dollar_norm.rolling(50).mean().iloc[-1]
        ma_200 = dollar_norm.rolling(200).mean().iloc[-1]

        c1, c2, c3 = st.columns(3)
        c1.metric("Current Level", f"{current_level:.1f}")
        c2.metric("50D MA", f"{ma_50:.1f}")
        c3.metric("200D MA", f"{ma_200:.1f}")

        if current_level > ma_50 and ma_50 > ma_200:
            st.info("💪 **Strong Dollar:** Bullish trend - favorable for US consumers, headwind for commodities/EM")
        elif current_level < ma_50 and ma_50 < ma_200:
            st.warning("📉 **Weak Dollar:** Bearish trend - favorable for commodities/EM, headwind for US consumers")
        else:
            st.info("🔄 **Transitioning:** Mixed signals - watch for trend confirmation")
    else:
        st.warning("⚠️ Dollar index data unavailable")

    # FRED ECONOMIC INDICATORS (if available)
    if FRED_AVAILABLE and "fred" in st.secrets:
        st.markdown("---")
        st.markdown("### 📈 FRED Economic Indicators")
        st.caption("Key macro data from Federal Reserve Economic Data")

        fred_data = get_fred_indicators()

        if not fred_data.empty:
            # Display latest values
            latest_vals = fred_data.iloc[-1]

            cols = st.columns(len(latest_vals))
            for i, (indicator, value) in enumerate(latest_vals.items()):
                if pd.notna(value):
                    cols[i].metric(indicator, f"{value:.2f}")

            # Show trend chart for selected indicator
            selected_indicator = st.selectbox(
                "Select Indicator to Chart",
                options=fred_data.columns.tolist()
            )

            if selected_indicator:
                fig_fred = go.Figure()
                fig_fred.add_trace(go.Scatter(
                    x=fred_data.index,
                    y=fred_data[selected_indicator],
                    mode='lines',
                    name=selected_indicator,
                    line=dict(color='cyan', width=2)
                ))

                fig_fred.update_layout(
                    title=f"{selected_indicator} Historical Trend",
                    yaxis_title="Value",
                    height=350,
                    hovermode='x unified'
                )
                st.plotly_chart(fig_fred, use_container_width=True)
        else:
            st.info("💡 Configure FRED API key in secrets.toml for economic indicators")

# ==============================================================================
# TAB 3: GLOBAL EQUITY MARKETS
# ==============================================================================
with tab3:
    st.subheader("🌎 Global Equity Markets - Regional & Country Analysis")
    st.caption("Track major equity markets across all regions")

    # Define comprehensive equity markets
    equity_markets = {
        # North America
        "SPY": "🇺🇸 US - S&P 500",
        "QQQ": "🇺🇸 US - Nasdaq 100",
        "DIA": "🇺🇸 US - Dow Jones",
        "IWM": "🇺🇸 US - Russell 2000",
        "EWC": "🇨🇦 Canada",
        "EWW": "🇲🇽 Mexico",
        # Europe
        "VGK": "🇪🇺 Europe (Broad)",
        "EWU": "🇬🇧 United Kingdom",
        "EWG": "🇩🇪 Germany",
        "EWQ": "🇫🇷 France",
        "EWI": "🇮🇹 Italy",
        "EWP": "🇪🇸 Spain",
        "EWL": "🇨🇭 Switzerland",
        "EWD": "🇸🇪 Sweden",
        # Asia-Pacific Developed
        "EWJ": "🇯🇵 Japan",
        "EWY": "🇰🇷 South Korea",
        "EWA": "🇦🇺 Australia",
        "EWH": "🇭🇰 Hong Kong",
        "EWS": "🇸🇬 Singapore",
        # Emerging Markets
        "EEM": "🌏 Emerging Markets (Broad)",
        "INDA": "🇮🇳 India",
        "MCHI": "🇨🇳 China",
        "EWZ": "🇧🇷 Brazil",
        "EZA": "🇿🇦 South Africa",
        "RSX": "🇷🇺 Russia",
        "EWT": "🇹🇼 Taiwan",
        "EIDO": "🇮🇩 Indonesia",
        "THD": "🇹🇭 Thailand"
    }

    regions = {
        # North America
        "SPY": "North America", "QQQ": "North America", "DIA": "North America",
        "IWM": "North America", "EWC": "North America", "EWW": "North America",
        # Europe
        "VGK": "Europe", "EWU": "Europe", "EWG": "Europe", "EWQ": "Europe",
        "EWI": "Europe", "EWP": "Europe", "EWL": "Europe", "EWD": "Europe",
        # Asia-Pacific Developed
        "EWJ": "Asia-Pacific Dev", "EWY": "Asia-Pacific Dev", "EWA": "Asia-Pacific Dev",
        "EWH": "Asia-Pacific Dev", "EWS": "Asia-Pacific Dev",
        # Emerging Markets
        "EEM": "Emerging Markets", "INDA": "Emerging Markets", "MCHI": "Emerging Markets",
        "EWZ": "Emerging Markets", "EZA": "Emerging Markets", "RSX": "Emerging Markets",
        "EWT": "Emerging Markets", "EIDO": "Emerging Markets", "THD": "Emerging Markets"
    }

    # Fetch data
    equity_data = get_price_data(list(equity_markets.keys()), years=5)

    if not equity_data.empty:
        # Calculate performance
        equity_perf = calculate_perf_table(equity_data)
        equity_perf["Ticker"] = equity_perf.index
        equity_perf["Market"] = [equity_markets.get(t, t) for t in equity_perf.index]
        equity_perf["Region"] = [regions.get(t, "Other") for t in equity_perf.index]
        equity_perf = equity_perf.reset_index(drop=True)

        # LIVE GLOBAL EQUITY ROTATION ANALYSIS - Multi-Timeframe
        st.markdown("### 📊 Live Global Equity Rotation Analysis (Multi-Timeframe)")

        # Helper to get performance
        def get_reg_perf(ticker, period):
            if period in equity_perf.columns and ticker in equity_perf["Ticker"].values:
                return equity_perf[equity_perf["Ticker"] == ticker][period].values[0]
            return 0

        # Get multi-period performance for key regions
        spy_1w, spy_ytd, spy_1y = get_reg_perf("SPY", "1 Week"), get_reg_perf("SPY", "YTD"), get_reg_perf("SPY", "1 Year")
        eem_1w, eem_ytd, eem_1y = get_reg_perf("EEM", "1 Week"), get_reg_perf("EEM", "YTD"), get_reg_perf("EEM", "1 Year")
        vgk_1w, vgk_ytd, vgk_1y = get_reg_perf("VGK", "1 Week"), get_reg_perf("VGK", "YTD"), get_reg_perf("VGK", "1 Year")
        ewj_1w, ewj_ytd, ewj_1y = get_reg_perf("EWJ", "1 Week"), get_reg_perf("EWJ", "YTD"), get_reg_perf("EWJ", "1 Year")

        # Display multi-timeframe performance
        commentary = []
        commentary.append(f"**US (SPY):** 1W {spy_1w:+.1f}% | YTD {spy_ytd:+.1f}% | 1Y {spy_1y:+.1f}%")
        commentary.append(f"**EM (EEM):** 1W {eem_1w:+.1f}% | YTD {eem_ytd:+.1f}% | 1Y {eem_1y:+.1f}%")
        commentary.append(f"**Europe (VGK):** 1W {vgk_1w:+.1f}% | YTD {vgk_ytd:+.1f}% | 1Y {vgk_1y:+.1f}%")
        commentary.append(f"**Japan (EWJ):** 1W {ewj_1w:+.1f}% | YTD {ewj_ytd:+.1f}% | 1Y {ewj_1y:+.1f}%")
        commentary.append("")

        # Multi-timeframe rotation analysis
        em_leading_1w = eem_1w > spy_1w
        em_leading_ytd = eem_ytd > spy_ytd
        em_leading_1y = eem_1y > spy_1y

        if em_leading_1w and em_leading_ytd:
            st.success("🟢 **EMERGING MARKETS LEADING**\n\n" + "\n".join(commentary))

            if em_leading_1w and em_leading_ytd and em_leading_1y:
                st.info("✅ **Sustained EM Outperformance:** Leading across all timeframes (1W, YTD, 1Y)\n\n"
                       "→ **Action:** Maximum EM allocation (30-40%), long INDA/MCHI/EWZ, materials/XLB, commodities. Structural global growth trend.")
            elif em_leading_1w and em_leading_ytd and not em_leading_1y:
                st.info("⚠️ **Recent EM Strength:** Leading 1W/YTD but lagging 1Y → Emerging rotation\n\n"
                       "→ **Action:** Increase EM allocation (20-25%), monitor for sustainability, could be start of multi-year rotation")
            else:
                st.info("→ **Action:** Rotate to EM (20-30%), favor global growth cyclicals, materials, energy, international diversification")

        elif spy_1w > eem_1w + 2.0 and spy_ytd > eem_ytd:
            st.warning("🔵 **US EXCEPTIONALISM**\n\n" + "\n".join(commentary))

            if spy_ytd > 10 and eem_ytd < 5:
                st.warning("⚠️ **Persistent US Dominance:** Multi-year US outperformance\n\n"
                          "→ **Action:** Overweight US (70-80%), large-cap growth/QQQ, reduce international, monitor for mean reversion")
            else:
                st.info("→ **Action:** Favor US (60-70%), quality large-caps, reduce EM exposure, dollar strength likely persists")

        elif eem_1w < -2 and spy_1w < -2 and vgk_1w < -2:
            st.error("🔴 **GLOBAL EQUITY WEAKNESS**\n\n" + "\n".join(commentary))
            st.error("🚨 **Broad-based selloff across all regions** → Systemic risk-off\n\n"
                    "→ **Action:** Reduce global equities (40-50%), raise cash (20%+), long TLT/GLD, hedge portfolios, defensive sectors")

        else:
            st.info("⚖️ **REGIONAL DIVERGENCE**\n\n" + "\n".join(commentary))

            # Identify specific rotation opportunities
            if vgk_1w > spy_1w and vgk_ytd > spy_ytd:
                st.success("🇪🇺 **Europe Leading:** VGK outperforming US → EUR strength, ECB dynamics\n\n"
                          "→ **Action:** Increase European allocation (15-20%), favor VGK/EWG/EWQ, European financials/industrials")
            elif ewj_1w > 2 and ewj_ytd > spy_ytd:
                st.success("🇯🇵 **Japan Leading:** Yen weakness supporting exports, corporate reforms\n\n"
                          "→ **Action:** Increase Japan allocation (10-15%), favor EWJ, Japanese exporters")
            else:
                st.info("→ **Action:** Maintain diversified global exposure (US 50%, International 30%, EM 20%), watch for emerging trends")

        st.markdown("---")

        # Reorder columns
        cols = ["Market", "Region", "Ticker", "1 Day", "1 Week", "MTD", "YTD", "1 Year", "3 Years", "5 Years"]
        avail = [c for c in cols if c in equity_perf.columns]
        short_term = [c for c in ["1 Day", "1 Week", "MTD", "YTD"] if c in avail]
        long_term = [c for c in ["1 Year", "3 Years", "5 Years"] if c in avail]

        st.dataframe(
            equity_perf[avail].sort_values("YTD", ascending=False).style
            .format("{:+.2f}%", subset=[c for c in avail if c not in ["Market", "Region", "Ticker"]])
            .background_gradient(cmap="RdYlGn", vmin=-10, vmax=10, subset=short_term)
            .background_gradient(cmap="RdYlGn", vmin=-20, vmax=20, subset=long_term),
            use_container_width=True, height=600, hide_index=True
        )

        st.markdown("---")

        # Regional Performance Comparison
        st.markdown("### 📊 Regional Performance (YTD)")

        if "Region" in equity_perf.columns and "YTD" in equity_perf.columns:
            regional_avg = equity_perf.groupby("Region")["YTD"].mean().sort_values(ascending=False)

            fig_regional = go.Figure(data=[
                go.Bar(
                    x=regional_avg.index,
                    y=regional_avg.values,
                    marker_color=['green' if x > 0 else 'red' for x in regional_avg.values],
                    text=[f"{x:+.2f}%" for x in regional_avg.values],
                    textposition='outside'
                )
            ])

            fig_regional.update_layout(
                title="Average YTD Performance by Region",
                yaxis_title="Return (%)",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_regional, use_container_width=True)

        # Relative Performance Chart (Normalized)
        st.markdown("### 📈 Relative Performance - Selected Markets (Normalized to 100)")

        default_markets = ["SPY", "EEM", "VGK", "EWJ", "INDA", "MCHI"]
        available_defaults = [m for m in default_markets if m in equity_data.columns]

        selected_markets = st.multiselect(
            "Select Markets to Compare",
            options=list(equity_markets.keys()),
            default=available_defaults[:6],
            format_func=lambda x: equity_markets.get(x, x)
        )

        if selected_markets:
            # Normalize to 100
            norm_data = equity_data[selected_markets].copy()
            norm_data = (norm_data / norm_data.iloc[0]) * 100

            fig_norm = go.Figure()
            for col in norm_data.columns:
                fig_norm.add_trace(go.Scatter(
                    x=norm_data.index,
                    y=norm_data[col],
                    name=equity_markets.get(col, col),
                    mode='lines'
                ))

            fig_norm.update_layout(
                title="Relative Performance (Base = 100)",
                yaxis_title="Indexed Value",
                height=500,
                hovermode='x unified'
            )
            st.plotly_chart(fig_norm, use_container_width=True)

    else:
        st.error("⚠️ Unable to load global equity data")

# ==============================================================================
# TAB 4: CURRENCY DASHBOARD
# ==============================================================================
with tab4:
    st.subheader("💱 Global Currency Markets Dashboard")
    st.caption("Track major and emerging market currency pairs")

    st.markdown("### 💵 Major Currency Pairs")

    # Major FX pairs (using both direct pairs and ETF proxies)
    fx_pairs_direct = ['EURUSD=X', 'USDJPY=X', 'GBPUSD=X', 'AUDUSD=X', 'USDCAD=X', 'USDCHF=X']
    fx_names_direct = {
        'EURUSD=X': '🇪🇺 EUR/USD', 'USDJPY=X': '🇯🇵 USD/JPY',
        'GBPUSD=X': '🇬🇧 GBP/USD', 'AUDUSD=X': '🇦🇺 AUD/USD',
        'USDCAD=X': '🇨🇦 USD/CAD', 'USDCHF=X': '🇨🇭 USD/CHF'
    }

    # Currency ETFs as backup/supplement
    fx_etfs = ['UUP', 'FXE', 'FXY', 'FXB', 'FXA', 'FXC', 'FXF']
    fx_etf_names = {
        'UUP': '💵 US Dollar Index', 'FXE': '🇪🇺 Euro', 'FXY': '🇯🇵 Yen',
        'FXB': '🇬🇧 Pound', 'FXA': '🇦🇺 Aussie Dollar',
        'FXC': '🇨🇦 Canadian Dollar', 'FXF': '🇨🇭 Swiss Franc'
    }

    # Try to get FX data
    fx_data = get_fx_data(fx_pairs_direct, days=730)

    # If FX pairs fail, use ETFs
    if fx_data.empty or len(fx_data.columns) < 3:
        st.info("Using Currency ETFs for FX analysis")
        fx_data = get_price_data(fx_etfs, years=2)
        fx_display_names = fx_etf_names
    else:
        fx_display_names = fx_names_direct

    if not fx_data.empty:
        # Performance table
        fx_perf = calculate_perf_table(fx_data)
        fx_perf["Ticker"] = fx_perf.index
        fx_perf["Currency"] = [fx_display_names.get(t, t) for t in fx_perf.index]
        fx_perf = fx_perf.reset_index(drop=True)

        # LIVE FX ANALYSIS & CARRY TRADE SIGNALS - Multi-Timeframe
        st.markdown("### 📊 Live FX Analysis & Carry Trade Signals (Multi-Timeframe)")

        # Helper to get FX performance
        def get_fx_ret(ticker, period):
            if period in fx_perf.columns and ticker in fx_perf["Ticker"].values:
                return fx_perf[fx_perf["Ticker"] == ticker][period].values[0]
            return None

        # Get multi-period performance
        usd_1w, usd_ytd = get_fx_ret("UUP", "1 Week"), get_fx_ret("UUP", "YTD")
        jpy_1w, jpy_ytd = get_fx_ret("FXY", "1 Week"), get_fx_ret("FXY", "YTD")
        eur_1w, eur_ytd = get_fx_ret("FXE", "1 Week"), get_fx_ret("FXE", "YTD")
        aud_1w, aud_ytd = get_fx_ret("FXA", "1 Week"), get_fx_ret("FXA", "YTD")

        # Display multi-timeframe performance
        commentary = []
        if usd_1w is not None and usd_ytd is not None:
            commentary.append(f"**USD (UUP):** 1W {usd_1w:+.1f}% | YTD {usd_ytd:+.1f}%")
        if eur_1w is not None and eur_ytd is not None:
            commentary.append(f"**EUR (FXE):** 1W {eur_1w:+.1f}% | YTD {eur_ytd:+.1f}%")
        if jpy_1w is not None and jpy_ytd is not None:
            commentary.append(f"**JPY (FXY):** 1W {jpy_1w:+.1f}% | YTD {jpy_ytd:+.1f}%")
        if aud_1w is not None and aud_ytd is not None:
            commentary.append(f"**AUD (FXA):** 1W {aud_1w:+.1f}% | YTD {aud_ytd:+.1f}%")
        commentary.append("")

        # Multi-timeframe USD analysis
        if usd_1w is not None and usd_ytd is not None:
            if usd_1w > 0.5 and usd_ytd > 3.0:
                st.warning("💵 **SUSTAINED USD STRENGTH**\n\n" + "\n".join(commentary))
                st.warning("⚠️ **Persistent Dollar Rally:** Strong across 1W and YTD → Multi-month dollar bull trend\n\n"
                          "→ **Action:** Significant headwind for commodities/EM. Favor US domestic stocks, reduce international exposure, short commodity currencies (AUD/CAD). Monitor for Fed pivot.")

            elif usd_1w < -0.5 and usd_ytd < -3.0:
                st.success("💵 **SUSTAINED USD WEAKNESS**\n\n" + "\n".join(commentary))
                st.success("✅ **Persistent Dollar Decline:** Weak across 1W and YTD → Multi-month dollar bear trend\n\n"
                          "→ **Action:** Major tailwind for commodities/EM/gold. Maximize commodity exposure (30%+), long EM/EEM, materials/XLB, gold/GLD. Structural dollar weakness.")

            elif usd_1w > 1.0 and usd_ytd < 0:
                st.info("💵 **USD REVERSAL ATTEMPT**\n\n" + "\n".join(commentary))
                st.info("⚠️ **Recent Dollar Strength in Downtrend:** 1W up but YTD negative → Potential trend reversal or bounce\n\n"
                       "→ **Action:** Monitor closely (60% conviction). Wait for YTD to turn positive before rotating away from commodities/EM.")

            elif usd_1w < -1.0 and usd_ytd > 3.0:
                st.warning("💵 **PULLBACK IN USD UPTREND**\n\n" + "\n".join(commentary))
                st.info("💡 **Short-term weakness in uptrend:** YTD still strong → Likely temporary pullback\n\n"
                       "→ **Action:** Don't chase commodities/EM on this dip. Dollar strength likely resumes.")

            else:
                st.info("💵 **USD RANGE-BOUND**\n\n" + "\n".join(commentary))
                st.info("→ **Action:** Neutral FX positioning (50/50 US/International), wait for clear directional break")

        # Multi-timeframe JPY analysis (risk gauge)
        if jpy_1w is not None and jpy_ytd is not None:
            if jpy_1w > 1.5:
                st.error(f"🇯🇵 **JPY STRENGTH / RISK-OFF** (FXY 1W {jpy_1w:+.1f}%, YTD {jpy_ytd:+.1f}%)\n\n"
                        f"- Yen rallying → Flight to safety, carry trade unwind\n"
                        f"- Crisis signal if sharp + persistent\n"
                        f"- **Action:** Exit carry trades immediately, reduce leverage, hedge portfolios, raise cash (20%+)")

            elif jpy_ytd < -5.0 and jpy_1w < -1.0:
                st.success(f"🇯🇵 **SUSTAINED JPY WEAKNESS** (FXY 1W {jpy_1w:+.1f}%, YTD {jpy_ytd:+.1f}%)\n\n"
                          f"- Persistent Yen weakness across timeframes → Carry trade golden period\n\n"
                          f"- **Action:** Maximize carry trade exposure. Long AUD/JPY, MXN/JPY, BRL/JPY. Monitor VIX < 15 for confirmation.")

        # Carry trade setup with multi-timeframe confirmation
        if aud_1w is not None and aud_ytd is not None and jpy_ytd is not None:
            if aud_1w > 0 and aud_ytd > 5 and jpy_ytd < -3:
                st.success(f"🏄 **PRIME CARRY TRADE SETUP** (AUD 1W {aud_1w:+.1f}%, YTD {aud_ytd:+.1f}% | JPY YTD {jpy_ytd:+.1f}%)\n\n"
                          f"- AUD strong + JPY weak across multiple timeframes → Structural carry opportunity\n"
                          f"- Check VIX < 15 for risk-on confirmation\n\n"
                          f"- **Action:** Long AUD/JPY carry (2-3% position), CAD/JPY, use 3-5% stop-loss, exit immediately if VIX > 20")

        st.markdown("---")

        cols = ["Currency", "Ticker", "1 Day", "1 Week", "MTD", "YTD", "1 Year", "2 Years"]
        avail = [c for c in cols if c in fx_perf.columns]

        st.dataframe(
            fx_perf[avail].style
            .format("{:+.2f}%", subset=[c for c in avail if c not in ["Currency", "Ticker"]])
            .background_gradient(cmap="RdYlGn", vmin=-5, vmax=5,
                               subset=[c for c in avail if c not in ["Currency", "Ticker"]]),
            use_container_width=True, height=350, hide_index=True
        )

        st.markdown("---")

        # FX Charts
        st.markdown("### 📈 Currency Trends")

        selected_fx = st.selectbox(
            "Select Currency to Chart",
            options=list(fx_data.columns),
            format_func=lambda x: fx_display_names.get(x, x)
        )

        if selected_fx:
            fig_fx = go.Figure()

            # Price chart
            fig_fx.add_trace(go.Scatter(
                x=fx_data.index,
                y=fx_data[selected_fx],
                name=fx_display_names.get(selected_fx, selected_fx),
                line=dict(color='cyan', width=2)
            ))

            # Add moving averages
            ma_50 = fx_data[selected_fx].rolling(50).mean()
            ma_200 = fx_data[selected_fx].rolling(200).mean()

            fig_fx.add_trace(go.Scatter(
                x=fx_data.index, y=ma_50,
                name='50-Day MA', line=dict(color='orange', width=1, dash='dash')
            ))
            fig_fx.add_trace(go.Scatter(
                x=fx_data.index, y=ma_200,
                name='200-Day MA', line=dict(color='red', width=1, dash='dash')
            ))

            fig_fx.update_layout(
                title=f"{fx_display_names.get(selected_fx, selected_fx)} - 2 Year Chart",
                yaxis_title="Price / Rate",
                height=450,
                hovermode='x unified'
            )
            st.plotly_chart(fig_fx, use_container_width=True)

            # Current levels vs MA
            current = fx_data[selected_fx].iloc[-1]
            current_ma50 = ma_50.iloc[-1]
            current_ma200 = ma_200.iloc[-1]

            c1, c2, c3 = st.columns(3)
            c1.metric("Current", f"{current:.4f}")
            c2.metric("vs 50-MA", f"{((current/current_ma50 - 1) * 100):+.2f}%")
            c3.metric("vs 200-MA", f"{((current/current_ma200 - 1) * 100):+.2f}%")

            # Trend signal
            if current > current_ma50 and current_ma50 > current_ma200:
                st.success("🟢 **Strong Uptrend:** Price above both MAs, bullish momentum")
            elif current < current_ma50 and current_ma50 < current_ma200:
                st.error("🔴 **Strong Downtrend:** Price below both MAs, bearish momentum")
            else:
                st.warning("🟡 **Mixed Trend:** Consolidation or transition phase")

    else:
        st.error("⚠️ Unable to load FX data")


# ==============================================================================
# TAB 5: COMMODITIES DASHBOARD
# ==============================================================================
with tab5:
    st.subheader("📦 Commodities & Raw Materials Dashboard")
    st.caption("Track precious metals, energy, agriculture, and industrial commodities")

    # Comprehensive commodities coverage
    commodity_map = {
        # Precious Metals
        "GLD": "🥇 Gold",
        "SLV": "🥈 Silver",
        "PPLT": "⚪ Platinum",
        "PALL": "🔘 Palladium",
        # Energy
        "USO": "🛢️ Crude Oil",
        "UNG": "🔥 Natural Gas",
        "UGA": "⛽ Gasoline",
        "BNO": "🌊 Brent Oil",
        # Agriculture
        "DBA": "🌾 Agriculture (Broad)",
        "CORN": "🌽 Corn",
        "WEAT": "🌾 Wheat",
        "SOYB": "🫘 Soybeans",
        "CANE": "🍬 Sugar",
        "JO": "☕ Coffee",
        # Industrial
        "CPER": "🔶 Copper",
        "URA": "☢️ Uranium",
        "WOOD": "🪵 Lumber"
    }

    commodity_categories = {
        "GLD": "Precious Metals", "SLV": "Precious Metals", "PPLT": "Precious Metals", "PALL": "Precious Metals",
        "USO": "Energy", "UNG": "Energy", "UGA": "Energy", "BNO": "Energy",
        "DBA": "Agriculture", "CORN": "Agriculture", "WEAT": "Agriculture",
        "SOYB": "Agriculture", "CANE": "Agriculture", "JO": "Agriculture",
        "CPER": "Industrial", "URA": "Industrial", "WOOD": "Industrial"
    }

    # Fetch commodity data
    cmdty_data = get_price_data(list(commodity_map.keys()), years=5)

    if not cmdty_data.empty:
        # Performance table
        cmdty_perf = calculate_perf_table(cmdty_data)
        cmdty_perf["Ticker"] = cmdty_perf.index
        cmdty_perf["Commodity"] = [commodity_map.get(t, t) for t in cmdty_perf.index]
        cmdty_perf["Category"] = [commodity_categories.get(t, "Other") for t in cmdty_perf.index]
        cmdty_perf = cmdty_perf.reset_index(drop=True)

        # LIVE COMMODITY MARKET ANALYSIS - Multi-Timeframe
        st.markdown("### 📊 Live Commodity Market Signals & Trading Setups (Multi-Timeframe)")

        # Helper to get commodity performance
        def get_cmd_ret(ticker, period):
            if period in cmdty_perf.columns and ticker in cmdty_perf["Ticker"].values:
                return cmdty_perf[cmdty_perf["Ticker"] == ticker][period].values[0]
            return None

        # Get multi-period performance
        gold_1w, gold_ytd, gold_1y = get_cmd_ret("GLD", "1 Week"), get_cmd_ret("GLD", "YTD"), get_cmd_ret("GLD", "1 Year")
        oil_1w, oil_ytd, oil_1y = get_cmd_ret("USO", "1 Week"), get_cmd_ret("USO", "YTD"), get_cmd_ret("USO", "1 Year")
        copper_1w, copper_ytd = get_cmd_ret("CPER", "1 Week"), get_cmd_ret("CPER", "YTD")
        silver_1w, silver_ytd = get_cmd_ret("SLV", "1 Week"), get_cmd_ret("SLV", "YTD")

        # Display multi-timeframe performance
        commentary = []
        if gold_1w is not None and gold_ytd is not None:
            commentary.append(f"**Gold (GLD):** 1W {gold_1w:+.1f}% | YTD {gold_ytd:+.1f}% | 1Y {gold_1y:+.1f}%" if gold_1y else f"**Gold (GLD):** 1W {gold_1w:+.1f}% | YTD {gold_ytd:+.1f}%")
        if oil_1w is not None and oil_ytd is not None:
            commentary.append(f"**Oil (USO):** 1W {oil_1w:+.1f}% | YTD {oil_ytd:+.1f}% | 1Y {oil_1y:+.1f}%" if oil_1y else f"**Oil (USO):** 1W {oil_1w:+.1f}% | YTD {oil_ytd:+.1f}%")
        if copper_1w is not None and copper_ytd is not None:
            commentary.append(f"**Copper (CPER):** 1W {copper_1w:+.1f}% | YTD {copper_ytd:+.1f}%")
        commentary.append("")

        # Multi-timeframe Gold analysis
        if gold_1w is not None and gold_ytd is not None and gold_1y is not None:
            if gold_1w > 1.0 and gold_ytd > 8.0 and gold_1y > 15.0:
                st.success("🥇 **GOLD BULL MARKET**\n\n" + "\n".join(commentary))
                st.success("✅ **Sustained Gold Rally:** Strong across all timeframes → Multi-year bull market\n\n"
                          "→ **Action:** Maximum gold allocation (15-20%), long GLD/miners/GDX, structural safe-haven/inflation hedge. Secular trend.")

            elif gold_1w > 1.5 and gold_ytd < 0:
                st.warning("🥇 **GOLD REVERSAL ATTEMPT**\n\n" + "\n".join(commentary))
                st.info("⚠️ **Recent strength in downtrend:** 1W up but YTD negative → Potential trend change or bounce\n\n"
                       "→ **Action:** Moderate gold allocation (8-10%), wait for YTD to turn positive for confirmation. Monitor yields/USD.")

            elif gold_1w > 1.5:
                st.warning("🥇 **GOLD RALLYING**\n\n" + "\n".join(commentary))
                st.info("💡 **Check yield environment:** Yields falling = safety bid | Yields rising = inflation fears\n\n"
                       "→ **Action:** Increase gold (10-15%), long GLD, add TIPS/TIP for inflation hedge, reduce equity beta")

        # Multi-timeframe Oil analysis
        if oil_1w is not None and oil_ytd is not None and oil_1y is not None:
            if oil_1w > 2.0 and oil_ytd > 15.0:
                st.success("🛢️ **OIL BULL MARKET**\n\n" + "\n".join(commentary))
                st.success("✅ **Sustained Oil Rally:** Strong YTD → Inflation building, supply tightening\n\n"
                          "→ **Action:** Maximum energy allocation (15-20%), long XLE/USO, TIPS/TIP for inflation, monitor Fed reaction")

            elif oil_1w < -2.0 and oil_ytd < -15.0:
                st.error("🛢️ **OIL BEAR MARKET**\n\n" + "\n".join(commentary))
                st.error("🚨 **Sustained Oil Weakness:** Weak across timeframes → Demand destruction, recession fears\n\n"
                        "→ **Action:** Avoid energy (0-5%), disinflationary pressure, monitor for economic slowdown signals")

            elif oil_1w > 3.0 and oil_ytd < 0:
                st.warning("🛢️ **OIL REVERSAL ATTEMPT**\n\n" + "\n".join(commentary))
                st.info("⚠️ **Sharp bounce in downtrend:** Supply shock or short-covering?\n\n"
                       "→ **Action:** Tactical energy trade (5-8%), tight stops, wait for YTD confirmation")

        # Multi-timeframe Copper analysis (growth indicator)
        if copper_1w is not None and copper_ytd is not None:
            if copper_1w > 1.5 and copper_ytd > 10.0:
                st.success(f"🔶 **COPPER CONFIRMS GROWTH** (1W {copper_1w:+.1f}%, YTD {copper_ytd:+.1f}%)\n\n"
                          f"- Industrial metals strong across timeframes → Economic expansion confirmed\n\n"
                          f"→ **Action:** Pro-growth portfolio (90% equities), long cyclicals/XLI, materials/XLB, EM/EEM, infrastructure plays")

        # Commodity Super-Cycle Detection with YTD confirmation
        if "1 Week" in cmdty_perf.columns and "YTD" in cmdty_perf.columns and "Category" in cmdty_perf.columns:
            cat_1w = cmdty_perf.groupby("Category")["1 Week"].mean()
            cat_ytd = cmdty_perf.groupby("Category")["YTD"].mean()

            pm_1w = cat_1w.get("Precious Metals", 0)
            pm_ytd = cat_ytd.get("Precious Metals", 0)
            energy_1w = cat_1w.get("Energy", 0)
            energy_ytd = cat_ytd.get("Energy", 0)
            industrial_1w = cat_1w.get("Industrial", 0)
            industrial_ytd = cat_ytd.get("Industrial", 0)

            if pm_ytd > 8 and energy_ytd > 10 and industrial_ytd > 8:
                st.success(f"🌟 **COMMODITY SUPER-CYCLE DETECTED**\n\n"
                          f"**YTD Performance:** Precious {pm_ytd:+.1f}% | Energy {energy_ytd:+.1f}% | Industrial {industrial_ytd:+.1f}%\n\n"
                          f"✅ **All categories strong YTD** → Multi-year commodity bull market\n\n"
                          f"→ **Action:** MAJOR commodity allocation (30-40%), long XLE/XLB/GLD, DBC commodity index, reduce bonds, overweight materials/energy sectors. Structural multi-year trend.")

        st.markdown("---")

        cols = ["Commodity", "Category", "Ticker", "1 Day", "1 Week", "MTD", "YTD", "1 Year", "3 Years", "5 Years"]
        avail = [c for c in cols if c in cmdty_perf.columns]
        short_term = [c for c in ["1 Day", "1 Week", "MTD", "YTD"] if c in avail]
        long_term = [c for c in ["1 Year", "3 Years", "5 Years"] if c in avail]

        st.dataframe(
            cmdty_perf[avail].sort_values("YTD", ascending=False).style
            .format("{:+.2f}%", subset=[c for c in avail if c not in ["Commodity", "Category", "Ticker"]])
            .background_gradient(cmap="RdYlGn", vmin=-15, vmax=15, subset=short_term)
            .background_gradient(cmap="RdYlGn", vmin=-25, vmax=25, subset=long_term),
            use_container_width=True, height=500, hide_index=True
        )

        st.markdown("---")

        # Category Performance
        st.markdown("### 📊 Commodity Category Performance (YTD)")

        if "Category" in cmdty_perf.columns and "YTD" in cmdty_perf.columns:
            cat_avg = cmdty_perf.groupby("Category")["YTD"].mean().sort_values(ascending=False)

            fig_cat = go.Figure(data=[
                go.Bar(
                    x=cat_avg.index,
                    y=cat_avg.values,
                    marker_color=['green' if x > 0 else 'red' for x in cat_avg.values],
                    text=[f"{x:+.2f}%" for x in cat_avg.values],
                    textposition='outside'
                )
            ])

            fig_cat.update_layout(
                title="Average YTD Return by Commodity Category",
                yaxis_title="Return (%)",
                height=350,
                showlegend=False
            )
            st.plotly_chart(fig_cat, use_container_width=True)

        # Commodity Comparison Chart
        st.markdown("### 📈 Commodity Price Comparison (Normalized to 100)")

        default_cmdty = ["GLD", "USO", "DBA", "CPER"]
        available_cmdty = [c for c in default_cmdty if c in cmdty_data.columns]

        selected_cmdty = st.multiselect(
            "Select Commodities to Compare",
            options=list(commodity_map.keys()),
            default=available_cmdty,
            format_func=lambda x: commodity_map.get(x, x)
        )

        if selected_cmdty:
            norm_cmdty = cmdty_data[selected_cmdty].copy()
            norm_cmdty = (norm_cmdty / norm_cmdty.iloc[0]) * 100

            fig_cmdty = go.Figure()
            for col in norm_cmdty.columns:
                fig_cmdty.add_trace(go.Scatter(
                    x=norm_cmdty.index,
                    y=norm_cmdty[col],
                    name=commodity_map.get(col, col),
                    mode='lines'
                ))

            fig_cmdty.update_layout(
                title="Relative Commodity Performance (Base = 100)",
                yaxis_title="Indexed Value",
                height=500,
                hovermode='x unified'
            )
            st.plotly_chart(fig_cmdty, use_container_width=True)

    else:
        st.error("⚠️ Unable to load commodity data")


# ==============================================================================
# TAB 6: CROSS-ASSET CORRELATIONS
# ==============================================================================
with tab6:
    st.subheader("🔗 Cross-Asset Correlation Dashboard")
    st.caption("Analyze correlations across equities, bonds, FX, commodities, and crypto")

    # Trading Insights at the top
    st.markdown("### 💡 Correlation-Based Portfolio & Risk Management")
    st.info("""
    **Diversification Strategy:** Target correlations < 0.5 for effective risk reduction | Negative correlations = best hedges

    **Classic Hedges:** SPY vs TLT (negative in normal times) | SPY vs Gold (low/negative) | USD vs Commodities (negative)

    **Regime Detection - Normal Market:** Stocks/Bonds negatively correlated, VIX low → Traditional 60/40 portfolio works

    **Regime Detection - Crisis:** ALL correlations → +1 (everything falls together) → Liquidity crisis signal, raise cash immediately

    **Inflation Regime:** Stocks/Bonds positively correlated (both down), Commodities/TIPS up → Need real assets for protection

    **Risk Monitoring:** Rising SPY/EM correlation = contagion risk | SPY/HYG correlation > 0.8 = credit risk matters

    **Crypto Behavior:** BTC correlation with SPY increasing (was 0, now 0.5+) → Crypto = risk asset, not safe-haven

    **Portfolio Construction:** Build with assets having correlation < 0.3 to each other → Maximum diversification benefit
    """)
    st.markdown("---")

    # Comprehensive asset selection for correlation
    correlation_assets = {
        # Equities
        "SPY": "US S&P 500", "QQQ": "Nasdaq", "EEM": "EM Equities", "VGK": "Europe Equities",
        # Fixed Income
        "TLT": "20Y Treasury", "HYG": "High Yield", "LQD": "Corp Bonds", "TIP": "TIPS",
        # Commodities
        "GLD": "Gold", "USO": "Oil", "DBA": "Agriculture", "CPER": "Copper",
        # FX
        "UUP": "USD Index", "FXE": "Euro", "FXY": "Yen",
        # Crypto
        "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum"
    }

    st.markdown("### 📊 Rolling Correlation Matrix")

    # Fetch data for correlation analysis
    corr_data = get_price_data(list(correlation_assets.keys()), years=3)

    if not corr_data.empty:
        # Correlation window selector
        corr_window = st.slider("Correlation Window (days)", 30, 252, 90, key="corr_window_cross")

        # Calculate correlations
        returns = corr_data.pct_change().tail(corr_window)
        corr_matrix = returns.corr()

        # Rename for display
        display_names = [correlation_assets.get(t, t) for t in corr_matrix.index]
        corr_matrix.index = display_names
        corr_matrix.columns = display_names

        # Heatmap
        fig_corr = px.imshow(
            corr_matrix,
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            title=f"{corr_window}-Day Rolling Correlation Matrix",
            aspect='auto',
            labels=dict(color="Correlation")
        )
        fig_corr.update_layout(height=700)
        st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown("---")

        # Correlation time series
        st.markdown("### 📈 Correlation Evolution Over Time")

        col1, col2 = st.columns(2)
        with col1:
            asset1 = st.selectbox(
                "Select First Asset",
                options=list(correlation_assets.keys()),
                format_func=lambda x: correlation_assets[x],
                index=0
            )
        with col2:
            asset2 = st.selectbox(
                "Select Second Asset",
                options=list(correlation_assets.keys()),
                format_func=lambda x: correlation_assets[x],
                index=1
            )

        if asset1 and asset2 and asset1 in corr_data.columns and asset2 in corr_data.columns:
            # Calculate rolling correlation
            rolling_corr = corr_data[asset1].pct_change().rolling(60).corr(corr_data[asset2].pct_change())

            fig_rolling_corr = go.Figure()
            fig_rolling_corr.add_trace(go.Scatter(
                x=rolling_corr.index,
                y=rolling_corr,
                fill='tozeroy',
                name='60-Day Rolling Correlation',
                line=dict(color='purple', width=2)
            ))

            fig_rolling_corr.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_rolling_corr.add_hline(y=0.5, line_dash="dot", line_color="green", annotation_text="Strong +")
            fig_rolling_corr.add_hline(y=-0.5, line_dash="dot", line_color="red", annotation_text="Strong -")

            fig_rolling_corr.update_layout(
                title=f"Rolling Correlation: {correlation_assets[asset1]} vs {correlation_assets[asset2]}",
                yaxis_title="Correlation Coefficient",
                yaxis_range=[-1, 1],
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig_rolling_corr, use_container_width=True)

            # Current correlation
            current_corr = rolling_corr.iloc[-1]
            c1, c2, c3 = st.columns(3)
            c1.metric("Current Correlation", f"{current_corr:.3f}")

            if current_corr > 0.5:
                c2.metric("Relationship", "🟢 Strong Positive")
                c3.info("Assets move together - limited diversification benefit")
            elif current_corr < -0.5:
                c2.metric("Relationship", "🔴 Strong Negative")
                c3.success("Assets move opposite - excellent diversification")
            elif abs(current_corr) < 0.3:
                c2.metric("Relationship", "⚪ Uncorrelated")
                c3.info("Assets move independently - good diversification")
            else:
                c2.metric("Relationship", "🟡 Moderate")
                c3.warning("Moderate correlation - some diversification benefit")

    else:
        st.error("⚠️ Unable to load correlation data")


# ==============================================================================
# TAB 11: RISK ANALYTICS
# ==============================================================================
with tab11:
    st.subheader("⚠️ Comprehensive Risk Analytics Dashboard")

    # LIVE RISK ASSESSMENT - Analyze default assets for immediate insights
    st.markdown("### 📊 Live Risk Assessment")

    # Quick analysis on default assets to provide immediate commentary
    quick_risk_assets = ['SPY', 'QQQ', 'TLT', 'GLD', 'BTC-USD']
    quick_risk_data = get_price_data(quick_risk_assets, years=2)

    if not quick_risk_data.empty:
        quick_returns = quick_risk_data.pct_change().dropna()

        # Calculate market breadth for quick assessment
        breadth_quick = calculate_market_breadth(quick_risk_data)

        commentary = []

        if not breadth_quick.empty:
            current_breadth = breadth_quick['Breadth'].iloc[-1]
            prev_breadth = breadth_quick['Breadth'].iloc[-22] if len(breadth_quick) > 22 else current_breadth
            breadth_trend = current_breadth - prev_breadth

            # Market Breadth Analysis
            commentary.append(f"**Market Breadth:** {current_breadth:.1f}% of major assets above 50-day MA")

            if current_breadth >= 70:
                st.success(f"🟢 **BROAD MARKET STRENGTH** ({current_breadth:.1f}% breadth)\n\n"
                          f"- Most assets in uptrends → Healthy market environment\n"
                          f"- Breadth trend: {breadth_trend:+.1f}% vs 1 month ago\n"
                          f"- **Action:** Can increase portfolio exposure to 100%+, favor growth/momentum strategies")

            elif current_breadth >= 50:
                st.info(f"🟡 **MIXED MARKET CONDITIONS** ({current_breadth:.1f}% breadth)\n\n"
                       f"- Moderate breadth → Selective opportunities\n"
                       f"- Breadth trend: {breadth_trend:+.1f}% vs 1 month ago\n"
                       f"- **Action:** Maintain 70-85% exposure, be selective with new positions")

            elif current_breadth >= 30:
                st.warning(f"🟠 **MARKET WEAKNESS DETECTED** ({current_breadth:.1f}% breadth)\n\n"
                          f"- Limited breadth → Many assets in downtrends\n"
                          f"- Breadth trend: {breadth_trend:+.1f}% vs 1 month ago\n"
                          f"- **Action:** Reduce exposure to 50-60%, defensive positioning, raise cash")

            else:
                st.error(f"🔴 **BROAD MARKET WEAKNESS** ({current_breadth:.1f}% breadth)\n\n"
                        f"- Severe weakness → Most assets below 50-MA\n"
                        f"- Breadth trend: {breadth_trend:+.1f}% vs 1 month ago\n"
                        f"- **Action:** Reduce exposure to 25-40% or less, hedge with puts, increase cash allocation")

        # Calculate quick risk metrics for major indices
        if 'SPY' in quick_returns.columns:
            spy_metrics = calculate_risk_metrics(quick_returns['SPY'])
            spy_sharpe = spy_metrics.get('Sharpe Ratio', 0)
            spy_dd = spy_metrics.get('Max Drawdown', 0)

            risk_msg = []
            risk_msg.append(f"**SPY Risk Profile:** Sharpe {spy_sharpe:.2f}, Max DD {spy_dd:.1%}")

            if spy_sharpe > 1.0 and spy_dd > -0.15:
                risk_msg.append("→ Strong risk-adjusted performance with controlled drawdowns")
            elif spy_sharpe < 0.5:
                risk_msg.append("→ Poor risk-adjusted returns - consider reducing equity exposure")

            if spy_dd < -0.25:
                risk_msg.append(f"→ Deep drawdown ({spy_dd:.1%}) indicates high volatility environment")

            st.markdown("\n".join(risk_msg))

    st.markdown("---")

    # Select assets for risk analysis
    st.markdown("### 🎯 Select Assets for Risk Analysis")
    default_risk_assets = ['SPY', 'QQQ', 'TLT', 'GLD', 'BTC-USD']
    risk_assets_input = st.text_input(
        "Enter tickers (comma-separated)",
        value=', '.join(default_risk_assets),
        help="Example: SPY, QQQ, TLT, GLD"
    )
    risk_assets = [t.strip().upper() for t in risk_assets_input.split(',')]

    if st.button("📊 Analyze Risk Profile"):
        risk_data = get_price_data(risk_assets, years=5)

        if not risk_data.empty:
            returns = risk_data.pct_change().dropna()

            # Calculate risk metrics for each asset
            st.markdown("### 📋 Risk Metrics Comparison")
            risk_comparison = {}

            for ticker in risk_data.columns:
                if ticker in returns.columns:
                    metrics = calculate_risk_metrics(returns[ticker])
                    risk_comparison[ticker] = metrics

            risk_df = pd.DataFrame(risk_comparison).T

            # Display formatted table
            formatted_risk = risk_df.style.format({
                'Annual Return': '{:.2%}',
                'Annual Volatility': '{:.2%}',
                'Sharpe Ratio': '{:.3f}',
                'Sortino Ratio': '{:.3f}',
                'Max Drawdown': '{:.2%}',
                'Calmar Ratio': '{:.3f}',
                'VaR (95%)': '{:.2%}',
                'VaR (99%)': '{:.2%}',
                'CVaR (95%)': '{:.2%}',
                'Skewness': '{:.3f}',
                'Kurtosis': '{:.3f}',
                'Win Rate': '{:.2%}'
            }).background_gradient(cmap='RdYlGn', subset=['Sharpe Ratio', 'Sortino Ratio'])

            st.dataframe(formatted_risk, use_container_width=True, height=400)

            st.markdown("---")

            # CORRELATION HEATMAP
            st.markdown("### 🔥 Correlation Heatmap (Full History)")
            corr_matrix = returns.corr()

            fig_corr_risk = px.imshow(
                corr_matrix,
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1,
                title="Asset Correlation Matrix",
                aspect='auto',
                text_auto='.2f'
            )
            fig_corr_risk.update_layout(height=500)
            st.plotly_chart(fig_corr_risk, use_container_width=True)

            st.markdown("---")

            # ROLLING CORRELATION
            st.markdown("### 📈 Rolling Correlation Analysis")
            if len(risk_assets) >= 2:
                ticker1 = st.selectbox("Asset 1", risk_assets, index=0, key='corr1')
                ticker2 = st.selectbox("Asset 2", risk_assets, index=min(1, len(risk_assets)-1), key='corr2')

                if ticker1 in returns.columns and ticker2 in returns.columns:
                    rolling_corr = returns[ticker1].rolling(60).corr(returns[ticker2])

                    fig_rolling_corr = go.Figure()
                    fig_rolling_corr.add_trace(go.Scatter(
                        x=rolling_corr.index,
                        y=rolling_corr,
                        name=f'{ticker1} vs {ticker2}',
                        line=dict(color='blue')
                    ))
                    fig_rolling_corr.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig_rolling_corr.update_layout(
                        title=f"60-Day Rolling Correlation: {ticker1} vs {ticker2}",
                        yaxis_title="Correlation",
                        height=350
                    )
                    st.plotly_chart(fig_rolling_corr, use_container_width=True)

            st.markdown("---")

            # DRAWDOWN COMPARISON
            st.markdown("### 🌊 Maximum Drawdown Comparison")

            fig_dd_compare = go.Figure()

            for ticker in risk_data.columns:
                if ticker in returns.columns:
                    cum_rets = (1 + returns[ticker]).cumprod()
                    running_max = cum_rets.expanding().max()
                    dd = (cum_rets - running_max) / running_max * 100

                    fig_dd_compare.add_trace(go.Scatter(
                        x=dd.index,
                        y=dd,
                        name=ticker,
                        mode='lines'
                    ))

            fig_dd_compare.update_layout(
                title="Drawdown Comparison (%)",
                yaxis_title="Drawdown (%)",
                height=450,
                hovermode='x unified'
            )
            fig_dd_compare.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_dd_compare, use_container_width=True)

            st.markdown("---")

            # RISK-RETURN SCATTER
            st.markdown("### 🎯 Risk-Return Profile")
            scatter_data = []
            for ticker in risk_data.columns:
                if ticker in returns.columns:
                    annual_ret = returns[ticker].mean() * 252 * 100
                    annual_vol = returns[ticker].std() * np.sqrt(252) * 100
                    sharpe = risk_df.loc[ticker, 'Sharpe Ratio']
                    scatter_data.append({
                        'Ticker': ticker,
                        'Return': annual_ret,
                        'Volatility': annual_vol,
                        'Sharpe': sharpe
                    })

            scatter_df = pd.DataFrame(scatter_data)

            # Create size column that's always positive (Sharpe + offset)
            # This ensures all values are > 0 for plotly's size parameter
            min_sharpe = scatter_df['Sharpe'].min()
            sharpe_offset = abs(min_sharpe) + 0.5 if min_sharpe < 0 else 0
            scatter_df['Size'] = scatter_df['Sharpe'] + sharpe_offset

            fig_scatter = px.scatter(
                scatter_df,
                x='Volatility',
                y='Return',
                size='Size',
                color='Sharpe',
                text='Ticker',
                color_continuous_scale='RdYlGn',
                title='Annualized Risk-Return Profile',
                labels={'Return': 'Annual Return (%)', 'Volatility': 'Annual Volatility (%)'},
                hover_data={'Sharpe': ':.3f', 'Size': False}
            )
            fig_scatter.update_traces(textposition='top center', marker=dict(sizemode='area', sizeref=0.1))
            fig_scatter.update_layout(height=500)
            st.plotly_chart(fig_scatter, use_container_width=True)

            st.markdown("---")

            # MARKET BREADTH ANALYSIS
            st.markdown("### 📊 Market Breadth Indicators")
            st.caption("Percentage of assets trading above their 50-day moving average")

            # Calculate breadth for the selected assets
            breadth_data = calculate_market_breadth(risk_data)

            if not breadth_data.empty:
                fig_breadth = go.Figure()

                fig_breadth.add_trace(go.Scatter(
                    x=breadth_data.index,
                    y=breadth_data['Breadth'],
                    fill='tozeroy',
                    name='Breadth %',
                    line=dict(color='cyan', width=2)
                ))

                # Add reference lines
                fig_breadth.add_hline(y=50, line_dash="dash", line_color="gray",
                                     annotation_text="Neutral (50%)")
                fig_breadth.add_hline(y=70, line_dash="dot", line_color="green",
                                     annotation_text="Strong (70%)")
                fig_breadth.add_hline(y=30, line_dash="dot", line_color="red",
                                     annotation_text="Weak (30%)")

                fig_breadth.update_layout(
                    title="Market Breadth: % Assets Above 50-Day MA",
                    yaxis_title="Breadth (%)",
                    yaxis_range=[0, 100],
                    height=400,
                    hovermode='x unified'
                )

                st.plotly_chart(fig_breadth, use_container_width=True)

                # Current breadth metric
                current_breadth = breadth_data['Breadth'].iloc[-1]
                prev_breadth = breadth_data['Breadth'].iloc[-22] if len(breadth_data) > 22 else current_breadth

                col1, col2, col3 = st.columns(3)
                col1.metric("Current Breadth", f"{current_breadth:.1f}%")
                col2.metric("1M Change", f"{current_breadth - prev_breadth:+.1f}%")

                # Interpretation
                if current_breadth >= 70:
                    col3.metric("Signal", "🟢 Strong", help="Broad market strength")
                    st.success("✅ **Broad Market Strength:** Most assets in uptrends - healthy market environment")
                elif current_breadth >= 50:
                    col3.metric("Signal", "🟡 Neutral", help="Mixed market")
                    st.info("⚖️ **Mixed Market:** Moderate breadth - selective opportunities")
                elif current_breadth >= 30:
                    col3.metric("Signal", "🟠 Weak", help="Market weakness")
                    st.warning("⚠️ **Market Weakness:** Limited breadth - defensive positioning warranted")
                else:
                    col3.metric("Signal", "🔴 Very Weak", help="Broad market weakness")
                    st.error("🚨 **Broad Market Weakness:** Most assets in downtrends - high risk environment")

        else:
            st.error("Could not fetch data for risk analysis.")

# ==============================================================================
# SIDEBAR: DATA QUALITY & EXPORT
# ==============================================================================
with st.sidebar:
    st.markdown("---")
    st.markdown("### 💾 Export Tools")

    # Export functionality
    if st.button("📥 Export Global Data"):
        df_export = get_price_data(list(asset_map.keys()), years=5)
        if not df_export.empty:
            csv = df_export.to_csv()
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"global_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No data available to export")

    # Show data quality for main global view
    df_quality = get_price_data(list(asset_map.keys()), years=5)
    if not df_quality.empty:
        show_data_quality(df_quality, location="sidebar")

st.markdown("---")
st.caption("🦅 Professional Macro-Quant Workstation | Powered by TIINGO & OpenBB | Built with Streamlit")
