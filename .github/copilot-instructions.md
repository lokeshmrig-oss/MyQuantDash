# Copilot Instructions for MyQuantDash

## Project Overview
**MyQuantDash** is a Streamlit-based quantitative analysis dashboard for multi-asset portfolio analysis, backtesting, and sentiment analysis. The entire application is contained in a single file: `dashboard.py`.

### Key Technologies
- **Streamlit**: Web UI framework (set to wide layout with `st.set_page_config(layout="wide")`)
- **OpenBB**: Data provider with fallback logic (Tiingo → Yahoo Finance)
- **Plotly**: Interactive charting via `plotly.express` and `plotly.graph_objects`
- **Pandas**: Data manipulation (including `pandas_ta` for technical indicators)
- **yfinance**: Stock fundamentals (via `yf.Ticker()`)
- **Transformers (FinBERT)**: Sentiment analysis on financial news
- **PRAW**: Reddit API for WSB sentiment scanning

## Architecture & Data Flow

### Single File Design
All functionality is in `dashboard.py`. Additions should extend existing patterns rather than splitting into separate modules (unless codebase grows significantly).

### API Provider Strategy
Data fetching uses **cascading fallbacks**:
1. **Tiingo** (premium provider if `st.secrets["tiingo"]` exists)
2. **Yahoo Finance** (free fallback via `yfinance` or OpenBB)

See `get_price_data()` function (~lines 50-85) for the pattern. Always wrap API calls in try-except, log errors to debug logs, and allow graceful degradation. Use `@st.cache_data(ttl=3600)` to cache price data for 1 hour.

### Tab Architecture (5 Tabs)
Each tab is a self-contained feature with clear boundaries:
1. **Global View** (Tab 1): Multi-asset performance matrix with CAGR/YTD analysis
2. **Portfolio Sim** (Tab 2): User-defined portfolio backtesting with drag-&-drop weights
3. **Sectors & Factors** (Tab 3): Sector/factor performance with risk-reward scatter plot
4. **Stock Fundamentals** (Tab 4): Individual stock deep-dive with technicals and ratios
5. **AI Sentiment** (Tab 5): Reddit sentiment (WSB) + FinBERT news sentiment

## Critical Patterns & Conventions

### Performance Table Calculation
Function `calculate_perf_table()` (~lines 22-46) computes period returns (1D, MTD, YTD, 1Y, 3Y, 5Y) with **CAGR annualization** for multi-year periods. Key behaviors:
- Uses `pd.DateOffset()` for date arithmetic
- Handles missing dates with `get_indexer(..., method='pad')`
- Returns DataFrame indexed by asset name with period columns

**When adding new performance metrics:** Follow this pattern for consistency.

### Debug Mode Pattern
`debug_mode` checkbox (lines 15-19) controls conditional logging:
```python
if debug_mode: st.write(f"Trying Tiingo...")  # for process flow
if debug_mode: st.error(f"DATA FAILURE...")   # for errors
```
Always use this for non-essential diagnostics to avoid UI clutter.

### Streamlit Caching Strategy
- **`@st.cache_data(ttl=3600)`**: For API data fetches (price data, news, etc.)
- **`@st.cache_resource`**: For expensive initialization (FinBERT model, Reddit client)

These prevent re-running on widget changes. Update TTL based on data freshness needs.

### Performance Visualization
Use Plotly with consistent styling:
- **Background gradients**: `style.background_gradient(cmap="RdYlGn", vmin=..., vmax=...)`
- **Multi-trace charts**: `go.Figure()` with `.add_trace()` for layered comparisons (see Portfolio Sim)
- **Scatter plots**: `px.scatter()` with color mapping for grouping (Sectors tab uses `{"Sector": "#1f77b4", "Factor": "#ff7f0e"}`)

### Data Transformation Convention
Before displaying DataFrames:
1. Calculate metrics (e.g., `calculate_perf_table()`)
2. Add human-readable names/groupings (e.g., `asset_map`, `sec_map`)
3. Reset/reindex for display
4. Filter available columns (`[c for c in cols if c in df.columns]`)
5. Apply styling and formatting

See **Global View** tab (lines 110-150) for full example.

### Conditional Data Availability
Many API calls may fail (invalid ticker, rate limits, data gaps). Always:
1. Check `if not df.empty:` before visualization
2. Display relevant error message with `st.error()` or `st.warning()`
3. Never leave blank spaces—users should see clear feedback

## Secrets Management
`.streamlit/secrets.toml` contains:
- `tiingo`: Dict with API key for OpenBB Tiingo provider
- `reddit`: Dict with `client_id`, `client_secret`, `user_agent` for PRAW

Access via `st.secrets["key"]`. Always check existence before using to allow graceful fallbacks.

## Common Workflows & Debugging

### Testing Data Fetch
Enable "Show Debug Logs" in sidebar. API failures will display error messages. Check internet, API rate limits, and ticker validity.

### Adding a New Asset Class
1. Add ticker to an asset mapping dict (e.g., `asset_map`, `sec_map`)
2. Call `get_price_data()` with the ticker list
3. Compute returns with `calculate_perf_table()`
4. Visualize with Plotly (follow existing pattern for consistency)

### Modifying Performance Metrics
Update `calculate_perf_table()` periods dict to add/remove timeframes. Ensure all downstream styling logic (short_term/long_term lists) is updated.

## Key File References
- **`dashboard.py`**: Entire application (388 lines)
- **`.streamlit/secrets.toml`**: API credentials and configuration
- **`.ruff_cache/`**: Linting cache (ignore in edits)

## Quality Guidelines
- **Keep functions modular**: Each tab and helper function should be ~30-80 lines
- **Error handling**: Always wrap external API calls in try-except; provide debug feedback
- **Naming**: Use descriptive variable names for asset mappings (e.g., `asset_map`, `sec_map`, `full_map`)
- **Performance**: Use Streamlit's caching aggressively to avoid re-computing data
- **UI/UX**: Maintain consistent color schemes and layout patterns across tabs
