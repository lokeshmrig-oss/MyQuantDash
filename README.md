# 🦅 Professional Macro-Quant Workstation

A comprehensive multi-asset dashboard for institutional-grade macro and quantitative analysis.

## 🎯 Features

- **12 Professional Dashboards** covering global markets, macro indicators, sectors, factors, and risk analytics
- **Live Multi-Timeframe Analysis** with actionable trading signals across 1W, YTD, and 1Y
- **150+ Instruments Tracked** including equities, bonds, FX, commodities, and crypto
- **Real-Time Data** from TIINGO, OpenBB, Yahoo Finance, and FRED APIs
- **AI-Powered Sentiment Analysis** using FinBERT NLP models
- **Portfolio Analytics** with Monte Carlo simulation and efficient frontier optimization

## 📊 Dashboard Tabs

1. **🌍 Global View** - Quick snapshot of 14 major market benchmarks
2. **📈 Macro Dashboard** - Economic indicators, yield curve, VIX, credit spreads
3. **🌎 Global Equity Markets** - 30+ countries with regional rotation analysis
4. **💱 Currency Dashboard** - FX trends and carry trade setups
5. **📦 Commodities Dashboard** - 17+ commodities across all categories
6. **🔗 Cross-Asset Correlations** - Portfolio diversification analysis
7. **📊 Sector Rotation** - US sector analysis and economic cycle detection
8. **🎯 Factor Analysis** - Smart beta and factor timing strategies
9. **💎 Stock Fundamentals** - Individual stock deep-dive
10. **🧠 AI Sentiment** - FinBERT news sentiment analysis
11. **⚠️ Risk Analytics** - Comprehensive risk metrics and market breadth
12. **⚖️ Portfolio Analytics** - Custom portfolio backtesting

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

Create `.streamlit/secrets.toml` with your API keys:

```toml
[tiingo]
token = "YOUR_TIINGO_API_KEY"

[fred]
api_key = "YOUR_FRED_API_KEY"

[reddit]
client_id = "YOUR_REDDIT_CLIENT_ID"
client_secret = "YOUR_REDDIT_SECRET"
user_agent = "MyDash/1.0"
```

### Run Locally

```bash
streamlit run dashboard.py
```

## 🔑 API Keys (Optional but Recommended)

- **TIINGO API** (Free): https://www.tiingo.com/
- **FRED API** (Free): https://fred.stlouisfed.org/docs/api/api_key.html
- **Reddit API** (Free): https://www.reddit.com/prefs/apps

## 💎 Value Proposition

**Equivalent Professional Tools:**
- Bloomberg Terminal: $25,000/year
- Koyfin Pro: $600/year
- TradingView Premium: $300/year

**This Dashboard: FREE** (just API keys)

## 📚 Documentation

See included markdown files for detailed documentation:
- `DASHBOARD_REORGANIZATION.md` - Dashboard structure overview
- `TRADING_INSIGHTS_ADDED.md` - Trading signal guide
- `GLOBAL_VIEW_SIMPLIFIED.md` - Asset coverage details

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Data Sources:** OpenBB Platform, TIINGO, FRED, Yahoo Finance
- **Analytics:** Pandas, NumPy, Plotly
- **AI/ML:** Transformers (FinBERT), PyTorch

## 📝 License

For educational and personal use only.

## 🤝 Contributing

Issues and pull requests welcome!

---

Built with ❤️ for macro traders and quant investors
