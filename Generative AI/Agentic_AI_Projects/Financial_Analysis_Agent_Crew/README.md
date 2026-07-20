# 📈 AI Financial Analyst & Investment Strategist Swarm

A multi-agent financial intelligence swarm designed to scrape live market data, calculate pricing ratios, analyze company reports, and compile executive-grade stock strategy briefs. Powered by **Agno (formerly Phidata)**, **Gemini 2.0 Flash**, and **Yahoo Finance APIs**.

---

## 🧠 Swarm Architecture & Collaboration

The system delegates research and writing tasks across four specialized agent roles:

```
                  ┌──────────────────────┐
                  │     User Query       │
                  │ (e.g. AAPL,TSLA,NVDA)│
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │    Lead Strategist   │
                  │  (Team Coordinator)  │
                  └──────┬────────┬──────┘
                         │        │
            ┌────────────┘        └────────────┐
            ▼                                  ▼
 ┌────────────────────┐             ┌────────────────────┐
 │   Market Analyst   │             │ Company Researcher │
 │  (Price & Volatility)│           │ (Fundamentals/News)│
 └──────────┬─────────┘             └──────────┬─────────┘
            │                                  │
            └────────────┬─────────────────────┘
                         ▼
              ┌────────────────────┐
              │  Stock Strategist  │
              │  (Risk Scoring)    │
              └────────────────────┘
```

1. **Market Analyst Agent**: Collects historical price matrices, analyzes trading volume, and calculates relative performance trends.
2. **Company Fundementals Researcher**: Scrapes balance sheet parameters, industry sectors, profit margins, and recent news bulletins.
3. **Risk & Stock Strategist**: Estimates risk-to-reward metrics, relative valuation ratios, and industry sector headwinds.
4. **Lead Aggregator Agent**: Synthesizes inputs into a structured markdown report recommending top stocks to buy, hold, or sell.

---

## ✨ Features

- **Dynamic Data Scraping**: Real-time integration with `yfinance` to capture price action, volume, and company profiles.
- **Relative Analysis Model**: Calculates stock metrics over 6-month horizons.
- **Plotly Visuals**: Fully interactive dark-mode dashboard plots comparing tickers.
- **Multi-Agent Coordination**: Direct messaging agent nodes via the Agno orchestration engine.

---

## 🚀 Setup & Launch

1. **Install Dependencies**:
   ```bash
   pip install streamlit yfinance agno plotly google-generativeai
   ```

2. **Configure API Key**:
   Set your Gemini API token:
   ```bash
   export GEMINI_API_KEY="your_api_key_here"
   ```

3. **Start Dashboard**:
   ```bash
   streamlit run investment.py
   ```
