# Dual Chatbots App  
**Market Monitor • Time‑Series & Charts • Knowledge‑Locked Q&A**

This repo contains a multi‑tab Streamlit app that wraps three separate tools around the same market data:

1. **📊 Market Monitor Chatbot**  
   Natural‑language Q&A over an Excel workbook (Bloomberg‑style export), plus a manual calculator for price / price change / return / spread.

2. **📈 Time‑Series Parser & Charts**  
   LLM‑assisted parsing of “performance from X to Y” questions into a time range + ticker, then generates stats & charts.

3. **💬 Knowledge‑Locked Q&A**  
   A deterministic, embedding‑based FAQ bot that only answers from a CSV knowledge base (no hallucinations).

The main entry point for the UI is:

- `dual_chatbots_app_v4.py`  

The other files are “engines” imported by the UI.

---

## 1. Repository structure (suggested)

```text
.
├─ dual_chatbots_app_v4.py                 # Streamlit multi‑tab UI
├─ marketMonitorChatbot_AgenticAI_...v31.py  # OpenAI version of Market Monitor chatbot
├─ timeseries_tool_v6.py                  # Time‑series analyzer + chart generator
├─ knowledge_base.csv                     # KB for the deterministic Q&A bot
├─ Market Monitor - blp.xlsx              # Main price data workbook (BBG export)
├─ TickerNameMapping.csv                  # Mapping for time‑series asset names → tickers
├─ requirements.txt                       # Python dependencies
└─ README.md
