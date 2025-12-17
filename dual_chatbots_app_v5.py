# dual_chatbots_app_v5.py
# Streamlit UI Wrapper for 3 chatbots: market data, plotting, knowledge-locked Q&A
# V5 changes:
#  - add start and end date price for price change/return function. Add asset 1 and asset 2 price for comparison function
#  - add "Erase" button

from __future__ import annotations

import os
from pathlib import Path
from datetime import date
import json
import html as html_mod
import hashlib
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors


# ============================================================
# OpenAI key bootstrap (MUST run before importing other modules)
# ============================================================

def _bootstrap_openai_env_from_streamlit_secrets() -> None:
    """
    Streamlit Community Cloud best practice:
    - Put OPENAI_API_KEY in Streamlit Secrets
    - Copy it into process env var so any imported modules that rely on os.environ can work
    """
    try:
        if "OPENAI_API_KEY" in st.secrets and st.secrets["OPENAI_API_KEY"]:
            # Only set if not already set (allows local env usage)
            os.environ.setdefault("OPENAI_API_KEY", st.secrets["OPENAI_API_KEY"])
    except Exception:
        # st.secrets may not be available in some local contexts; ignore
        pass

_bootstrap_openai_env_from_streamlit_secrets()


def get_openai_key() -> str | None:
    """
    Unified key getter used by this wrapper file.
    Priority:
    1) Streamlit secrets
    2) Environment variable (local setx / system env / cloud env)
    """
    try:
        if "OPENAI_API_KEY" in st.secrets and st.secrets["OPENAI_API_KEY"]:
            return str(st.secrets["OPENAI_API_KEY"])
    except Exception:
        pass
    k = os.getenv("OPENAI_API_KEY")
    return k if k else None


@st.cache_resource(show_spinner=False)
def get_openai_client():
    """
    One shared OpenAI client instance for this app process.
    """
    api_key = get_openai_key()
    if not api_key:
        return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    except Exception:
        return None


# ---------- Import your existing tools (no changes to their code) ----------
# NOTE: These imports must happen AFTER we bootstrap OPENAI_API_KEY into env.

# Market Monitor v31 (Q&A over workbook + manual calculator)
from marketMonitorChatbot_AgenticAI_Canonicalization_Batch_v31 import (
    MarketMonitorChatbot,
    MarketData,
    _iso,                  # helper for dates
    EXCEL_PATH,            # default workbook path
    render_manual_calculator,
)

# Time-Series analyzer (parsing + charts)
from timeseries_tool_v6 import (
    load_prices,
    analyze_asset,
    load_mapping,
    llm_extract,
    PerfStats,
)

# Embeddings client for the deterministic KB bot (shared client)
KB_OPENAI_CLIENT = get_openai_client()


# ---------- Page setup ----------

TOOLTIP_CSS = """
<style>
.mm-hover-wrapper {
  position: relative;
  display: inline-block;
}
.mm-hover-label {
  font-weight: 600;
  cursor: default;
}
.mm-tooltip {
  display: none;
  position: absolute;
  z-index: 9999;
  top: 1.6rem;
  left: 0;
  background-color: #111;
  color: #f9f9f9;
  padding: 8px 10px;
  border-radius: 4px;
  box-shadow: 0 2px 6px rgba(0,0,0,0.4);
  font-size: 0.85rem;
  line-height: 1.4;
}
.mm-hover-wrapper:hover .mm-tooltip {
  display: block;
}
.mm-tooltip.tips {
  width: 260px;
}
.mm-tooltip.tickers {
  width: 420px;
  max-height: 320px;
  overflow-y: auto;
}
.mm-tooltip ul {
  margin: 4px 0 0 1.1rem;
  padding: 0;
}
.mm-tooltip li {
  margin-bottom: 4px;
}
.mm-tickers-list div {
  margin-bottom: 2px;
  white-space: normal;
}
.mm-tickers-list code {
  font-size: 0.8rem;
}
</style>
"""

st.set_page_config(
    page_title="Dual Chatbots — Market Monitor, Time Series, KB",
    layout="wide",
)

# Inject tooltip CSS once
st.markdown(TOOLTIP_CSS, unsafe_allow_html=True)


# ---------- CACHED LOADERS (isolated per tool) ----------

@st.cache_resource(show_spinner=True)
def get_mm(md_path: str) -> tuple[MarketData, MarketMonitorChatbot]:
    """Cache Market Monitor workbook + chatbot (v31)."""
    md = MarketData(md_path)
    bot = MarketMonitorChatbot(md)
    return md, bot

@st.cache_resource(show_spinner=True)
def get_prices(file_path: str, sheet: str | None) -> pd.DataFrame:
    """Cache time-series DataFrame (timeseries_tool_v6)."""
    return load_prices(file_path, sheet=sheet)

@st.cache_resource(show_spinner=True)
def get_mapping(map_path: str) -> pd.DataFrame:
    """Cache ticker-name mapping for the time-series tool."""
    return load_mapping(map_path)


# ---------- Helpers ----------

def _metric_row(cols, items):
    for c, (label, value) in zip(cols, items):
        c.metric(label, value)

def _fmt_pct(x: float) -> str:
    return "NA" if pd.isna(x) else f"{x*100:,.2f}%"


# ============================================================
# TAB 1: Market Monitor Chatbot (v31 UI + manual calculator)
# ============================================================

def render_market_monitor_tab():
    st.header("📊 Market Monitor Chatbot")

    colL, colR = st.columns([0.6, 0.4])

    # Right column: status, tickers tooltip, tips (with hover), manual calculator
    with colR:
        st.subheader("Status")

        wb_env_default = os.environ.get("MARKET_MONITOR_XLSX", EXCEL_PATH)
        workbook = st.text_input(
            "Workbook (Excel path)",
            value=wb_env_default,
            key="mm_workbook",
        )

        # Load workbook + bot
        try:
            md, bot = get_mm(workbook)
        except Exception as e:
            st.error(f"Failed to load workbook: {e}")
            return

        show_debug = st.checkbox(
            "Show debug info (rewriter / router / candidates)",
            key="mm_show_debug",
            value=False,
        )

        st.caption("Loaded workbook & index")
        k1, k2, k3 = st.columns(3)
        k1.metric("Last data date", _iso(md.last_ts))
        k2.metric("# Tickers", f"{len(md.df_prices.columns):,}")
        k3.metric("# Rows", f"{md.df_prices.shape[0]:,}")

        # Hoverable "Tickers in data" (full scrollable list)
        tickers_in_prices = [str(t) for t in md.df_prices.columns]
        rows_html = []
        for t in tickers_in_prices:
            desc = md.ticker_to_name.get(t, "")
            rows_html.append(
                f"<div><code>{html_mod.escape(t)}</code> — "
                f"{html_mod.escape(desc)}</div>"
            )
        tickers_inner = (
            "\n".join(rows_html) or "<div><em>No tickers loaded.</em></div>"
        )
        tickers_tooltip_html = f"""
        <div class="mm-hover-wrapper">
          <span class="mm-hover-label">
            Tickers in data: {len(tickers_in_prices)} ▾
          </span>
          <div class="mm-tooltip tickers">
            <strong>Tickers in workbook ({len(tickers_in_prices)}):</strong><br/>
            <div class="mm-tickers-list">
              {tickers_inner}
            </div>
          </div>
        </div>
        """
        st.markdown(tickers_tooltip_html, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("**Tips**")
        st.markdown(
            "- Ask things like:\n"
            "  - *How's the market doing?*\n"
            "  - *Spread between US 10Y swap rate and Treasury rate*\n"
            "  - *Canada BBB 10Y corporate yield change YTD*\n"
            "  - *How much higher is US 3M vs Canada?*\n"
        )

        extra_tips_questions = [
            "What's the YTD return for SPX?",
            "How's the TSX doing month-to-date?",
            "Compare US 2Y vs 10Y government yields.",
            "Spread between US 10Y swap rate and Treasury rate.",
            "Canada BBB 10Y corporate yield change YTD.",
            "How much higher is US 3M bill vs Canada 3M bill?",
            "Return of HSI Index over the last 6 months.",
            "Compare SPX vs SPTSX60 Index today.",
            "How has SHCOMP done YTD?",
            "Spread between US 10Y gov yield and swap rate.",
        ]
        extra_items = "".join(
            f"<li>{html_mod.escape(q)}</li>" for q in extra_tips_questions
        )
        tips_tooltip_html = f"""
        <div class="mm-hover-wrapper">
          <span class="mm-hover-label">More tips ▾</span>
          <div class="mm-tooltip tips">
            <strong>More example questions:</strong>
            <ul>
              {extra_items}
            </ul>
          </div>
        </div>
        """
        st.markdown(tips_tooltip_html, unsafe_allow_html=True)

        st.markdown("---")

        render_manual_calculator(md, bot)

    with colL:
        st.subheader("Ask a question")

        if st.button("Erase all", key="mm_erase_btn"):
            st.session_state["mm_history"] = []

        if "mm_history" not in st.session_state:
            st.session_state["mm_history"] = []

        with st.form(key="mm_form", clear_on_submit=True):
            mm_q = st.text_input(
                " ",
                placeholder="e.g., How's the market doing?",
                key="mm_question",
            )
            submitted = st.form_submit_button("Send", use_container_width=False)

        if submitted and mm_q.strip():
            try:
                resp = bot.answer(mm_q.strip())
            except Exception as e:
                resp = {
                    "used_function": "error",
                    "message": f"Exception: {type(e).__name__}: {e}",
                    "router_decision": {},
                }
            st.session_state["mm_history"].append({"q": mm_q.strip(), "resp": resp})

        for i, item in enumerate(st.session_state["mm_history"], 1):
            q = item["q"]
            resp = item["resp"]
            st.markdown(f"**You #{i}:** {q}")
            st.write("")

            with st.container(border=True):
                used_fn = resp.get("used_function")
                params = resp.get("parameters", {}) or {}
                result = resp.get("result", {}) or {}
                rewriter = resp.get("rewriter", {}) or {}

                if rewriter.get("final_query"):
                    st.markdown(f"**Rewritten:** {rewriter.get('final_query')}")
                st.markdown(f"**Function:** `{used_fn or 'none'}`")

                if used_fn in {"price", "price_change", "return", "comparison"}:
                    if used_fn == "price":
                        st.markdown(
                            f"**Asset:** {params.get('asset_label') or params.get('asset')}"
                        )
                        st.markdown(f"**Requested date:** {params.get('date')}")
                        st.markdown(
                            f"**Price (actual date {result.get('date_actual')}):** "
                            f"{result.get('price')}"
                        )

                    elif used_fn == "price_change":
                        asset_label = params.get("asset_label") or params.get("asset")
                        st.markdown(f"**Asset:** {asset_label}")
                        st.markdown(
                            f"**Requested range:** {params.get('start_date')} → {params.get('end_date')}"
                        )
                        st.markdown(
                            f"**Prices (aligned to data):** "
                            f"{result.get('start_date_actual')} = {result.get('start_value')}, "
                            f"{result.get('end_date_actual')} = {result.get('end_value')}"
                        )
                        st.markdown(
                            f"**Change:** {result.get('absolute_change')}  |  "
                            f"**Pct:** {result.get('pct_change')}%"
                        )

                    elif used_fn == "return":
                        asset_label = params.get("asset_label") or params.get("asset")
                        st.markdown(f"**Asset:** {asset_label}")
                        st.markdown(
                            f"**Requested range:** {params.get('start_date')} → {params.get('end_date')}"
                        )
                        st.markdown(
                            f"**Prices (aligned to data):** "
                            f"{result.get('start_date_actual')} = {result.get('start_value')}, "
                            f"{result.get('end_date_actual')} = {result.get('end_value')}"
                        )
                        st.markdown(
                            f"**Total Return:** {result.get('total_return_pct')}%"
                        )

                    elif used_fn == "comparison":
                        a1_label = params.get("asset1_label") or params.get("asset1")
                        a2_label = params.get("asset2_label") or params.get("asset2")
                        st.markdown(f"**Asset 1:** {a1_label}")
                        st.markdown(f"**Asset 2:** {a2_label}")
                        st.markdown(
                            f"**Date (requested → actual):** "
                            f"{params.get('date')} → {result.get('date_actual')}"
                        )
                        st.markdown(
                            f"**Prices:** {a1_label} = {result.get('asset1_value')}, "
                            f"{a2_label} = {result.get('asset2_value')}"
                        )
                        st.markdown(
                            f"**Spread (Asset 1 − Asset 2):** {result.get('spread')}"
                        )

                    st.caption(resp.get("message"))
                else:
                    st.warning(resp.get("message", "Unable to answer."))

                if show_debug:
                    with st.expander(
                        "Rewriter / Router / Params / Result (debug)", expanded=False
                    ):
                        st.code(
                            json.dumps(resp, indent=2, ensure_ascii=False),
                            language="json",
                        )

    st.markdown("---")


# ============================================================
# TAB 2: Time-Series Chat (parsing + charts)
# ============================================================

def render_timeseries_tab():
    st.header("📈 Time-Series Parser & Charts")
    left, right = st.columns([0.6, 0.4])

    if "ts_history" not in st.session_state:
        st.session_state["ts_history"] = []

    with right:
        st.subheader("Data")
        ts_file = st.text_input(
            "Prices file (CSV/XLSX)",
            value="Market Monitor - blp.xlsx",
            key="ts_file",
        )
        ts_sheet = st.text_input(
            "Excel sheet (optional)",
            value="BBG Data",
            key="ts_sheet",
        )
        map_path = st.text_input(
            "TickerNameMapping.csv path",
            value="TickerNameMapping.csv",
            key="ts_map",
        )

        df = None
        mapping_df = None
        try:
            df = get_prices(ts_file, ts_sheet or None)
            st.caption(f"Loaded prices: shape={df.shape}")
        except Exception as e:
            st.error(f"Failed to load price file: {e}")

        try:
            mapping_df = get_mapping(map_path)
            st.caption(f"Loaded mapping: {mapping_df.shape[0]} rows")
        except Exception as e:
            st.error(f"Failed to load mapping: {e}")

    with left:
        st.subheader("Ask for a chartable period")

        if st.button("Erase all", key="ts_erase_btn"):
            st.session_state["ts_history"] = []

        with st.form(key="ts_form", clear_on_submit=True):
            ts_q = st.text_input(
                " ",
                placeholder="e.g., Performance of S&P 500 from Jan 2024 to Jun 2024",
                key="ts_question",
            )
            ts_submit = st.form_submit_button("Parse & Plot")

        if ts_submit and ts_q.strip():
            parsed = None
            result = None
            err = None
            try:
                if mapping_df is None:
                    raise RuntimeError("Mapping not loaded.")
                parsed = llm_extract(ts_q.strip(), mapping_df)

                with st.expander("Parsed details"):
                    st.json(parsed)

                if df is None:
                    raise RuntimeError("Prices not loaded.")
                result = analyze_asset(
                    df,
                    parsed["asset"],
                    start=parsed.get("start"),
                    end=parsed.get("end"),
                    rf=0.0,
                )
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
            st.session_state["ts_history"].append(
                {"q": ts_q.strip(), "parsed": parsed, "result": result, "error": err}
            )

        for i, item in enumerate(st.session_state["ts_history"], 1):
            st.markdown(f"**You #{i}:** {item['q']}")
            with st.container(border=True):
                if item["error"]:
                    st.error(item["error"])
                    continue

                parsed = item["parsed"]
                result = item["result"]
                if parsed:
                    st.markdown(
                        f"**Asset:** {parsed.get('asset_name')} "
                        f"(`{parsed.get('asset')}`)"
                    )
                    st.markdown(
                        f"**Range:** {parsed.get('start')} → {parsed.get('end')}"
                    )
                if result:
                    s: PerfStats = result["stats"]
                    k1, k2, k3, k4, k5, k6 = st.columns(6)
                    _metric_row(
                        (k1, k2, k3, k4, k5, k6),
                        [
                            ("CAGR", _fmt_pct(s.cagr)),
                            ("Ann. Vol", _fmt_pct(s.ann_vol)),
                            ("Sharpe", f"{s.sharpe:.2f}" if not pd.isna(s.sharpe) else "NA"),
                            ("Max DD", _fmt_pct(s.max_dd)),
                            ("DD Length", f"{s.max_dd_len}d"),
                            ("Win Rate", _fmt_pct(s.win_rate)),
                        ],
                    )
                    outdir: Path = result["output_dir"]
                    imgs = sorted([p for p in outdir.glob("*.png")])
                    if imgs:
                        grid = st.columns(2)
                        for j, pth in enumerate(imgs):
                            with grid[j % 2]:
                                st.image(
                                    str(pth),
                                    use_container_width=True,
                                    caption=pth.name,
                                )
                    else:
                        st.info("No charts were generated.")


# ============================================================
# TAB 3: Deterministic KB Chatbot (Knowledge-Locked)
# ============================================================

KB_CSV_PATH = "knowledge_base.csv"
EMBED_CACHE_PATH = "kb_embeds.npz"
MODEL_NAME = "text-embedding-3-large"
DEFAULT_THRESHOLD = 0.25
DEFAULT_MARGIN = 0.05
REQUIRED_COLS = ["question", "answer"]
OPTIONAL_COLS = ["id", "aliases", "tags"]


def kb_get_embedding(texts: List[str], model: str = MODEL_NAME) -> np.ndarray:
    if KB_OPENAI_CLIENT is None:
        raise RuntimeError(
            "OpenAI client not initialized. Add OPENAI_API_KEY to Streamlit Secrets "
            "(Manage app → Settings → Secrets)."
        )

    out_vectors: List[List[float]] = []
    BATCH = 96
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        resp = KB_OPENAI_CLIENT.embeddings.create(model=model, input=batch)
        out_vectors.extend([d.embedding for d in resp.data])
    arr = np.array(out_vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr / norms


# (rest of your KB code unchanged)
# ... keep everything below exactly as you had it ...

# ---------- Main layout ----------
tab1, tab2, tab3 = st.tabs(
    [
        "Market Monitor Chatbot",
        "Time-Series & Charts",
        "Knowledge-Locked Q&A",
    ]
)

with tab1:
    render_market_monitor_tab()
with tab2:
    render_timeseries_tab()
with tab3:
    render_knowledge_tab()
