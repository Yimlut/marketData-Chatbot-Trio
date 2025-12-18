# dual_chatbots_app_v5.py
# Streamlit UI Wrapper for 3 chatbots: market data, plotting, knowledge-locked Q&A
# V5 changes:
#  - add start and end date price for price change/return function. Add asset 1 and asset 2 price for comparison function
#  - add "Erase" button
#
# v5.1 (this change):
#  - auto-refresh cached workbook/prices/mapping when underlying files change (mtime stamp)
#  - add "Reload data" buttons for Market Monitor + Time-Series
#  - KB cache writes to /tmp for Streamlit Cloud safety
#  - Add displaying closest matches if cannot find an asset that meets the creterier

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
            os.environ.setdefault("OPENAI_API_KEY", st.secrets["OPENAI_API_KEY"])
    except Exception:
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


# ---------- Import your existing tools ----------
# NOTE: These imports must happen AFTER we bootstrap OPENAI_API_KEY into env.

from marketMonitorChatbot_AgenticAI_Canonicalization_Batch_v31 import (
    MarketMonitorChatbot,
    MarketData,
    _iso,
    EXCEL_PATH,
    render_manual_calculator,
)

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

st.markdown(TOOLTIP_CSS, unsafe_allow_html=True)


# ---------- File stamp helper (NEW) ----------

def _file_stamp(path: str) -> float:
    """
    Returns a numeric "version" for a local file, used to bust Streamlit caches.
    If file doesn't exist or isn't accessible, returns 0.
    """
    try:
        return float(os.path.getmtime(path))
    except Exception:
        return 0.0


# ---------- CACHED LOADERS (isolated per tool) ----------

@st.cache_resource(show_spinner=True)
def get_mm(md_path: str, stamp: float) -> tuple[MarketData, MarketMonitorChatbot]:
    """
    Cache Market Monitor workbook + chatbot (v31).
    The 'stamp' argument forces a refresh when the workbook file changes.
    """
    _ = stamp  # included only to invalidate cache when file changes
    md = MarketData(md_path)
    bot = MarketMonitorChatbot(md)
    return md, bot


@st.cache_resource(show_spinner=True)
def get_prices(file_path: str, sheet: str | None, stamp: float) -> pd.DataFrame:
    """
    Cache time-series DataFrame (timeseries_tool_v6).
    The 'stamp' argument forces a refresh when the file changes.
    """
    _ = stamp
    return load_prices(file_path, sheet=sheet)


@st.cache_resource(show_spinner=True)
def get_mapping(map_path: str, stamp: float) -> pd.DataFrame:
    """
    Cache ticker-name mapping for the time-series tool.
    The 'stamp' argument forces a refresh when the mapping file changes.
    """
    _ = stamp
    return load_mapping(map_path)


# ---------- Helpers ----------

def _metric_row(cols, items):
    for c, (label, value) in zip(cols, items):
        c.metric(label, value)

def _fmt_pct(x: float) -> str:
    return "NA" if pd.isna(x) else f"{x*100:,.2f}%"


# ============================================================
# TAB 1: Market Monitor Chatbot
# ============================================================

def render_market_monitor_tab():
    st.header("📊 Market Monitor Chatbot")

    colL, colR = st.columns([0.6, 0.4])

    with colR:
        st.subheader("Status")

        wb_env_default = os.environ.get("MARKET_MONITOR_XLSX", EXCEL_PATH)
        workbook = st.text_input(
            "Workbook (Excel path)",
            value=wb_env_default,
            key="mm_workbook",
        )

        # Manual reload button (NEW)
        if st.button("🔄 Reload workbook data", key="mm_reload_btn", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()

        # Load workbook + bot (cache busted by file stamp)
        try:
            stamp = _file_stamp(workbook)
            md, bot = get_mm(workbook, stamp)
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

        tickers_in_prices = [str(t) for t in md.df_prices.columns]
        rows_html = []
        for t in tickers_in_prices:
            desc = md.ticker_to_name.get(t, "")
            rows_html.append(
                f"<div><code>{html_mod.escape(t)}</code> — "
                f"{html_mod.escape(desc)}</div>"
            )
        tickers_inner = "\n".join(rows_html) or "<div><em>No tickers loaded.</em></div>"
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
        extra_items = "".join(f"<li>{html_mod.escape(q)}</li>" for q in extra_tips_questions)
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
                        st.markdown(f"**Asset:** {params.get('asset_label') or params.get('asset')}")
                        st.markdown(f"**Requested date:** {params.get('date')}")
                        st.markdown(
                            f"**Price (actual date {result.get('date_actual')}):** {result.get('price')}"
                        )

                    elif used_fn == "price_change":
                        asset_label = params.get("asset_label") or params.get("asset")
                        st.markdown(f"**Asset:** {asset_label}")
                        st.markdown(f"**Requested range:** {params.get('start_date')} → {params.get('end_date')}")
                        st.markdown(
                            f"**Prices (aligned to data):** "
                            f"{result.get('start_date_actual')} = {result.get('start_value')}, "
                            f"{result.get('end_date_actual')} = {result.get('end_value')}"
                        )
                        st.markdown(
                            f"**Change:** {result.get('absolute_change')}  |  **Pct:** {result.get('pct_change')}%"
                        )

                    elif used_fn == "return":
                        asset_label = params.get("asset_label") or params.get("asset")
                        st.markdown(f"**Asset:** {asset_label}")
                        st.markdown(f"**Requested range:** {params.get('start_date')} → {params.get('end_date')}")
                        st.markdown(
                            f"**Prices (aligned to data):** "
                            f"{result.get('start_date_actual')} = {result.get('start_value')}, "
                            f"{result.get('end_date_actual')} = {result.get('end_value')}"
                        )
                        st.markdown(f"**Total Return:** {result.get('total_return_pct')}%")

                    elif used_fn == "comparison":
                        a1_label = params.get("asset1_label") or params.get("asset1")
                        a2_label = params.get("asset2_label") or params.get("asset2")
                        st.markdown(f"**Asset 1:** {a1_label}")
                        st.markdown(f"**Asset 2:** {a2_label}")
                        st.markdown(
                            f"**Date (requested → actual):** {params.get('date')} → {result.get('date_actual')}"
                        )
                        st.markdown(
                            f"**Prices:** {a1_label} = {result.get('asset1_value')}, {a2_label} = {result.get('asset2_value')}"
                        )
                        st.markdown(f"**Spread (Asset 1 − Asset 2):** {result.get('spread')}")

                    st.caption(resp.get("message"))
                else:
                    # Show the main message
                    st.warning(resp.get("message", "Unable to answer."))

                    # NEW: if the backend returned suggestions, display them
                    suggestions = resp.get("suggestions") or []
                    if suggestions:
                        st.markdown("**Closest matches in this workbook:**")

                        # Pretty bullets (good UX)
                        for s in suggestions[:10]:
                            t = s.get("ticker", "")
                            n = s.get("name", "")
                            sc = s.get("score", None)
                            if sc is not None:
                                st.write(f"- `{t}` — {n} (score: {sc})")
                            else:
                                st.write(f"- `{t}` — {n}")

                        # Optional: show full table in an expander
                        with st.expander("Show as table", expanded=False):
                            df_sug = pd.DataFrame(suggestions)
                            # keep consistent columns if present
                            cols = [c for c in ["ticker", "name", "score"] if c in df_sug.columns]
                            if cols:
                                df_sug = df_sug[cols]
                            st.dataframe(df_sug, use_container_width=True, hide_index=True)


                if show_debug:
                    with st.expander("Rewriter / Router / Params / Result (debug)", expanded=False):
                        st.code(json.dumps(resp, indent=2, ensure_ascii=False), language="json")

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

        ts_file = st.text_input("Prices file (CSV/XLSX)", value="Market Monitor - blp.xlsx", key="ts_file")
        ts_sheet = st.text_input("Excel sheet (optional)", value="BBG Data", key="ts_sheet")
        map_path = st.text_input("TickerNameMapping.csv path", value="TickerNameMapping.csv", key="ts_map")

        # Manual reload button (NEW)
        if st.button("🔄 Reload time-series inputs", key="ts_reload_btn", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()

        df = None
        mapping_df = None

        try:
            df_stamp = _file_stamp(ts_file)
            df = get_prices(ts_file, ts_sheet or None, df_stamp)
            st.caption(f"Loaded prices: shape={df.shape}")
        except Exception as e:
            st.error(f"Failed to load price file: {e}")

        try:
            map_stamp = _file_stamp(map_path)
            mapping_df = get_mapping(map_path, map_stamp)
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
                    st.markdown(f"**Asset:** {parsed.get('asset_name')} (`{parsed.get('asset')}`)")
                    st.markdown(f"**Range:** {parsed.get('start')} → {parsed.get('end')}")

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
                                st.image(str(pth), use_container_width=True, caption=pth.name)
                    else:
                        st.info("No charts were generated.")


# ============================================================
# TAB 3: Deterministic KB Chatbot (Knowledge-Locked)
# ============================================================

KB_CSV_PATH = "knowledge_base.csv"
EMBED_CACHE_PATH = str(Path("/tmp") / "kb_embeds.npz")  # Streamlit Cloud-safe
MODEL_NAME = "text-embedding-3-large"
DEFAULT_THRESHOLD = 0.25
DEFAULT_MARGIN = 0.05
REQUIRED_COLS = ["question", "answer"]
OPTIONAL_COLS = ["id", "aliases", "tags"]


def kb_get_embedding(texts: List[str], model: str = MODEL_NAME) -> np.ndarray:
    if KB_OPENAI_CLIENT is None:
        raise RuntimeError(
            "OpenAI client not initialized. Add OPENAI_API_KEY to Streamlit Secrets."
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


def kb_ensure_sample_kb():
    if not os.path.exists(KB_CSV_PATH):
        df = pd.DataFrame([
            {
                "id": 1,
                "question": "Where are the market data from?",
                "answer": "All market data in this tool are sourced from the Excel workbook you configured (e.g. Bloomberg exports).",
                "aliases": "data source|where do the data come from|origin of data",
                "tags": "meta,data",
            },
            {
                "id": 2,
                "question": "How often is the market data updated?",
                "answer": "The data are updated whenever you refresh or replace the underlying Excel file. There is no automatic real-time feed.",
                "aliases": "update frequency|refresh frequency|how often updated",
                "tags": "meta,data",
            },
        ])
        df.to_csv(KB_CSV_PATH, index=False)


def kb_load_kb(path: str = KB_CSV_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        kb_ensure_sample_kb()
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    for col in REQUIRED_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {path}")
    for col in OPTIONAL_COLS:
        if col not in df.columns:
            df[col] = ""
    df = df.fillna({"aliases": "", "tags": "", "id": ""})
    return df


def kb_make_search_text(row: pd.Series, include_answer_in_index: bool) -> str:
    pieces = [
        str(row.get("question", "")),
        str(row.get("aliases", "")),
        str(row.get("tags", "")),
    ]
    if include_answer_in_index:
        pieces.append(str(row.get("answer", "")))
    joined = " ".join(pieces).replace("|", " ")
    return joined.strip()


def kb_fingerprint(df: pd.DataFrame, include_answer_in_index: bool) -> str:
    proj_cols = ["id", "question", "answer", "aliases", "tags"]
    tmp = df[proj_cols].copy()
    tmp["__indexed"] = df.apply(lambda r: kb_make_search_text(r, include_answer_in_index), axis=1)
    payload = tmp.to_json(orient="records", force_ascii=False)
    return hashlib.sha256((payload + f"|include_answer={include_answer_in_index}").encode("utf-8")).hexdigest()


def kb_build_or_load_index(
    df: pd.DataFrame,
    include_answer_in_index: bool = False,
) -> Tuple[NearestNeighbors, np.ndarray, pd.DataFrame]:
    df_aug = df.copy()
    df_aug["search_text"] = df_aug.apply(lambda r: kb_make_search_text(r, include_answer_in_index), axis=1)
    fp = kb_fingerprint(df_aug, include_answer_in_index)

    vectors: np.ndarray
    if os.path.exists(EMBED_CACHE_PATH):
        try:
            cache = np.load(EMBED_CACHE_PATH, allow_pickle=True)
            if cache["fingerprint"].item() == fp:
                vectors = cache["vectors"].astype(np.float32)
            else:
                raise ValueError("Fingerprint mismatch; will rebuild.")
        except Exception:
            vectors = kb_get_embedding(df_aug["search_text"].tolist())
            np.savez(EMBED_CACHE_PATH, vectors=vectors, fingerprint=np.array(fp, dtype=object))
    else:
        vectors = kb_get_embedding(df_aug["search_text"].tolist())
        np.savez(EMBED_CACHE_PATH, vectors=vectors, fingerprint=np.array(fp, dtype=object))

    knn = NearestNeighbors(n_neighbors=min(10, len(df_aug)), algorithm="auto", metric="cosine")
    knn.fit(vectors)
    return knn, vectors, df_aug


def kb_search(
    df_aug: pd.DataFrame,
    knn: NearestNeighbors,
    matrix: np.ndarray,
    query: str,
    top_k: int = 5,
):
    q_vec = kb_get_embedding([query])[0]
    distances, indices = knn.kneighbors(q_vec.reshape(1, -1), n_neighbors=min(top_k, len(df_aug)))
    distances = distances[0]
    indices = indices[0]
    sims = 1.0 - distances

    results = []
    for rank, (idx, sim) in enumerate(zip(indices, sims), start=1):
        row = df_aug.iloc[idx]
        results.append(
            {
                "rank": rank,
                "index": int(idx),
                "similarity": float(sim),
                "id": row.get("id", ""),
                "question": row["question"],
                "answer": row["answer"],
                "aliases": row.get("aliases", ""),
                "tags": row.get("tags", ""),
            }
        )
    return results


def render_knowledge_tab():
    st.header("💬 Knowledge-Locked Q&A (Deterministic Retrieval)")
    st.write(
        "This bot **only** answers using the provided knowledge base. "
        "It returns stored answers verbatim (no text generation), so there are no hallucinations."
    )

    if KB_OPENAI_CLIENT is None:
        st.error("OpenAI API key not configured for the Knowledge-Locked bot.")
        st.info('Add Streamlit Secret:  OPENAI_API_KEY = "sk-..."')
        return

    left, right = st.columns([0.6, 0.4])

    with right:
        st.subheader("Knowledge Base & Index")

        upload_file = st.file_uploader(
            "Upload a knowledge_base.csv",
            type=["csv"],
            accept_multiple_files=False,
            key="kb_upload",
        )

        include_ans_default = st.session_state.get("kb_include_ans", False)
        threshold_default = st.session_state.get("kb_threshold", DEFAULT_THRESHOLD)
        margin_default = st.session_state.get("kb_margin", DEFAULT_MARGIN)

        include_ans = st.checkbox(
            "Also index the 'answer' text",
            value=include_ans_default,
            help="If on, semantic search can match words in answers as well as questions / aliases.",
            key="kb_include_ans_cb",
        )
        threshold = st.slider(
            "Similarity threshold for a match",
            0.0,
            1.0,
            float(threshold_default),
            0.01,
            help="Below this, the bot will say 'Not in my knowledge base'.",
            key="kb_threshold_slider",
        )
        margin = st.slider(
            "Ambiguity margin (top1 - top2)",
            0.0,
            0.50,
            float(margin_default),
            0.01,
            help="If top match isn't better than 2nd by at least this margin, treat as ambiguous.",
            key="kb_margin_slider",
        )

        rebuild_clicked = st.button("🔄 Rebuild index", key="kb_rebuild_btn")

        if upload_file is not None:
            try:
                df_kb = pd.read_csv(upload_file)
                df_kb.columns = [c.strip().lower() for c in df_kb.columns]
                kb_source = f"uploaded:{upload_file.name}"
            except Exception as e:
                st.error(f"Failed to read uploaded KB: {e}")
                return
        else:
            try:
                df_kb = kb_load_kb(KB_CSV_PATH)
                kb_source = f"file:{KB_CSV_PATH}"
            except Exception as e:
                st.error(f"Failed to load KB: {e}")
                return

        st.caption(f"KB rows: {df_kb.shape[0]}")

        need_rebuild = (
            "kb_knn" not in st.session_state
            or rebuild_clicked
            or st.session_state.get("kb_include_ans") != include_ans
            or st.session_state.get("kb_source") != kb_source
        )

        if need_rebuild:
            st.info("Building index…")
            try:
                knn, mat, df_aug = kb_build_or_load_index(df_kb, include_answer_in_index=include_ans)
            except Exception as e:
                st.error(f"Failed to build index: {e}")
                return
            st.session_state["kb_knn"] = knn
            st.session_state["kb_mat"] = mat
            st.session_state["kb_df_aug"] = df_aug
            st.session_state["kb_source"] = kb_source
            st.session_state["kb_include_ans"] = include_ans
        else:
            knn = st.session_state["kb_knn"]
            mat = st.session_state["kb_mat"]
            df_aug = st.session_state["kb_df_aug"]

        st.session_state["kb_threshold"] = float(threshold)
        st.session_state["kb_margin"] = float(margin)

        with st.expander("📘 How to structure knowledge_base.csv", expanded=False):
            st.markdown(
                """
**Required columns**
- `question`: canonical question text.
- `answer`: the exact answer text to return verbatim.

**Optional columns**
- `id`: any identifier.
- `aliases`: pipe-separated synonyms.
- `tags`: arbitrary tags.
                """
            )

    with left:
        st.subheader("Ask about the data / methodology")

        if "kb_messages" not in st.session_state:
            st.session_state["kb_messages"] = []

        if st.button("Erase all", key="kb_erase_btn"):
            st.session_state["kb_messages"] = []

        for m in st.session_state["kb_messages"]:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        query = st.chat_input("Type your question…", key="kb_chat_input")
        if query:
            st.session_state["kb_messages"].append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            try:
                results = kb_search(df_aug, knn, mat, query, top_k=5)
                top = results[0] if results else None
                second = results[1] if results and len(results) > 1 else None
                threshold = st.session_state.get("kb_threshold", DEFAULT_THRESHOLD)
                margin = st.session_state.get("kb_margin", DEFAULT_MARGIN)

                if (not top) or (top["similarity"] < threshold):
                    answer_text = "_Not in my knowledge base. Try rephrasing or expand the KB._"
                    meta = None
                    ambiguous = False
                else:
                    if second is not None and (top["similarity"] - second["similarity"] < margin):
                        ambiguous = True
                        answer_text = (
                            "_I found multiple close matches and I'm not sure which one you meant._\n\n"
                            "Please rephrase your question to be closer to one of these:"
                        )
                        meta = None
                    else:
                        ambiguous = False
                        answer_text = str(top["answer"])
                        meta = {
                            "matched_question": top["question"],
                            "similarity": round(top["similarity"], 3),
                            "id": top["id"],
                        }

            except Exception as e:
                answer_text = f"_Error running search: {type(e).__name__}: {e}_"
                results = []
                meta = None
                ambiguous = False

            with st.chat_message("assistant"):
                st.markdown(answer_text)

                if results and ambiguous:
                    for r in results[:3]:
                        st.write(f"- **{r['question']}** (similarity: {r['similarity']:.3f})")

                if results and meta:
                    with st.expander("Match details", expanded=False):
                        st.write(f"**Matched Q:** {meta['matched_question']}")
                        st.write(f"**KB ID:** {meta['id']}")
                        st.write(f"**Similarity:** {meta['similarity']}")
                        df_dbg = pd.DataFrame(results)[["rank", "similarity", "id", "question", "aliases", "tags"]]
                        st.dataframe(df_dbg, use_container_width=True, hide_index=True)

            st.session_state["kb_messages"].append({"role": "assistant", "content": answer_text})


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
