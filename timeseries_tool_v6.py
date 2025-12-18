#timeseries_tool_v6.py
"""
Financial Time Series Analysis Tool + Streamlit UI (+ Natural Language Parsing)
-------------------------------------------------------------------------------

- CLI: python timeseries_tool.py --file <path> --asset <column> [--sheet ... --start ... --end ... --rf ...]
- UI:  streamlit run timeseries_tool.py

Input: CSV/XLSX with a 'Date' column and one or more asset columns.

New (AI):
- Load TickerNameMapping.csv (first col = friendly asset Name, second col = Bloomberg Ticker)
- Natural language box → uses OpenAI API to extract start, end, asset (by Name)
- Auto-fills the UI parameters from the parsed result (you can edit before running)

Update:
- Relative date phrases like "past 5 years" are now anchored to a provided reference_date
  (ideally the last date in the dataset).
- OpenAI parsing failures are no longer silently swallowed; we return source=openai_error.
- Deterministic normalization: if OpenAI returns a Ticker in asset_name, map it to Name.
"""
from __future__ import annotations

import argparse
import math
import sys
import os
import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from datetime import date
from calendar import monthrange

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Matplotlib defaults (clean, readable) ----
plt.rcParams["figure.figsize"] = (10, 5.5)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["font.size"] = 10


# ===================== OpenAI Key Helper =====================

def _get_openai_key() -> str:
    """
    Return OpenAI API key if available.

    Priority:
      1) Environment variable OPENAI_API_KEY (local dev, CI, or if wrapper sets it)
      2) Streamlit secrets OPENAI_API_KEY (Streamlit Community Cloud)

    Returns "" if not found.
    """
    k = os.getenv("OPENAI_API_KEY", "").strip()
    if k:
        return k

    # Optional Streamlit secrets fallback (won't break CLI if streamlit isn't installed)
    try:
        import streamlit as st  # type: ignore
        sk = str(st.secrets.get("OPENAI_API_KEY", "")).strip()
        if sk:
            return sk
    except Exception:
        pass

    return ""


# ---------- Data Loading ----------
def load_prices(file: str | Path, sheet: Optional[str] = None) -> pd.DataFrame:
    file = Path(file)
    if not file.exists():
        raise FileNotFoundError(f"File not found: {file}")

    if file.suffix.lower() in [".xls", ".xlsx", ".xlsm"]:
        df = pd.read_excel(file, sheet_name=sheet)
    elif file.suffix.lower() in [".csv", ".txt"]:
        df = pd.read_csv(file)
    else:
        raise ValueError("Unsupported file type. Use CSV or Excel (.xlsx).")

    if "Date" not in df.columns:
        for cand in ["date", "DATE", "TradeDate", "dt"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "Date"})
                break

    if "Date" not in df.columns:
        raise ValueError("The input file must contain a 'Date' column.")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


# ---------- Utility Stats ----------
@dataclass
class PerfStats:
    cagr: float
    ann_vol: float
    sharpe: float
    calmar: float
    max_dd: float
    max_dd_len: int
    win_rate: float
    skew: float
    kurtosis: float
    start: pd.Timestamp
    end: pd.Timestamp
    n_days: int


def _cagr(prices: pd.Series) -> float:
    if len(prices) < 2:
        return np.nan
    start_val = prices.iloc[0]
    end_val = prices.iloc[-1]
    years = (prices.index[-1] - prices.index[0]).days / 365.25
    if years <= 0 or start_val <= 0:
        return np.nan
    return (end_val / start_val) ** (1 / years) - 1


def _max_drawdown(cumret: pd.Series) -> Tuple[float, int]:
    peaks = cumret.cummax()
    dd = (cumret / peaks) - 1.0
    min_dd = dd.min() if len(dd) else np.nan
    below = dd < 0
    lengths = (below.groupby((~below).cumsum()).cumcount() + 1) * below
    max_len = int(lengths.max()) if len(lengths) else 0
    return float(min_dd), max_len


def _annualize_vol(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    return float(daily_returns.std(ddof=0) * np.sqrt(periods_per_year))


def _sharpe(daily_returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    if daily_returns.std(ddof=0) == 0 or len(daily_returns) == 0:
        return np.nan
    adj = rf / periods_per_year
    return float((daily_returns.mean() - adj) / daily_returns.std(ddof=0) * np.sqrt(periods_per_year))


def compute_stats(prices: pd.Series, rf: float = 0.0) -> PerfStats:
    prices = prices.dropna()
    if len(prices) < 3:
        raise ValueError("Not enough data after dropping NaNs.")
    ret = prices.pct_change().dropna()
    cum = (1 + ret).cumprod()
    cagr = _cagr(prices)
    ann_vol = _annualize_vol(ret)
    sharpe = _sharpe(ret, rf=rf)
    max_dd, max_dd_len = _max_drawdown(cum)
    calmar = (cagr / abs(max_dd)) if (max_dd and max_dd != 0 and not math.isnan(cagr)) else np.nan
    non_zero = ret[ret != 0]
    win_rate = float((non_zero > 0).mean()) if len(non_zero) else float('nan')
    skew = float(ret.skew())
    kurt = float(ret.kurtosis())
    return PerfStats(
        cagr=cagr, ann_vol=ann_vol, sharpe=sharpe, calmar=calmar,
        max_dd=max_dd, max_dd_len=int(max_dd_len), win_rate=win_rate,
        skew=skew, kurtosis=kurt, start=prices.index[0], end=prices.index[-1], n_days=len(prices)
    )


# ---------- Plot Helpers (one plot per figure) ----------
def _savefig(fig, outdir: Path, name: str):
    outdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outdir / f"{name}.png", dpi=160)
    plt.close(fig)


def plot_price_with_sma(prices: pd.Series, outdir: Path, asset: str):
    sma20 = prices.rolling(20).mean()
    sma50 = prices.rolling(50).mean()
    sma200 = prices.rolling(200).mean()

    fig, ax = plt.subplots()
    ax.plot(prices.index, prices.values, label=f"{asset} Price", linewidth=1.5)
    ax.plot(sma20.index, sma20.values, label="SMA 20")
    ax.plot(sma50.index, sma50.values, label="SMA 50")
    ax.plot(sma200.index, sma200.values, label="SMA 200")
    ax.set_title(f"{asset} — Price with 20/50/200 SMA")
    ax.legend(loc="best")
    _savefig(fig, outdir, "01_price_sma")


def plot_drawdown(prices: pd.Series, outdir: Path, asset: str):
    ret = prices.pct_change().dropna()
    cum = (1 + ret).cumprod()
    peaks = cum.cummax()
    dd = (cum / peaks) - 1.0

    fig, ax = plt.subplots()
    ax.plot(dd.index, dd.values)
    ax.set_title(f"{asset} — Drawdown (Underwater)")
    ax.set_ylabel("Drawdown")
    _savefig(fig, outdir, "02_drawdown")


def plot_rolling_stats(prices: pd.Series, outdir: Path, asset: str):
    ret = prices.pct_change().dropna()
    roll = 252
    roll_vol = ret.rolling(roll).std() * np.sqrt(252)
    roll_sharpe = (ret.rolling(roll).mean() / ret.rolling(roll).std()) * np.sqrt(252)

    fig1, ax1 = plt.subplots()
    ax1.plot(roll_vol.index, roll_vol.values)
    ax1.set_title(f"{asset} — Rolling {roll//21}M Ann. Vol (252d)")
    ax1.set_ylabel("Annualized Volatility")
    _savefig(fig1, outdir, "03_rolling_vol")

    fig2, ax2 = plt.subplots()
    ax2.plot(roll_sharpe.index, roll_sharpe.values)
    ax2.set_title(f"{asset} — Rolling {roll//21}M Sharpe (no RF)")
    ax2.set_ylabel("Sharpe")
    _savefig(fig2, outdir, "04_rolling_sharpe")


def plot_return_hist(prices: pd.Series, outdir: Path, asset: str):
    ret = prices.pct_change().dropna()
    fig, ax = plt.subplots()
    ax.hist(ret.values, bins=50)
    ax.set_title(f"{asset} — Daily Return Distribution")
    ax.set_xlabel("Daily Return")
    ax.set_ylabel("Frequency")
    _savefig(fig, outdir, "05_hist_daily_returns")


def plot_monthly_bars(prices: pd.Series, outdir: Path, asset: str) -> pd.DataFrame:
    ret = prices.pct_change().dropna()
    monthly = (1 + ret).resample("M").prod() - 1
    fig, ax = plt.subplots()
    ax.bar(monthly.index, monthly.values, width=20)
    ax.set_title(f"{asset} — Monthly Returns")
    ax.set_ylabel("Return")
    _savefig(fig, outdir, "06_monthly_returns")
    return monthly.to_frame(name="MonthlyReturn")


def plot_autocorr(prices: pd.Series, outdir: Path, asset: str):
    ret = prices.pct_change().dropna()
    acf_vals = [ret.autocorr(lag) for lag in range(1, 13)]
    fig, ax = plt.subplots()
    ax.bar(range(1, 13), acf_vals)
    ax.set_title(f"{asset} — Autocorrelation (lags 1–12)")
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("ACF")
    _savefig(fig, outdir, "07_acf_1_12")


# ---------- Main API ----------
def analyze_asset(df: pd.DataFrame, asset: str, start: Optional[str] = None, end: Optional[str] = None,
                  rf: float = 0.0, outdir: Optional[str | Path] = None) -> dict:
    if asset not in df.columns:
        raise KeyError(f"Asset '{asset}' not found in columns.")

    px = df[asset].dropna()
    if start:
        px = px[px.index >= pd.to_datetime(start)]
    if end:
        px = px[px.index <= pd.to_datetime(end)]
    if len(px) < 10:
        raise ValueError("Insufficient data in the chosen period.")

    safe_asset = asset.replace("/", "_").replace(" ", "_")
    period_tag = f"{px.index[0].date()}_{px.index[-1].date()}"
    outdir = Path(outdir or Path("figures") / f"{safe_asset}_{period_tag}")

    stats = compute_stats(px, rf=rf)

    plot_price_with_sma(px, outdir, asset)
    plot_drawdown(px, outdir, asset)
    plot_rolling_stats(px, outdir, asset)
    monthly_df = plot_monthly_bars(px, outdir, asset)
    plot_return_hist(px, outdir, asset)
    plot_autocorr(px, outdir, asset)

    daily_ret = px.pct_change().dropna().to_frame(name="DailyReturn")
    annual = (1 + daily_ret["DailyReturn"]).resample("Y").prod() - 1
    monthly_pivot = (1 + daily_ret["DailyReturn"]).resample("M").prod() - 1
    monthly_table = monthly_pivot.to_frame(name="Return")
    monthly_table["Year"] = monthly_table.index.year
    monthly_table["Month"] = monthly_table.index.month
    monthly_table = monthly_table.pivot(index="Year", columns="Month", values="Return").sort_index()

    return {
        "asset": asset,
        "period": (px.index[0], px.index[-1]),
        "stats": stats,
        "daily_returns": daily_ret,
        "annual_returns": annual.to_frame(name="AnnualReturn"),
        "monthly_returns": monthly_df,
        "monthly_table": monthly_table,
        "output_dir": outdir,
    }


# ===================== AI HELPERS =====================

def load_mapping(mapping_path: str | Path) -> pd.DataFrame:
    """Load TickerNameMapping.* (CSV or XLSX). First col = Name, second col = Ticker."""
    mp = Path(mapping_path)
    if not mp.exists():
        raise FileNotFoundError(f"TickerNameMapping file not found: {mp}")
    df = pd.read_csv(mp) if mp.suffix.lower() == ".csv" else pd.read_excel(mp)
    if len(df.columns) < 2:
        raise ValueError("Mapping file needs at least two columns: Name, Ticker")
    df = df.rename(columns={df.columns[0]: "Name", df.columns[1]: "Ticker"})
    df["Name"] = df["Name"].astype(str)
    df["Ticker"] = df["Ticker"].astype(str)
    return df


def _quarter_end(y: int, q: int) -> date:
    m = {1: 3, 2: 6, 3: 9, 4: 12}[q]
    return date(y, m, monthrange(y, m)[1])


def _parse_dates_fallback(q: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Lightweight rules to catch common phrases if LLM is unavailable.
    Returns ISO strings (YYYY-MM-DD) or (None, None).
    """
    ql = q.lower()

    m = re.search(r"(20\d{2}|19\d{2})", ql)
    if m and ("begin" in ql or "start" in ql) and ("end" in ql or "through" in ql or "to " in ql):
        y = int(m.group(1))
        return f"{y}-01-01", f"{y}-12-31"

    m = re.search(r"(q[1-4]|first quarter|second quarter|third quarter|fourth quarter)\s*(?:of|,| )?\s*(20\d{2}|19\d{2})", ql)
    if m:
        qmap = {"q1": 1, "first quarter": 1, "q2": 2, "second quarter": 2,
                "q3": 3, "third quarter": 3, "q4": 4, "fourth quarter": 4}
        qnum = qmap[m.group(1)]
        y = int(m.group(2))
        start = date(y, (qnum - 1) * 3 + 1, 1)
        end = _quarter_end(y, qnum)
        return start.isoformat(), end.isoformat()

    m = re.search(
        r"from\s+(\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{4})\s+(?:to|through|until)\s+(\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{4})",
        ql,
    )
    if m:
        def norm(s):
            s = s.replace("/", "-")
            parts = [int(p) for p in s.split("-")]
            if parts[0] > 31:
                y, mth, d = parts
            else:
                mth, d, y = parts
            return f"{y:04d}-{mth:02d}-{d:02d}"
        return norm(m.group(1)), norm(m.group(2))

    return None, None


def llm_extract(
    query: str,
    mapping_df: pd.DataFrame,
    model: str = "gpt-4o-mini",
    reference_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Use OpenAI to map user's natural question to:
      - start (YYYY-MM-DD), end (YYYY-MM-DD), asset_name (exact from mapping list)
    Then map asset_name -> Ticker via mapping_df.

    Deterministic upgrades:
      - anchor relative dates to reference_date (treated as "today")
      - map ticker->name if the model returns a ticker in asset_name
      - case-insensitive exact match for asset_name
      - do NOT silently swallow OpenAI exceptions
    """
    allowed_assets: List[str] = mapping_df["Name"].dropna().astype(str).tolist()
    ref = reference_date or date.today().isoformat()

    api_key = _get_openai_key().strip()
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)

            system = (
                "Extract a time range and the best-matching asset from the allowed list.\n"
                f"Treat reference_date={ref} as 'today' for relative phrases like 'past 5 years', "
                "'last 12 months', YTD/MTD/QTD.\n"
                "Return strict JSON with keys: start, end, asset_name.\n"
                "Dates must be ISO YYYY-MM-DD.\n"
                "If the user does not provide an end date, set end = reference_date.\n"
                "Expand quarters/years to full ranges.\n"
                "asset_name MUST be one of the allowed names verbatim."
            )
            asset_blob = "\n".join(f"- {a}" for a in allowed_assets[:800])
            user = (
                f"Allowed assets:\n{asset_blob}\n\n"
                f"Question: '''{query}'''\n"
                "Respond JSON like: {\"start\":\"2020-01-01\",\"end\":\"2025-12-18\",\"asset_name\":\"S&P 500 INDEX\"}"
            )

            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)

            start = data.get("start")
            end = data.get("end") or ref
            asset_name = data.get("asset_name")
            asset_name = str(asset_name).strip() if asset_name is not None else None

            # Deterministic: if model returned a TICKER, map to Name via mapping_df
            ticker_set = set(mapping_df["Ticker"].astype(str))
            if asset_name and asset_name in ticker_set:
                asset_name = mapping_df.loc[mapping_df["Ticker"].astype(str) == asset_name, "Name"].iloc[0]

            # Deterministic: case-insensitive exact match for Name
            low_map = {a.lower(): a for a in allowed_assets}
            if asset_name and asset_name.lower() in low_map:
                asset_name = low_map[asset_name.lower()]

            if asset_name in allowed_assets and start and end:
                ticker = mapping_df.loc[mapping_df["Name"] == asset_name, "Ticker"].iloc[0]
                return {
                    "start": start,
                    "end": end,
                    "asset": ticker,
                    "asset_name": asset_name,
                    "source": "openai",
                    "reference_date": ref,
                }

            # OpenAI succeeded but didn't meet our contract
            return {
                "start": start,
                "end": end,
                "asset": None,
                "asset_name": asset_name,
                "source": "openai_unmatched",
                "reference_date": ref,
                "note": "OpenAI returned a result but asset_name was not an allowed Name (or start/end missing).",
            }

        except Exception as e:
            # IMPORTANT: don't silently fall back — return the real reason
            return {
                "start": None,
                "end": None,
                "asset": None,
                "asset_name": None,
                "source": "openai_error",
                "reference_date": ref,
                "error": f"{type(e).__name__}: {e}",
            }

    # Fallback: fuzzy name + simple date rules (original behavior)
    try:
        import difflib
        best = difflib.get_close_matches(query, allowed_assets, n=1, cutoff=0.2)
        asset_name = best[0] if best else allowed_assets[0]
    except Exception:
        asset_name = allowed_assets[0]

    ticker = mapping_df.loc[mapping_df["Name"] == asset_name, "Ticker"].iloc[0]
    s, e = _parse_dates_fallback(query)
    if e is None:
        e = ref

    return {
        "start": s,
        "end": e,
        "asset": ticker,
        "asset_name": asset_name,
        "source": "fallback",
        "reference_date": ref,
    }


# ---------- CLI ----------
def _fmt_pct(x: float) -> str:
    return "NA" if pd.isna(x) else f"{x*100:,.2f}%"


def main_cli():
    p = argparse.ArgumentParser(description="Financial Time Series Analysis Tool")
    p.add_argument("--file", required=True, help="Path to CSV or Excel with a 'Date' column.")
    p.add_argument("--sheet", default=None, help="Excel sheet name (if using .xlsx).")
    p.add_argument("--asset", required=True, help="Column name of the asset to analyze.")
    p.add_argument("--start", default=None, help="Start date (YYYY-MM-DD).")
    p.add_argument("--end", default=None, help="End date (YYYY-MM-DD).")
    p.add_argument("--rf", type=float, default=0.0, help="Annual risk-free rate (e.g., 0.02).")
    args = p.parse_args()

    df = load_prices(args.file, sheet=args.sheet)
    result = analyze_asset(df, args.asset, start=args.start, end=args.end, rf=args.rf)

    s: PerfStats = result["stats"]
    print("\n=== Performance Summary ===")
    print(f"Asset: {result['asset']}")
    start, end = result["period"]
    print(f"Period: {start.date()} to {end.date()} ({s.n_days} trading days)")
    print(f"CAGR:         {_fmt_pct(s.cagr)}")
    print(f"Ann. Vol:     {_fmt_pct(s.ann_vol)}")
    print(f"Sharpe:       {s.sharpe:.2f}" if not pd.isna(s.sharpe) else "Sharpe: NA")
    print(f"Max Drawdown: {_fmt_pct(s.max_dd)}")
    print(f"DD Length:    {s.max_dd_len} days")
    print(f"Calmar:       {s.calmar:.2f}" if not pd.isna(s.calmar) else "Calmar: NA")
    print(f"Win Rate:     {_fmt_pct(s.win_rate)}")
    print(f"Skew/Kurt:    {s.skew:.2f} / {s.kurtosis:.2f}")
    print(f"\nFigures saved to: {result['output_dir'].resolve()}")


# ---------- Streamlit UI ----------
def run_streamlit_app():
    import io
    import streamlit as st

    st.set_page_config(page_title="Financial Time Series Analyzer", layout="wide")
    st.title("📈 Financial Time Series Analyzer")

    with st.sidebar:
        st.header("1) Load data")
        src = st.radio("Data source", ["Upload file", "Path on disk"], horizontal=True)
        sheet = None
        df = None

        if src == "Upload file":
            up = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls", "xlsm"])
            sheet = st.text_input("Excel sheet name (optional)", value="")
            if up is not None:
                try:
                    if up.name.lower().endswith((".xls", ".xlsx", ".xlsm")):
                        df = pd.read_excel(up, sheet_name=sheet or None)
                    else:
                        content = up.read()
                        df = pd.read_csv(io.BytesIO(content))
                except Exception as e:
                    st.error(f"Failed to read file: {e}")
        else:
            path = st.text_input("File path", value="Market Monitor - blp.xlsx")
            sheet = st.text_input("Excel sheet (optional)", value="BBG Data")
            if path:
                try:
                    df = load_prices(path, sheet=sheet or None)
                except Exception as e:
                    st.error(f"Failed to load: {e}")

        st.header("1b) Ticker mapping")
        map_src = st.radio("Mapping source", ["Upload mapping CSV", "Path on disk"], horizontal=True)
        mapping_df = None
        if map_src == "Upload mapping CSV":
            upm = st.file_uploader("Upload TickerNameMapping.csv", type=["csv"], key="mapup")
            if upm is not None:
                try:
                    mapping_df = pd.read_csv(upm)
                    mapping_df = mapping_df.rename(
                        columns={mapping_df.columns[0]: "Name", mapping_df.columns[1]: "Ticker"}
                    )[["Name", "Ticker"]]
                except Exception as e:
                    st.error(f"Failed to read mapping: {e}")
        else:
            map_path = st.text_input("Mapping path", value="TickerNameMapping.csv")
            if map_path:
                try:
                    mapping_df = load_mapping(map_path)
                except Exception as e:
                    st.error(f"Failed to load mapping: {e}")

        if df is not None:
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.set_index("Date").sort_index()
            st.success(f"Loaded shape: {df.shape}")
            st.caption("Preview (first 5 rows):")
            st.dataframe(df.head())

        st.header("2) Natural language")
        question = st.text_area(
            "Ask a question (e.g., 'What's the performance of SP500 from beginning of 2025 to the end of the second quarter?')",
            height=100,
        )
        parse_clicked = st.button("Parse with AI ✨")

        st.header("3) Parameters (you can edit after parsing)")

        if "pending_asset" not in st.session_state:
            st.session_state["pending_asset"] = None
        if "pending_dates" not in st.session_state:
            st.session_state["pending_dates"] = None

        if parse_clicked:
            if mapping_df is None:
                st.warning("Please provide TickerNameMapping.csv before parsing.")
            else:
                # Anchor "today" to the last date in the dataset if available
                ref_date = None
                if df is not None and not df.empty:
                    ref_date = df.index.max().date().isoformat()

                parsed = llm_extract(question, mapping_df, reference_date=ref_date)
                with st.expander("Parsed details"):
                    st.json(parsed)

                st.session_state["pending_asset"] = parsed.get("asset")
                try:
                    s = pd.to_datetime(parsed.get("start")).date() if parsed.get("start") else None
                    e = pd.to_datetime(parsed.get("end")).date() if parsed.get("end") else None
                    st.session_state["pending_dates"] = (s, e) if (s and e) else None
                except Exception:
                    st.session_state["pending_dates"] = None

                st.rerun()

        asset_cols = [c for c in df.columns if c != "Date"] if df is not None else []
        if st.session_state["pending_asset"] in asset_cols:
            default_asset_index = asset_cols.index(st.session_state["pending_asset"])
        else:
            default_asset_index = 0 if asset_cols else None

        if df is not None:
            min_d, max_d = df.index.min().date(), df.index.max().date()
        else:
            min_d, max_d = date(2000, 1, 1), date.today()

        if st.session_state["pending_dates"]:
            dr_default = st.session_state["pending_dates"]
        else:
            dr_default = (min_d, max_d)

        asset = st.selectbox(
            "Asset column (Bloomberg ticker from your price file)",
            options=asset_cols,
            index=default_asset_index,
            key="asset_select",
        )
        dr = st.date_input("Date range", dr_default, key="date_range")
        rf = st.number_input("Risk-free rate (annual)", value=0.0, step=0.001, format="%.4f")

        run = st.button("Run analysis ▶️", type="primary")

    if not run or df is None:
        st.stop()

    start, end = [pd.to_datetime(d) for d in st.session_state["date_range"]]
    result = analyze_asset(df, st.session_state["asset_select"], start=start, end=end, rf=rf)

    s: PerfStats = result["stats"]
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("CAGR", f"{s.cagr*100:,.2f}%")
    k2.metric("Ann. Vol", f"{s.ann_vol*100:,.2f}%")
    k3.metric("Sharpe", f"{s.sharpe:.2f}" if not pd.isna(s.sharpe) else "NA")
    k4.metric("Max DD", f"{s.max_dd*100:,.2f}%")
    k5.metric("DD Length", f"{s.max_dd_len}d")
    k6.metric("Win Rate (ex-zeros)", f"{s.win_rate*100:,.2f}%")

    outdir: Path = result["output_dir"]
    imgs = sorted([p for p in outdir.glob("*.png")])
    st.subheader("Charts")
    if imgs:
        grid = st.columns(2)
        for i, pth in enumerate(imgs):
            with grid[i % 2]:
                st.image(str(pth), use_container_width=True, caption=pth.name)
    else:
        st.info("No charts were generated.")

    st.subheader("Annual Returns")
    st.dataframe(result["annual_returns"].style.format("{:.2%}"))

    st.subheader("Monthly Returns Table")
    st.dataframe(result["monthly_table"].style.format("{:.2%}"))

    st.subheader("Risk (Daily Historical VaR / CVaR)")
    ret = result["daily_returns"]["DailyReturn"].dropna()
    if len(ret) > 0:
        def hist_var_cvar(alpha: float):
            q = ret.quantile(1 - alpha)
            var = -q
            tail = ret[ret <= q]
            cvar = -tail.mean() if len(tail) else np.nan
            return var, cvar
        rows = [{"CL": a, "VaR": hist_var_cvar(a)[0], "CVaR": hist_var_cvar(a)[1]} for a in (0.95, 0.99)]
        st.table(pd.DataFrame(rows).set_index("CL").style.format("{:.2%}"))
    else:
        st.info("Not enough returns to compute VaR/CVaR.")


# ---------------- Entry point ----------------
if __name__ == "__main__":
    if any(m.startswith("streamlit") for m in sys.modules.keys()):
        run_streamlit_app()
    else:
        main_cli()
