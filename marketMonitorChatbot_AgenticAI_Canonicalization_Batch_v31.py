# marketMonitorChatbot_AgenticAI_Canonicalization_Batch_v31.py
# ---------------------------------------------------------------------------------
# Market Monitor Chatbot — Pydantic types, no-cache embeddings, robust asset resolver,
# Query Rewriter with fail-open behavior, routing fallback, range/date alignment,
# and a simple Streamlit UI for normal users (batch runner remains available).
#
#
# v27 changes (relative to v26):
#   • Added tooltips to hover mouse on "Tips" 
#   • Added tooltips to hover mouse on "Tickers" 
# V28 changes (relative to v27)
#   • Tooltip for Tips shouldn’t get cut off on the left.
#   • Keep the original always‑visible Tips list, and show another ~10 example questions on hover.
# V29 changes (relative to v28)
#   • Added a manual calculator panel on the right
# V30 changes (relative to v29)
#   • Restored hover tooltips for Tips and Tickers with safer CSS (no left-edge clipping).
#   • Sidebar: keep always-visible short tips + extra examples on hover over “Tips”.
#   • Header: hovering “Tickers in data” shows a scrollable list of tickers + descriptions.
# V31 changes (relative to v30)
#   • QueryRewriter now runs in two passes (translation + canonicalization) to fix non-English “market” queries.
#   • Rewriter system prompt now includes explicit country-level equity index defaults
#     (US→SPX Index, Canada→SPTSX60 Index, Australia→ASX Index, EAFE→MSCI EAFE Index,
#      Hong Kong→HSI Index, Philippines→PCOMP Index, China→SHCOMP Index).
# ---------------------------------------------------------------------------------
from __future__ import annotations

import os
import re
import json
import argparse
import html as html_mod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal, Any
from datetime import datetime, date

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from openai import OpenAI
from rapidfuzz import process as fuzzprocess, fuzz

# ----------------------- Configuration -----------------------

EXCEL_PATH = os.environ.get("MARKET_MONITOR_XLSX", "Market Monitor - blp.xlsx")
SHEET_BBG = "BBG Data"
SHEET_TICKERS = "Tickers"

# Override via env if desired
DEFAULT_REASONING_MODEL = os.environ.get("OPENAI_REASONING_MODEL", "gpt-4o-mini")
DEFAULT_EMBED_MODEL = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-large")

client = OpenAI()

# ----------------------- Utilities ---------------------------

def _clean(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[’'`]", "", s)
    s = re.sub(r"[_\-:/|]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def _iso(d: date | datetime | pd.Timestamp | None) -> Optional[str]:
    if d is None:
        return None
    if isinstance(d, pd.Timestamp):
        return d.strftime("%Y-%m-%d")
    if isinstance(d, datetime):
        return d.date().isoformat()
    return d.isoformat()  # date

# ----------------------- Data Layer --------------------------

class MarketData:
    """
    Loads Excel and exposes price/return/spread calculations.
    Uses pandas internally, but Pydantic models never contain pandas types.
    """
    def __init__(self, path: str = EXCEL_PATH):
        self.path = path
        self.df_prices: pd.DataFrame = self._load_prices()
        self.df_tickers: pd.DataFrame = self._load_tickers()
        self.ticker_to_name: Dict[str, str] = {
            str(row["Ticker"]).strip(): str(row["Description"]).strip()
            for _, row in self.df_tickers.iterrows()
        }
        self.name_to_ticker: Dict[str, str] = {v: k for k, v in self.ticker_to_name.items()}
        self.available_dates: pd.DatetimeIndex = self.df_prices.index
        self.last_ts: pd.Timestamp = self.available_dates.max()   # pandas Timestamp
        self.last_date: date = self.last_ts.date()                # python date for LLM

    def _load_prices(self) -> pd.DataFrame:
        df = pd.read_excel(self.path, sheet_name=SHEET_BBG, engine="openpyxl")
        first_col = df.columns[0]
        assert _clean(str(first_col)) in {"date", "dates"}, \
            f"First column should be 'date', found: {first_col}"

        df.rename(columns={first_col: "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

        # Coerce numeric columns
        for c in df.columns:
            if c == "date":
                continue
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # Sort ascending for easy slicing; drop fully empty columns
        df = df.sort_values("date").set_index("date")
        df = df.dropna(axis=1, how="all")
        return df

    def _load_tickers(self) -> pd.DataFrame:
        df = pd.read_excel(self.path, sheet_name=SHEET_TICKERS, engine="openpyxl", header=0)
        cols = [_clean(str(c)) for c in df.columns]
        df.rename(columns={df.columns[i]: cols[i] for i in range(len(cols))}, inplace=True)
        # Expect 'ticker' & 'description'; fallback to first two columns if needed
        if "ticker" not in df.columns or "description" not in df.columns:
            df = df.iloc[:, :2]
            df.columns = ["ticker", "description"]
        df["Ticker"] = df["ticker"].astype(str).str.strip()
        df["Description"] = df["description"].astype(str).str.strip()
        return df[["Ticker", "Description"]]

    # ---- helpers ----
    def _as_pydate(self, d: date | datetime | pd.Timestamp) -> date:
        """Normalize to a plain Python date to avoid pd.Timestamp overflow on extreme years."""
        if isinstance(d, pd.Timestamp):
            return d.date()
        if isinstance(d, datetime):
            return d.date()
        return d  # already a date

    def get_series(self, ticker: str) -> pd.Series:
        if ticker not in self.df_prices.columns:
            raise KeyError(f"Ticker '{ticker}' not in price data.")
        # Only keep dates where the asset actually has data
        return self.df_prices[ticker].dropna()

    def nearest_on_or_before(self, sr: pd.Series, d: date | datetime) -> Tuple[pd.Timestamp, float]:
        """
        Return the last observation on or before the given date.
        Bounds-aware: compares using plain Python dates first, then converts safely.
        Raises if the requested date is earlier than the series start.
        """
        idx = sr.index
        first_dt: pd.Timestamp = idx[0]
        last_dt: pd.Timestamp = idx[-1]

        d_date = self._as_pydate(d)

        if d_date < first_dt.date():
            raise ValueError(f"No data available on or before {_iso(d_date)}.")
        if d_date >= last_dt.date():
            return last_dt, float(sr.iloc[-1])

        ts = pd.Timestamp(d_date)  # safe now
        pos = idx.searchsorted(ts, side="right") - 1
        dt = idx[pos]
        return dt, float(sr.iloc[pos])

    def nearest_on_or_before_or_first(self, sr: pd.Series, d: date | datetime) -> Tuple[pd.Timestamp, float]:
        """
        Single-date alignment:
          - If requested date is before the series starts, return earliest available.
          - If requested date is after the series ends, return latest available.
          - Otherwise, return the last observation on or before the requested date.
        """
        idx = sr.index
        first_dt: pd.Timestamp = idx[0]
        last_dt: pd.Timestamp = idx[-1]

        d_date = self._as_pydate(d)

        if d_date <= first_dt.date():
            return first_dt, float(sr.iloc[0])
        if d_date >= last_dt.date():
            return last_dt, float(sr.iloc[-1])

        ts = pd.Timestamp(d_date)  # safe now
        pos = idx.searchsorted(ts, side="right") - 1
        dt = idx[pos]
        return dt, float(sr.iloc[pos])

    def align_range_to_series(
        self, sr: pd.Series, start: date | datetime, end: date | datetime
    ) -> Tuple[pd.Timestamp, float, pd.Timestamp, float]:
        """
        Range alignment (start/end):
          - If requested start is before the first available observation, start = first.
          - If requested end is before the first available observation, end = first (both endpoints become 'first').
          - If requested start/end are after the last available, they snap to last.
          - Otherwise, both endpoints use 'on or before' alignment.
        """
        idx = sr.index
        first_dt: pd.Timestamp = idx[0]
        last_dt: pd.Timestamp = idx[-1]

        s_date = self._as_pydate(start)
        e_date = self._as_pydate(end)

        # Align start
        if s_date <= first_dt.date():
            d0, v0 = first_dt, float(sr.iloc[0])
        elif s_date >= last_dt.date():
            d0, v0 = last_dt, float(sr.iloc[-1])
        else:
            d0, v0 = self.nearest_on_or_before(sr, s_date)

        # Align end
        if e_date <= first_dt.date():
            d1, v1 = first_dt, float(sr.iloc[0])
        elif e_date >= last_dt.date():
            d1, v1 = last_dt, float(sr.iloc[-1])
        else:
            d1, v1 = self.nearest_on_or_before(sr, e_date)

        return d0, v0, d1, v1

    # ---- calculations ----
    def price(self, ticker: str, d: date | datetime) -> Tuple[pd.Timestamp, float]:
        """Bounds-aware single-date price lookup with earliest/latest snapping."""
        sr = self.get_series(ticker)
        return self.nearest_on_or_before_or_first(sr, d)

    def price_change(self, ticker: str, start: date | datetime, end: date | datetime) -> Dict[str, Any]:
        """Bounds-aware range lookup with clamping at both ends."""
        sr = self.get_series(ticker)
        d0, v0, d1, v1 = self.align_range_to_series(sr, start, end)
        pct = float((v1 / v0 - 1) * 100) if v0 not in (None, 0, np.nan) else float("nan")
        return {
            "start_date_actual": d0, "end_date_actual": d1,
            "start_value": float(v0), "end_value": float(v1),
            "absolute_change": float(v1 - v0), "pct_change": pct
        }

    def total_return(self, ticker: str, start: date | datetime, end: date | datetime) -> Dict[str, Any]:
        result = self.price_change(ticker, start, end)
        return {**result, "total_return_pct": result["pct_change"]}

    def compare_spread(self, ticker1: str, ticker2: str, d: date | datetime) -> Dict[str, Any]:
        # Single-date alignment (earliest/latest snapping) for initial dates
        d1, v1 = self.price(ticker1, d)
        d2, v2 = self.price(ticker2, d)
        # Align both assets to the later of the two actual dates to ensure same timestamp
        dt = max(d1, d2)
        d1u, v1u = self.price(ticker1, dt)
        d2u, v2u = self.price(ticker2, dt)
        return {"date_actual": dt, "asset1_value": float(v1u), "asset2_value": float(v2u), "spread": float(v1u - v2u)}

# ----------------------- Asset Resolver ----------------------

# Optional synonyms to enrich candidate text for embeddings
CURRENCY_SYNONYMS = {
    "sterling": ["gbp", "british pound", "pound sterling"],
    "loonie": ["cad", "canadian dollar"],
    "greenback": ["usd", "us dollar", "u.s. dollar"],
    "euro": ["eur", "euro area"],
    "yen": ["jpy", "japanese yen"],
    "aussie": ["aud", "australian dollar"],
    "kiwi": ["nzd", "new zealand dollar"],
    "swissy": ["chf", "swiss franc"],
    "yuan": ["cny", "renminbi", "rmb"],
}
TERM_SYNONYMS = {
    "gov": ["government", "treasury", "sovereign", "gilt"],
    "corp": ["corporate"],
    "note": ["bond"],
    "bill": ["t-bill", "treasury bill"],
    "yield": ["rate", "interest"],
    "index": ["benchmark"],
}

class AssetIndexItem(BaseModel):
    ticker: str
    name: str
    text: str
    embedding: Optional[List[float]] = None

def _enrich_text(ticker: str, name: str) -> str:
    base = _clean(name)
    base = re.sub(r"\b(index|curncy|generic|bid|ask|yield|note|bond)\b", "", base)
    t = _clean(ticker.replace(".", " "))
    parts = {ticker, name, base, t, t.replace(" ", ""), name.lower().replace("&", "and").replace("-", " ")}
    # add synonyms if present in base
    for k, syns in TERM_SYNONYMS.items():
        if k in base:
            parts.update(syns)
    for slang, variants in CURRENCY_SYNONYMS.items():
        if slang in base:
            parts.update(variants)
    return " | ".join([p for p in parts if p])

# ---- custom exception for graceful "asset not found" ----
class AssetResolutionError(Exception):
    def __init__(self, mention: str, suggestions: List[Tuple[str, str, float]]):
        super().__init__(f"Asset not found or not in dataset: {mention}")
        self.mention = mention
        self.suggestions = suggestions  # list[(ticker, name, score)]

class AssetResolver:
    """
    Embedding-based resolver limited to tickers that actually appear in BBG Data.
    No cache: index is rebuilt every run to reflect the current workbook.
    Acceptance is conservative; ambiguous cases go to LLM with hard constraints.
    Also produces per-asset debug info.
    """
    # --- token detectors / maps ---
    _RATING_RE = re.compile(
        r"\b("
        r"bbb[+-]?|bb[+-]?|b[+-]?|"
        r"ccc[+-]?|"
        r"aaa[+-]?|aa[+-]?|a(?:\s|-)?rated|"
        r"ig\b|investment(?:\s|-)?grade|inv(?:\s|-)?grade|"
        r"hy\b|high(?:\s|-)?yield"
        r")\b", re.IGNORECASE
    )
    _TENOR_RE = re.compile(r"\b(?:(\d{1,2})\s*(m|mo|mth|mths|month|months|y|yr|yrs|year|years))\b", re.IGNORECASE)

    _TYPE_MAP = {
        "government": ["government", "gov", "govt", "treasury", "gilt", "bund", "sovereign", "generic govt", "govt bonds"],
        "corporate": ["corporate", "corp"],
        "swap": ["swap", "swaps"],
        "municipal": ["municipal", "muni"],
        "provincial": ["provincial", "province", "prov"],
    }

    _COUNTRY_HINTS = {
        "us": ["us", "u.s", "usa", "united states", "america", "usd", "treasury"],
        "canada": ["canada", "canadian", "cad"],
    }

    def __init__(self, md: MarketData, embed_model: str = DEFAULT_EMBED_MODEL):
        self.md = md
        self.embed_model = embed_model
        self.allowed = tuple(sorted(str(c) for c in self.md.df_prices.columns))
        self.items: List[AssetIndexItem] = self._build_index()
        self.debug_log: List[Dict[str, Any]] = []

    def _cos(self, a, b) -> float:
        """Cosine similarity with numeric stability."""
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _build_index(self) -> List[AssetIndexItem]:
        items: List[AssetIndexItem] = []
        present = set(self.allowed)
        name_map = self.md.ticker_to_name
        for t in present:
            name = name_map.get(t, t)
            items.append(AssetIndexItem(ticker=t, name=name, text=_enrich_text(t, name)))
        if items:
            embeds = client.embeddings.create(model=self.embed_model, input=[x.text for x in items])
            for item, v in zip(items, embeds.data):
                item.embedding = v.embedding
        return items

    def _extract_tenor(self, text: str) -> Optional[str]:
        m = self._TENOR_RE.search(text or "")
        if not m:
            return None
        n, unit = m.group(1), m.group(2).lower()
        return f"{int(n)}M" if unit.startswith("m") else f"{int(n)}Y"

    def _mention_tokens(self, text: str) -> Dict[str, Any]:
        t = (text or "").lower()
        rating = None
        m = self._RATING_RE.search(t)
        if m:
            rating = m.group(0).lower().replace(" ", "")
        tenor = self._extract_tenor(t)
        a_type = None
        for k, vals in self._TYPE_MAP.items():
            if any(v in t for v in vals):
                a_type = k
                break
        country = None
        for k, vals in self._COUNTRY_HINTS.items():
            if any(v in t for v in vals):
                country = k
                break
        return {"rating": rating, "tenor": tenor, "type": a_type, "country": country}

    def _classify_candidate(self, label: str) -> Dict[str, Any]:
        lbl = (label or "").lower()
        country = None
        if any(tok in lbl for tok in self._COUNTRY_HINTS["canada"]):
            country = "canada"
        elif any(tok in lbl for tok in self._COUNTRY_HINTS["us"]):
            country = "us"
        subtype = None
        if "corporate" in lbl:
            subtype = "corporate"
        elif "provinc" in lbl or "province" in lbl:
            subtype = "provincial"
        elif "municip" in lbl:
            subtype = "municipal"
        elif "swap" in lbl:
            subtype = "swap"
        elif any(w in lbl for w in self._TYPE_MAP["government"]):
            subtype = "government"

        def _has_rating_token(rtoken: str) -> bool:
            rtoken = rtoken.lower()
            if rtoken.startswith("bbb"): return "bbb" in lbl
            if rtoken.startswith("aaa"): return "aaa" in lbl
            if rtoken.startswith("aa"):  return " aa" in f" {lbl} "
            if rtoken.startswith("ig") or "investment" in rtoken: return (" ig " in f" {lbl} ") or ("investment grade" in lbl)
            if rtoken.startswith("hy") or "high" in rtoken: return (" hy " in f" {lbl} ") or ("high yield" in lbl)
            if "a-rated" in rtoken: return (" a " in f" {lbl} ") or (" a-" in lbl) or (" a+" in lbl)
            if rtoken.startswith("b") and "bbb" not in rtoken: return " b " in f" {lbl} "
            return False

        def _has_tenor_token(ten: Optional[str]) -> bool:
            if not ten:
                return True
            if ten.endswith("Y"):
                n = ten[:-1]
                return any(s in lbl for s in [
                    f"{n}y", f"{n} yr", f"{n} yrs", f"{n} year", f"{n} years"
                ])
            n = ten[:-1]
            return any(s in lbl for s in [
                f"{n}m", f"{n} mth", f"{n} mths", f"{n} mo", f"{n} month", f"{n} months"
            ])

        return {"country": country, "subtype": subtype, "has_rating": _has_rating_token, "has_tenor": _has_tenor_token}

    def _bonus_for_candidate(self, cand_label: str, mention_tokens: Dict[str, Any]) -> float:
        lbl = cand_label.lower()
        bonus = 0.0
        r = mention_tokens.get("rating")
        if r:
            if "bbb" in r and "bbb" in lbl: bonus += 0.12
            elif "aaa" in r and "aaa" in lbl: bonus += 0.12
            elif "aa" in r and " aa" in f" {lbl} ": bonus += 0.10
            elif "hy" in r and ("hy" in lbl or "high yield" in lbl): bonus += 0.10
            elif "ig" in r and (" ig " in f" {lbl} " or "investment grade" in lbl): bonus += 0.08
            elif "a-rated" in r and (" a " in f" {lbl} " or " a-" in lbl or " a+" in lbl): bonus += 0.06
        ten = mention_tokens.get("tenor")
        if ten and (ten[:-1] + ("y" if ten.endswith("Y") else "m")) in lbl.replace(" ", ""):
            bonus += 0.08
        tp = mention_tokens.get("type")
        if tp:
            if tp == "government" and any(w in lbl for w in self._TYPE_MAP["government"]): bonus += 0.10
            if tp == "corporate"  and "corporate" in lbl: bonus += 0.10
            if tp == "swap"       and "swap" in lbl: bonus += 0.08
            if tp == "municipal"  and "municip" in lbl: bonus += 0.07
            if tp == "provincial" and "provinc" in lbl: bonus += 0.07
        ctry = mention_tokens.get("country")
        if ctry:
            if ctry == "us" and (("us " in f"{lbl} ") or (" united states" in lbl) or (" treasury" in lbl) or (" usd" in lbl)): bonus += 0.07
            if ctry == "canada" and ("canada" in lbl or "canadian" in lbl or " cad" in lbl): bonus += 0.07
        return min(bonus, 0.22)

    def _penalty_for_mismatch(self, cand_label: str, mention_tokens: Dict[str, Any]) -> float:
        info = self._classify_candidate(cand_label)
        lbl = cand_label.lower()
        pen = 0.0
        mc = mention_tokens.get("country")
        if mc and info["country"] and info["country"] != mc:
            pen -= 0.18
        mt = mention_tokens.get("type")
        if mt:
            desired = mt
            actual = info["subtype"]
            if desired == "government" and actual not in (None, "government"):
                pen -= 0.16
            elif desired == "corporate" and actual != "corporate":
                pen -= 0.16
            elif desired == "provincial" and actual != "provincial":
                pen -= 0.14
            elif desired == "municipal" and actual != "municipal":
                pen -= 0.14
        mr = mention_tokens.get("rating")
        if mr and not info["has_rating"](mention_tokens["rating"]):
            pen -= 0.12
        ten = mention_tokens.get("tenor")
        if ten and not info["has_tenor"](ten):
            pen -= 0.10
        return pen

    def suggest(self, mention: str, k: int = 10) -> List[Tuple[str, str, float]]:
        if not self.items:
            return []
        if not (mention or "").strip():
            # Can't embed empty input — return empty
            return []

        mention_tokens = self._mention_tokens(mention)

        # Embedding similarity to ALL items
        q_embed = client.embeddings.create(model=self.embed_model, input=[mention]).data[0].embedding
        sim_map: Dict[str, float] = {}
        for it in self.items:
            if it.embedding is None:
                continue
            sim_map[it.ticker] = self._cos(q_embed, it.embedding)

        # Fuzzy over ALL labels
        labels = [f"{it.ticker} || {it.name}" for it in self.items]
        label_to_item = {f"{it.ticker} || {it.name}": it for it in self.items}
        fuzz_hits = fuzzprocess.extract(mention, labels, scorer=fuzz.WRatio, limit=max(50, k))

        def blend(embed: float, fuzzy_score: float) -> float:
            return 0.85 * float(embed) + 0.15 * float(fuzzy_score) / 100.0

        combined: Dict[str, Tuple[str, str, float]] = {}

        for it in self.items:
            base = sim_map.get(it.ticker, 0.0)
            s = blend(base, 0.0)
            label = f"{it.ticker} {it.name}"
            s += self._bonus_for_candidate(label, mention_tokens)
            s += self._penalty_for_mismatch(label, mention_tokens)
            combined[it.ticker] = (it.ticker, it.name, s)

        for label, fuzzy_score, _ in fuzz_hits:
            it = label_to_item[label]
            base = sim_map.get(it.ticker, 0.0)
            s = blend(base, fuzzy_score)
            cand_label = f"{it.ticker} {it.name}"
            s += self._bonus_for_candidate(cand_label, mention_tokens)
            s += self._penalty_for_mismatch(cand_label, mention_tokens)
            prev = combined.get(it.ticker)
            if (not prev) or s > prev[2]:
                combined[it.ticker] = (it.ticker, it.name, s)

        ranked = sorted(combined.values(), key=lambda x: x[2], reverse=True)
        return ranked[:k]

    def _hard_filter(self, ranked: List[Tuple[str, str, float]], mention_tokens: Dict[str, Any]) -> List[Tuple[str, str, float]]:
        """Require matches for explicit constraints if present."""
        need_country = bool(mention_tokens.get("country"))
        need_type    = bool(mention_tokens.get("type"))
        need_rating  = bool(mention_tokens.get("rating"))
        need_tenor   = bool(mention_tokens.get("tenor"))

        if not (need_country or need_type or need_rating or need_tenor):
            return ranked

        def ok(ticker: str, name: str) -> bool:
            lbl = f"{ticker} {name}"
            info = self._classify_candidate(lbl)
            if need_country:
                if not info["country"] or info["country"] != mention_tokens["country"]:
                    return False
            if need_type:
                mt = mention_tokens["type"]
                if mt == "government":
                    if info["subtype"] != "government":
                        return False
                else:
                    if info["subtype"] != mt:
                        return False
            if need_rating and not info["has_rating"](mention_tokens["rating"]):
                return False
            if need_tenor and not info["has_tenor"](mention_tokens["tenor"]):
                return False
            return True

        filtered = [(t, n, s) for (t, n, s) in ranked if ok(t, n)]
        return filtered if filtered else []

    def _can_fast_accept(self, ranked: List[Tuple[str, str, float]], mention_tokens: Dict[str, Any]) -> bool:
        if not ranked:
            return False
        if any(mention_tokens.get(k) for k in ("rating", "tenor", "type", "country")):
            return False
        top = ranked[0][2]
        second = ranked[1][2] if len(ranked) > 1 else -1.0
        if top >= 0.58:
            return True
        if top >= 0.40 and (top - second) >= 0.15:
            return True
        return False

    def _defaults_to_tokens(self, fi_defaults: Dict[str, Any]) -> Dict[str, Optional[str]]:
        """
        Convert rewriter fixed-income defaults into the same token schema as _mention_tokens.
        We deliberately ignore any synthetic default rating; ratings should come only from the user's text.
        """
        out: Dict[str, Optional[str]] = {"rating": None, "tenor": None, "type": None, "country": None}
        if not fi_defaults:
            return out

        # Country
        country = (fi_defaults.get("country") or "").strip().lower()
        if country:
            if any(k in country for k in ["us", "united states", "america", "usa"]):
                out["country"] = "us"
            elif "canada" in country or "cad" in country:
                out["country"] = "canada"

        # Tenor (expect formats like '10Y', '3M', '2y', '6m')
        tenor = (fi_defaults.get("tenor") or "").strip().upper()
        if tenor:
            if tenor.endswith(("Y", "M")) and len(tenor) >= 2 and tenor[:-1].isdigit():
                out["tenor"] = tenor

        # Asset type -> type
        a_type = (fi_defaults.get("asset_type") or "").strip().lower()
        if a_type:
            if any(w in a_type for w in ["government", "gov", "treasury", "sovereign"]):
                out["type"] = "government"
            elif "corporate" in a_type or "corp" in a_type:
                out["type"] = "corporate"
            elif "swap" in a_type:
                out["type"] = "swap"
            elif "municip" in a_type:
                out["type"] = "municipal"
            elif "provinc" in a_type or "province" in a_type:
                out["type"] = "provincial"

        # Note: we intentionally do NOT propagate rating from defaults
        return out

    def resolve(
        self,
        mention: str,
        fi_defaults: Optional[Dict[str, Any]] = None,
        asset_class: Optional[str] = None,
    ) -> Tuple[str, str, float, List[Tuple[str, str, float]]]:
        mention = (mention or "").strip()
        if not mention:
            raise AssetResolutionError(mention, [])

        mention_tokens_raw = self._mention_tokens(mention)
        mention_tokens = dict(mention_tokens_raw)

        tokens_defaults = None
        if fi_defaults and (asset_class or "").lower() == "fixed_income":
            tokens_defaults = self._defaults_to_tokens(fi_defaults)
            # Backfill missing tokens from defaults, but never override explicit user text
            for key in ("country", "type", "tenor", "rating"):
                if not mention_tokens.get(key) and tokens_defaults.get(key):
                    mention_tokens[key] = tokens_defaults[key]

        ranked_all = self.suggest(mention, k=16)

        if not ranked_all:
            raise AssetResolutionError(mention, [])

        debug_entry: Dict[str, Any] = {
            "mention": mention,
            "tokens": mention_tokens,
            "tokens_raw": mention_tokens_raw,
            "tokens_defaults": tokens_defaults,
            "candidates_top3": [{"ticker": t, "name": n, "score": round(s, 4)} for t, n, s in ranked_all[:3]],
            "accept_path": None,
            "tie_breaker_prompt": None,
        }

        ranked = self._hard_filter(ranked_all, mention_tokens)
        if ranked:
            debug_entry["postfilter_top3"] = [{"ticker": t, "name": n, "score": round(s, 4)} for t, n, s in ranked[:3]]
        else:
            if any(mention_tokens.get(k) for k in ("country", "type", "rating", "tenor")):
                debug_entry["accept_path"] = "hard_filter_empty"
                self.debug_log.append(debug_entry)
                raise AssetResolutionError(mention, ranked_all)
            ranked = ranked_all

        if self._can_fast_accept(ranked, mention_tokens):
            t0, n0, s0 = ranked[0]
            debug_entry["accept_path"] = "fast_accept"
            debug_entry["accepted"] = {"ticker": t0, "name": n0, "score": round(s0, 4)}
            self.debug_log.append(debug_entry)
            return t0, n0, float(s0), ranked

        top = ranked[0][2]
        if top <= 0.35:
            debug_entry["accept_path"] = "plausibility_reject"
            self.debug_log.append(debug_entry)
            raise AssetResolutionError(mention, ranked)

        opts = [{"ticker": t, "name": n, "score": round(s, 4)} for t, n, s in ranked[:8]]
        system = (
            "Pick the single best candidate from the list matching the user's asset mention. "
            "If none are plausible, reply with {'ticker':'__unknown__','name':'unknown'}.\n"
            "HARD CONSTRAINTS (must hold if the user's mention includes them):\n"
            "  • Country (e.g., US/USA/United States; Canada) MUST match.\n"
            "  • Type (government/treasury/sovereign vs corporate vs swap vs municipal vs provincial) MUST match.\n"
            "  • Rating token (AAA, AA, A, BBB, IG, HY, ±) MUST match when present.\n"
            "  • Tenor (3M/6M/1Y/2Y/5Y/10Y/20Y/30Y) should match; if no exact tenor match exists but others do, prefer the closest tenor.\n"
            "Respond ONLY JSON (json): {ticker, name}."
        )
        user = f"Mention: {mention}\nCandidates: {json.dumps(opts, ensure_ascii=False)}"
        debug_entry["tie_breaker_prompt"] = {"system": system, "user": user}

        resp = client.chat.completions.create(
            model=DEFAULT_REASONING_MODEL, temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        try:
            picked = json.loads(resp.choices[0].message.content)
            if picked.get("ticker") == "__unknown__":
                debug_entry["accept_path"] = "tie_breaker_unknown"
                self.debug_log.append(debug_entry)
                raise AssetResolutionError(mention, ranked)
            match = next((x for x in ranked if x[0] == picked.get("ticker")), ranked[0])
        except Exception:
            match = ranked[0]

        t, n, s = match
        debug_entry["accept_path"] = "tie_breaker"
        debug_entry["accepted"] = {"ticker": t, "name": n, "score": round(float(s), 4)}
        self.debug_log.append(debug_entry)
        return t, n, float(s), ranked

# ----------------------- Pydantic Schemas --------------------

class PriceParams(BaseModel):
    asset: str
    date: date
    raw_date_text: Optional[str] = None

class PriceChangeParams(BaseModel):
    asset: str
    start_date: date
    end_date: date
    raw_range_text: Optional[str] = None

class ReturnParams(BaseModel):
    asset: str
    start_date: date
    end_date: date
    raw_range_text: Optional[str] = None

class ComparisonParams(BaseModel):
    asset1: str
    asset2: str
    date: date
    raw_date_text: Optional[str] = None

class RouteDecision(BaseModel):
    function: Literal["price", "price_change", "return", "comparison", "none"]
    confidence: float = Field(ge=0, le=1)
    reason: Optional[str] = None

# ----------------------- Query Rewriter ----------------------

class RewriteDecision(BaseModel):
    final_query: str
    function: Optional[Literal["price", "price_change", "return", "comparison"]] = None
    asset_class: Optional[Literal["fixed_income", "equity", "fx", "commodity", "other"]] = None
    asset_mention: Optional[str] = None
    asset1_mention: Optional[str] = None
    asset2_mention: Optional[str] = None
    date: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    used_defaults: Optional[Dict[str, str]] = None
    assumptions: Optional[List[str]] = None
    reject_reason: Optional[str] = None  # only when function=None due to out-of-scope or parsing failure

REWRITER_SYSTEM_PROMPT = """
rewriter_spec:
  name: "Market Monitor Query Rewriter"
  goal: >
    Convert ambiguous natural-language questions into fully specified, machine-actionable instructions
    for a downstream function-calling agent operating on an Excel workbook.
  reference_date: "{reference_date}"
  acceptance_gate: |
    You may set "function":null ONLY if the user's original prompt cannot be satisfied by EXACTLY one of:
      - "price" (asset, date)
      - "price_change" (asset, start_date, end_date)
      - "return" (asset, start_date, end_date)
      - "comparison" (asset1, asset2, date)
    Examples that SHOULD BE ACCEPTED (choose a function, do not decline):
      - "Spread between US 10Y swap rate and Treasury rate" -> comparison
      - "Compare US 2Y vs 10Y government yields" -> comparison
      - "How's the market doing?" -> price_change on SPX Index (most recent day)
      - "How's Canadian bond market doing?" -> price_change on Canadian interest rate (most recent day)    
      - "过去一周市场表现如何？" (Chinese for "How did the market do over the past week?") -> price_change on SPX Index over the past week
    Reasonable declines (set function to null and provide reject_reason):
      - "Interpolate US 10Y and 20Y swap to compute 11Y" (interpolation)
      - "Build a hedged portfolio targeting DV01=0" (hedging/portfolio)
      - "Forecast next month / run regression / VaR / correlations" (out-of-scope analytics)
      - "What is the standard deviation / volatility / variance of returns?" (risk/dispersion statistics are out of scope)
      - "Make a plot / chart / graph of prices or returns" (visualizations are out of scope; only numeric summaries are supported)
  MUST: |
    The final_query you output MUST be a fully specified, standardized instruction that reflects the chosen function
    and the canonical asset mention(s). Do NOT echo the user's vague wording. Always restate using the explicit
    asset_mention / asset1_mention / asset2_mention and the function.
  output_contract: |
    Return ONLY JSON (json) with this schema:
      {{
        "final_query": <string>,
        "function": "price" | "price_change" | "return" | "comparison" | null,
        "asset_class": "fixed_income" | "equity" | "fx" | "commodity" | "other" | null,
        "asset_mention": <string or null>,
        "asset1_mention": <string or null>,
        "asset2_mention": <string or null>,
        "date": <YYYY-MM-DD or null>,
        "start_date": <YYYY-MM-DD or null>,
        "end_date": <YYYY-MM-DD or null>,
        "used_defaults": {{"country": <string>, "tenor": <string>, "asset_type": <string>, "rating": <string>}} | null,
        "assumptions": [<strings>] | null,
        "reject_reason": <string or null>
      }}
  steps:
    - id: step1
      title: "Asset Class Classification"
      instructions: |
        Decide whether the user is asking about fixed income or other asset classes.
        Fixed income includes: government interest rates, corporate bond yields, swap rates, SOFR, LIBOR,
        all types of bonds (government bonds, corporate bonds, municipal/provincial bonds), MBS, ABS, credit spreads,
        swap spreads, cross-currency basis, etc.
        If the user uses a general term like "market", interpret as equities unless they explicitly indicate otherwise.
    - id: step2
      title: "Determine and Specify the Asset Name"
      instructions: |
        For FIXED INCOME, fill missing properties with defaults:
          • country=USA, tenor=10Y, asset_type=government rate.
        Do NOT assume a credit rating unless the user explicitly specifies one.
        Default overall FI asset: USA 10Y government rate.
        For example, when user asks "How's the CA bond market doing", fill missing properties with: country=Canada, tenor=10Y, asset_type=government rate.

        For NON-FI EQUITY "market level" questions (e.g., "how is the market doing?", "US equity market",
        "Chinese stock market", "过去一周市场表现如何？"):
          • Always collapse the vague wording into a SINGLE canonical equity index and write that index code
            explicitly in asset_mention or asset1_mention.
          • Use the following default mapping by country/region (emit these index tickers exactly as written):
              - US / USA / United States / "the market" with no other region specified → "SPX Index"
              - Canada / Canadian → "SPTSX60 Index"
              - Australia → "ASX Index"
              - EAFE → "MSCI EAFE Index"
              - Hong Kong / HK / Hang Seng → "HSI Index"
              - Philippines → "PCOMP Index"
              - China / Mainland China / Shanghai → "SHCOMP Index"
          • If the user says only "the market" with no region, treat it as the US equity market and use "SPX Index".

        For other equity queries where the user names a specific index, ETF, or stock
        (e.g., "MSFT", "SPTSX60 Index", "HSI Index"), keep that name as the asset_mention and do NOT
        replace it with a default.

        The asset_mention (or asset1_mention/asset2_mention) you output MUST never be a vague phrase like
        "US equity market" – it must be a concrete, tradeable asset name (for example, "SPX Index").
    - id: step3
      title: "Standardize the Function and Time"
      instructions: |
        'How much...' -> price.  'How is ... doing...' -> price_change.
        If no single date is provided, use {reference_date}.
        If performance is requested without an explicit range, interpret as the most recent day (previous trading day to {reference_date} through {reference_date}).
        Ensure date fields are explicit ISO dates.
        Do NOT repurpose requests for standard deviation, volatility, variance, or other risk statistics into a simple
        "return" or "price_change" query. Those must be declined (function=null) as described in the acceptance_gate.
        Requests to "plot", "chart", or "graph" data are also out of scope and must be declined.
    - id: step4
      title: "Propagate Fixed-Income Properties in Comparisons"
      instructions: |
        Ensure both assets have country, tenor, asset_type, and rating (if credit).
        Propagate missing properties from one mention to the other where omitted.
    - id: step5
      title: "Financial Jargon Replacement"
      instructions: |
        Normalize slang: sterling/pound/cable->GBP; loonie->CAD; euro/fiber->EUR; yen->JPY; greenback/dollar->USD;
        aussie->AUD; kiwi->NZD; swissy->CHF; yuan->CNY; gilt->UK government bond; JGB->Japanese government bond;
        Bund->German government bond; basis->cross-currency basis spread; note/bill->government bond.
    - id: step6
      title: "Final Output Requirements"
      instructions: |
        Produce a final rewritten instruction with: fully specified asset(s), standardized function,
        and explicit date(s). The final_query MUST restate the function over the canonical asset names.
"""

class QueryRewriter:
    def __init__(self, md: MarketData, model: str = DEFAULT_REASONING_MODEL):
        self.md = md
        self.model = model

    def _rewrite_once(self, user_prompt: str, ref: str) -> dict:
        """
        Single LLM call for rewriting. Used twice in cascade:
          1) on the original user prompt (possibly non-English / very vague)
          2) on the first pass's final_query (normalized English)
        """
        sys = REWRITER_SYSTEM_PROMPT.format(reference_date=ref)
        resp = client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": user_prompt},
            ],
        )
        data = json.loads(resp.choices[0].message.content)

        # normalize function
        fn = data.get("function")
        if fn not in {"price", "price_change", "return", "comparison", None}:
            fn = None
        data["function"] = fn

        # Ensure we always have a final_query for this pass
        if not data.get("final_query"):
            data["final_query"] = user_prompt

        return data

    def rewrite(self, user_prompt: str) -> RewriteDecision:
        ref = self.md.last_date.isoformat()
        try:
            # ---------- First pass: translation / coarse structuring ----------
            data1 = self._rewrite_once(user_prompt, ref)
            fq1 = data1.get("final_query") or user_prompt

            # ---------- Second pass: run again on the normalized query ----------
            # This is what lets:
            #   "过去一周市场表现如何？"
            # → "Get the price change for the US equity market ..."
            # → "Get the price change for SPX Index ..."
            if fq1 != user_prompt:
                try:
                    data2 = self._rewrite_once(fq1, ref)
                except Exception:
                    data2 = data1  # fall back to first pass if second fails

                # Merge: prefer second-pass fields, fall back to first-pass if missing
                data = dict(data2)
                for key in (
                    "asset_class",
                    "asset_mention",
                    "asset1_mention",
                    "asset2_mention",
                    "date",
                    "start_date",
                    "end_date",
                    "used_defaults",
                    "assumptions",
                    "reject_reason",
                ):
                    if data.get(key) is None and key in data1:
                        data[key] = data1[key]
            else:
                # If first pass is already fully specified (e.g. pure English query),
                # we just use that.
                data = data1

            fn = data.get("function")

            # ---------- Hard guard for unsupported analytics / plotting ----------
            text_for_gate = f"{user_prompt} {data.get('final_query', '')}".lower()
            unsupported_triggers = ["standard deviation", "std dev", "stdev", "volatility", "variance"]
            plotting_triggers = ["plot ", "plot of", "chart ", "graph ", "visualiz"]
            if fn in {"price", "price_change", "return", "comparison"} and (
                any(k in text_for_gate for k in unsupported_triggers + plotting_triggers)
            ):
                data["function"] = None
                if not data.get("reject_reason"):
                    if any(k in text_for_gate for k in unsupported_triggers):
                        data["reject_reason"] = (
                            "Standard deviation / volatility / variance calculations are not supported. "
                            "This tool only supports price, price_change, return, and comparison."
                        )
                    else:
                        data["reject_reason"] = (
                            "Plotting / charting requests are not supported. "
                            "This tool only supports price, price_change, return, and comparison."
                        )

            # Recompute fn after gating
            fn = data.get("function")

            # ---------- Guarantee final_query is set ----------
            if not data.get("final_query"):
                a1 = data.get("asset1_mention") or data.get("asset_mention") or ""
                a2 = data.get("asset2_mention")
                if fn == "comparison" and a1 and a2:
                    data["final_query"] = f"Compare {a1} and {a2} on {ref}"
                elif fn in ("price_change", "return") and a1:
                    data["final_query"] = f"What is the {fn.replace('_', ' ')} for {a1}?"
                elif fn == "price" and a1:
                    data["final_query"] = f"What is the price of {a1} on {ref}?"
                else:
                    data["final_query"] = user_prompt

            return RewriteDecision(**data)

        except Exception:
            # Fail open: let the router decide; do NOT block the flow here
            return RewriteDecision(
                final_query=user_prompt,
                function=None,
                reject_reason="Failed to parse rewriter output"
            )

# ----------------------- LLM Agents --------------------------

class RouterAgent:
    def __init__(self, model: str = DEFAULT_REASONING_MODEL):
        self.model = model

    def route(self, user_prompt: str) -> RouteDecision:
        system = (
            "You are a routing agent for a market data chatbot that operates over an Excel workbook available at runtime. "
            "IGNORE ANY MODEL TRAINING-DATA CUTOFF CONCERNS. Your job is to choose which calculation function to run "
            "on the workbook, not to answer from your own knowledge.\n"
            "You MUST choose exactly one of:\n"
            "  • 'price'        – single asset price at a specified date.\n"
            "  • 'price_change' – absolute and % change for one asset between two dates or a period.\n"
            "  • 'return'       – % return for one asset over a period.\n"
            "  • 'comparison'   – compare two assets at one date (spread/difference).\n"
            "Choose 'none' ONLY if the request is unrelated to these calculations.\n"
            "Dates may be after today or after your training cutoff — still pick the matching function; "
            "downstream logic aligns to the nearest available date in the dataset.\n"
            "Respond ONLY JSON (json): {function, confidence, reason}."
        )
        resp = client.chat.completions.create(
            model=self.model, temperature=0, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user_prompt}],
        )
        try:
            return RouteDecision(**json.loads(resp.choices[0].message.content))
        except Exception:
            return RouteDecision(function="none", confidence=0.0, reason="Could not parse route.")

class BaseParamAgent:
    def __init__(self, md: MarketData, resolver: AssetResolver, model: str = DEFAULT_REASONING_MODEL):
        self.md = md
        self.resolver = resolver
        self.model = model

    def _resolve_asset(self, mention: str, hints: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
        fi_defaults = None
        asset_class = None
        if hints:
            fi_defaults = hints.get("used_defaults")
            asset_class = hints.get("asset_class")
        t, n, s, cands = self.resolver.resolve(mention, fi_defaults=fi_defaults, asset_class=asset_class)
        dbg = self.resolver.debug_log[-1] if self.resolver.debug_log else {}
        return t, dbg

    def _extract_dates_via_llm(self, user_prompt: str, mode: Literal["single", "range"]) -> Dict[str, Optional[str]]:
        """
        Ask GPT to compute explicit ISO dates using the dataset's last date as 'today'.
        Returns ISO strings (never clamp here—alignment is handled in MarketData).
        """
        ref = self.md.last_date
        ref_str = ref.isoformat()
        if mode == "single":
            system = (
                "Extract a single calendar date relevant for a 'price at date' or 'spread at date' query. "
                f"Use reference_date={ref_str} as 'today' (latest in dataset). "
                "Always return ISO date (YYYY-MM-DD). "
                "If the user supplies a date, return it unchanged; if missing, return the reference_date. "
                "Respond ONLY JSON (json): {date: 'YYYY-MM-DD', raw_date_text: <string or null>}."
            )
        else:
            system = (
                "Extract start and end dates for a performance/return period. "
                f"Use reference_date={ref_str} as 'today' (latest in dataset). "
                "Interpret finance phrases yourself (YTD/QTD/MTD/WTD, 1D/5D/1W/1M/3M/6M/1Y/3Y/5Y/10Y, 'from X to Y', 'since X'). "
                "If asked for 'most recent day', construct the period from the trading day immediately before reference_date through reference_date. "
                "Ensure start_date <= end_date. Return explicit dates (unchanged if out of range). "
                "Respond ONLY JSON (json): {start_date:'YYYY-MM-DD', end_date:'YYYY-MM-DD', raw_range_text:<string or null>}."
            )
        resp = client.chat.completions.create(
            model=self.model, temperature=0, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user_prompt}],
        )
        return json.loads(resp.choices[0].message.content)

    # helper to adjust 1D if start==end, using series previous available date
    def _fix_one_day_if_equal(self, ticker: str, start_iso: str, end_iso: str) -> Tuple[str, str]:
        if not start_iso or not end_iso:
            return start_iso, end_iso
        if start_iso != end_iso:
            return start_iso, end_iso
        # same day — choose previous available trading day for that asset
        sr = self.md.get_series(ticker)
        end_ts = pd.Timestamp(end_iso)
        idx = sr.index
        # find last index strictly less than end_ts; if none, keep as-is
        pos = idx.searchsorted(end_ts, side="left") - 1
        if pos >= 0:
            prev_dt = idx[pos].date().isoformat()
            return prev_dt, end_iso
        return start_iso, end_iso

class PriceParamAgent(BaseParamAgent):
    def extract(self, user_prompt: str, hints: Optional[Dict[str, Any]] = None) -> Tuple[PriceParams, Dict[str, Any]]:
        mention_hint = (hints or {}).get("asset_mention") if hints else None
        date_hint = (hints or {}).get("date") if hints else None

        param_debug = {"param_agent_input_prompt": user_prompt, "hints": hints}

        if not mention_hint:
            system = (
                "Extract the asset mention for a 'price' question.\n"
                "Return ONLY JSON (json): {asset_mention: <string>, date: 'YYYY-MM-DD'}.\n"
                "If fixed income, include country/asset class/rating/tenor mentioned by the user."
            )
            resp = client.chat.completions.create(
                model=self.model, temperature=0, response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user_prompt}],
            )
            data = json.loads(resp.choices[0].message.content)
            mention = data.get("asset_mention", user_prompt)
        else:
            mention = mention_hint

        ticker, res_dbg = self._resolve_asset(mention, hints)

        if date_hint:
            date_str = date_hint
            raw_dt = None
        else:
            d = self._extract_dates_via_llm(user_prompt, mode="single")
            date_str, raw_dt = d.get("date") or self.md.last_date.isoformat(), d.get("raw_date_text")

        param_debug["resolver_debug"] = res_dbg
        return PriceParams(asset=ticker, date=datetime.fromisoformat(date_str).date(), raw_date_text=raw_dt), param_debug

class PriceChangeParamAgent(BaseParamAgent):
    def extract(self, user_prompt: str, hints: Optional[Dict[str, Any]] = None) -> Tuple[PriceChangeParams, Dict[str, Any]]:
        mention_hint = (hints or {}).get("asset_mention") if hints else None
        start_hint = (hints or {}).get("start_date") if hints else None
        end_hint = (hints or {}).get("end_date") if hints else None

        param_debug = {"param_agent_input_prompt": user_prompt, "hints": hints}

        if not mention_hint:
            system = (
                "Extract the asset mention for a 'price_change' question.\n"
                "Return ONLY JSON (json): {asset_mention: <string>}.\n"
                "If fixed income, include country/asset class/rating/tenor mentioned by the user."
            )
            resp = client.chat.completions.create(
                model=self.model, temperature=0, response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user_prompt}],
            )
            data = json.loads(resp.choices[0].message.content)
            mention = data.get("asset_mention", user_prompt)
        else:
            mention = mention_hint

        ticker, res_dbg = self._resolve_asset(mention, hints)

        if start_hint and end_hint:
            s, e, raw_rt = start_hint, end_hint, None
        else:
            dr = self._extract_dates_via_llm(user_prompt, mode="range")
            s, e, raw_rt = dr.get("start_date"), dr.get("end_date"), dr.get("raw_range_text")
            if not s or not e:
                ref = self.md.last_date
                s, e = date(ref.year, 1, 1).isoformat(), ref.isoformat()

        # Fix 1D same-day issue by shifting start to prior available for the asset
        s_fixed, e_fixed = self._fix_one_day_if_equal(ticker, s, e)

        param_debug["resolver_debug"] = res_dbg
        return (
            PriceChangeParams(
                asset=ticker,
                start_date=datetime.fromisoformat(s_fixed).date(),
                end_date=datetime.fromisoformat(e_fixed).date(),
                raw_range_text=raw_rt,
            ),
            param_debug
        )

class ReturnParamAgent(BaseParamAgent):
    def extract(self, user_prompt: str, hints: Optional[Dict[str, Any]] = None) -> Tuple[ReturnParams, Dict[str, Any]]:
        mention_hint = (hints or {}).get("asset_mention") if hints else None
        start_hint = (hints or {}).get("start_date") if hints else None
        end_hint = (hints or {}).get("end_date") if hints else None

        param_debug = {"param_agent_input_prompt": user_prompt, "hints": hints}

        if not mention_hint:
            system = (
                "Extract the asset mention for a 'return' question.\n"
                "Return ONLY JSON (json): {asset_mention: <string>}.\n"
                "If fixed income, include country/asset class/rating/tenor mentioned by the user."
            )
            resp = client.chat.completions.create(
                model=self.model, temperature=0, response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user_prompt}],
            )
            data = json.loads(resp.choices[0].message.content)
            mention = data.get("asset_mention", user_prompt)
        else:
            mention = mention_hint

        ticker, res_dbg = self._resolve_asset(mention, hints)

        if start_hint and end_hint:
            s, e, raw_rt = start_hint, end_hint, None
        else:
            dr = self._extract_dates_via_llm(user_prompt, mode="range")
            s, e, raw_rt = dr.get("start_date"), dr.get("end_date"), dr.get("raw_range_text")
            if not s or not e:
                ref = self.md.last_date
                s, e = date(ref.year, 1, 1).isoformat(), ref.isoformat()

        # Fix 1D same-day issue
        s_fixed, e_fixed = self._fix_one_day_if_equal(ticker, s, e)

        param_debug["resolver_debug"] = res_dbg
        return (
            ReturnParams(
                asset=ticker,
                start_date=datetime.fromisoformat(s_fixed).date(),
                end_date=datetime.fromisoformat(e_fixed).date(),
                raw_range_text=raw_rt,
            ),
            param_debug
        )

class ComparisonParamAgent(BaseParamAgent):
    def extract(self, user_prompt: str, hints: Optional[Dict[str, Any]] = None) -> Tuple[ComparisonParams, Dict[str, Any]]:
        a1_hint = (hints or {}).get("asset1_mention") if hints else None
        a2_hint = (hints or {}).get("asset2_mention") if hints else None
        date_hint = (hints or {}).get("date") if hints else None

        param_debug = {"param_agent_input_prompt": user_prompt, "hints": hints, "resolver_debug": []}

        if not (a1_hint and a2_hint):
            system = (
                "Extract two asset mentions for a 'comparison' (spread) question and CANONICALIZE them for fixed income.\n"
                "If a qualifier (country/region, asset class, rating, tenor) appears on the FIRST mention and is omitted on the SECOND, "
                "PROPAGATE it to the second mention. If the second has a DIFFERENT explicit qualifier, keep it.\n"
                "Return ONLY JSON (json): {asset1_mention: <string>, asset2_mention: <string>}."
            )
            resp = client.chat.completions.create(
                model=self.model, temperature=0, response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user_prompt}],
            )
            data = json.loads(resp.choices[0].message.content)
            m1 = data.get("asset1_mention", "")
            m2 = data.get("asset2_mention", "")
        else:
            m1, m2 = a1_hint, a2_hint

        t1, dbg1 = self._resolve_asset(m1, hints)
        t2, dbg2 = self._resolve_asset(m2, hints)
        param_debug["resolver_debug"] = [dbg1, dbg2]

        if date_hint:
            date_str, raw_dt = date_hint, None
        else:
            d = self._extract_dates_via_llm(user_prompt, mode="single")
            date_str, raw_dt = d.get("date") or self.md.last_date.isoformat(), d.get("raw_date_text")

        return (
            ComparisonParams(
                asset1=t1, asset2=t2,
                date=datetime.fromisoformat(date_str).date(),
                raw_date_text=raw_dt,
            ),
            param_debug
        )

# ----------------------- Orchestrator ------------------------

class MarketMonitorChatbot:
    def __init__(self, md: Optional[MarketData] = None):
        self.md = md or MarketData(EXCEL_PATH)
        self.rewriter = QueryRewriter(self.md)
        self.resolver = AssetResolver(self.md)
        self.router = RouterAgent()
        self.price_agent = PriceParamAgent(self.md, self.resolver)
        self.change_agent = PriceChangeParamAgent(self.md, self.resolver)
        self.return_agent = ReturnParamAgent(self.md, self.resolver)
        self.compare_agent = ComparisonParamAgent(self.md, self.resolver)

    # Required calculation functions
    def calc_price(self, p: PriceParams) -> Dict[str, Any]:
        d, v = self.md.price(p.asset, p.date or self.md.last_date)
        return {"date_actual": d, "price": float(v)}

    def calc_price_change(self, p: PriceChangeParams) -> Dict[str, Any]:
        return self.md.price_change(p.asset, p.start_date, p.end_date)

    def calc_return(self, p: ReturnParams) -> Dict[str, Any]:
        return self.md.total_return(p.asset, p.start_date, p.end_date)

    def calc_comparison(self, p: ComparisonParams) -> Dict[str, Any]:
        return self.md.compare_spread(p.asset1, p.asset2, p.date or self.md.last_date)

    def _debug_top3(self) -> List[Dict[str, Any]]:
        out = []
        for d in self.resolver.debug_log:
            dd = {
                "mention": d.get("mention"),
                "tokens": d.get("tokens"),
                "tokens_raw": d.get("tokens_raw"),
                "tokens_defaults": d.get("tokens_defaults"),
                "accept_path": d.get("accept_path"),
                "accepted": d.get("accepted"),
                "candidates_top3": d.get("candidates_top3"),
            }
            if d.get("tie_breaker_prompt"):
                dd["tie_breaker_prompt"] = d["tie_breaker_prompt"]
            if d.get("postfilter_top3"):
                dd["postfilter_top3"] = d["postfilter_top3"]
            out.append(dd)
        return out

    def _unsupported(self, decision: RouteDecision) -> Dict[str, Any]:
        return {
            "used_function": "none",
            "router_decision": decision.model_dump(),
            "message": (
                "Your prompt did not match an available function.\n"
                "Supported functions:\n"
                "  • price(asset, date)\n"
                "  • price_change(asset, start_date, end_date)\n"
                "  • return(asset, start_date, end_date)\n"
                "  • comparison(asset1, asset2, date)\n"
                "Additional capabilities may be added in future versions."
            ),
        }

    def answer(self, user_prompt: str) -> Dict[str, Any]:
        # Clear resolver debug for this turn
        self.resolver.debug_log.clear()

        # Rewriter first
        rw = self.rewriter.rewrite(user_prompt)

        # If the rewriter explicitly says "none", only hard-decline when it's truly out-of-scope.
        hard_block_keywords = [
            "interpolat", "extrapolat", "curve", "bootstrap", "forecast", "predict", "projection",
            "optimiz", "hedg", "dv01", "duration", "portfolio", "weights",
            "option", "swaption", "greeks", "implied",
            "volatility", "standard deviation", "std dev", "stdev", "variance", "risk",
            "var", "value at risk", "monte", "simulation", "scenario",
            "correlation", "covariance", "beta",
            "plot", "chart", "graph", "visualiz"
        ]
        def _looks_out_of_scope(reason: Optional[str]) -> bool:
            if not reason:
                return False
            r = reason.lower()
            return any(k in r for k in hard_block_keywords)

        if rw.function is None and _looks_out_of_scope(rw.reject_reason):
            resp = self._unsupported(RouteDecision(function="none", confidence=1.0, reason="Rewriter declined"))
            resp["rewriter"] = rw.model_dump()
            if rw.reject_reason:
                resp["message"] = f"Your prompt did not match an available function. {rw.reject_reason}"
            return resp

        # Proceed: let the router decide (use the rewriter's final_query if present)
        normalized_prompt = rw.final_query or user_prompt
        decision = self.router.route(normalized_prompt)
        if rw.function in {"price", "price_change", "return", "comparison"}:
            decision = RouteDecision(
                function=rw.function,
                confidence=max(0.95, float(decision.confidence or 0.0)),
                reason=f"Rewriter selected '{rw.function}'. Router suggestion: {decision.function}."
            )

        # Compose hints for param agents (include FI defaults & asset_class so resolver can use them)
        hints = {
            "asset_mention": rw.asset_mention,
            "asset1_mention": rw.asset1_mention,
            "asset2_mention": rw.asset2_mention,
            "date": rw.date,
            "start_date": rw.start_date,
            "end_date": rw.end_date,
            "used_defaults": rw.used_defaults,
            "asset_class": rw.asset_class,
        }

        try:
            if decision.function == "price":
                params, pdebug = self.price_agent.extract(normalized_prompt, hints=hints)
                out = self.calc_price(params)
                return self._format("price", params.model_dump(), out, decision, pdebug, rw)

            elif decision.function == "price_change":
                params, pdebug = self.change_agent.extract(normalized_prompt, hints=hints)
                out = self.calc_price_change(params)
                return self._format("price_change", params.model_dump(), out, decision, pdebug, rw)

            elif decision.function == "return":
                params, pdebug = self.return_agent.extract(normalized_prompt, hints=hints)
                out = self.calc_return(params)
                return self._format("return", params.model_dump(), out, decision, pdebug, rw)

            elif decision.function == "comparison":
                params, pdebug = self.compare_agent.extract(normalized_prompt, hints=hints)
                out = self.calc_comparison(params)
                return self._format("comparison", params.model_dump(), out, decision, pdebug, rw)

            else:
                resp = self._unsupported(decision)
                resp["rewriter"] = rw.model_dump()
                resp["resolution_debug"] = self._debug_top3()
                return resp

        except AssetResolutionError as e:
            suggestions = [{"ticker": t, "name": n, "score": round(s, 3)} for t, n, s in (e.suggestions or [])]
            resp = {
                "used_function": "none",
                "router_decision": decision.model_dump(),
                "message": (
                    f"I couldn't map '{e.mention}' to an asset in this workbook. "
                    "Please use one of the assets present in your data (ticker or name). "
                    "Here are the closest matches I found:"
                ),
                "suggestions": suggestions,
            }
            resp["rewriter"] = rw.model_dump()
            resp["resolution_debug"] = self._debug_top3()
            resp["param_agent_input_prompt"] = normalized_prompt
            resp["param_agent_hints"] = hints
            return resp

        except KeyError as e:
            return {
                "used_function": decision.function,
                "router_decision": decision.model_dump(),
                "rewriter": rw.model_dump(),
                "resolution_debug": self._debug_top3(),
                "param_agent_input_prompt": normalized_prompt,
                "param_agent_hints": hints,
                "message": f"Asset {str(e).strip()} is not in the price sheet (BBG Data). "
                           "Please verify the workbook or pick another asset.",
            }

        except ValidationError as ve:
            return {"used_function": "none",
                    "router_decision": decision.model_dump(),
                    "rewriter": rw.model_dump(),
                    "resolution_debug": self._debug_top3(),
                    "param_agent_input_prompt": normalized_prompt,
                    "param_agent_hints": hints,
                    "message": "Parameter validation failed.",
                    "details": json.loads(ve.json())}

    def _format(self, fn: str, params: Dict[str, Any], output: Dict[str, Any],
                decision: RouteDecision, pdebug: Dict[str, Any], rw: RewriteDecision) -> Dict[str, Any]:
        def fmt(x):
            if isinstance(x, (pd.Timestamp, datetime, date)):
                return _iso(x)
            return x

        out_ser = {k: fmt(v) for k, v in output.items()}
        params_ser = {k: fmt(v) for k, v in params.items()}

        def label_asset(t):
            return f"{t} ({self.md.ticker_to_name.get(t, 'unknown')})"
        if "asset" in params_ser:  params_ser["asset_label"] = label_asset(params_ser["asset"])
        if "asset1" in params_ser: params_ser["asset1_label"] = label_asset(params_ser["asset1"])
        if "asset2" in params_ser: params_ser["asset2_label"] = label_asset(params_ser["asset2"])

        return {
            "used_function": fn,
            "router_decision": decision.model_dump(),
            "parameters": params_ser,
            "result": out_ser,
            "message": f"Used '{fn}' with extracted parameters shown.",
            "rewriter": rw.model_dump(),
            "param_agent_input_prompt": pdebug.get("param_agent_input_prompt"),
            "param_agent_hints": pdebug.get("hints"),
            "resolution_debug": self._debug_top3(),
        }

# ----------------------- Batch utilities ---------------------

def load_prompts_file(path: str) -> List[Tuple[int, str]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    ext = p.suffix.lower()

    rows: List[Tuple[int, str]] = []
    if ext in {".csv"}:
        df = pd.read_csv(path)
        if "prompt" not in df.columns:
            if df.shape[1] == 1:
                df.columns = ["prompt"]
            else:
                raise ValueError("CSV must contain a 'prompt' column.")
        for i, r in df.iterrows():
            rid = int(r.get("id", i + 1))
            q = str(r["prompt"]).strip()
            if q:
                rows.append((rid, q))
    elif ext in {".txt"}:
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        rows = [(i + 1, ln) for i, ln in enumerate(lines)]
    else:
        raise ValueError("Unsupported input format. Please use .csv or .txt")

    return rows

def flatten_for_summary(row_id: int, prompt: str, resp: Dict[str, Any]) -> Dict[str, Any]:
    router = resp.get("router_decision", {}) or {}
    params = resp.get("parameters", {}) or {}
    result = resp.get("result", {}) or {}
    rewriter = resp.get("rewriter", {}) or {}

    def get_first(*keys):
        for k in keys:
            v = result.get(k)
            if v is not None:
                return v
        return None

    out = {
        "id": row_id,
        "prompt": prompt,
        "rewritten_prompt": rewriter.get("final_query"),
        "rewriter_function": rewriter.get("function"),
        "rewriter_asset_class": rewriter.get("asset_class"),
        "used_function": resp.get("used_function"),
        "router_function": router.get("function"),
        "router_confidence": router.get("confidence"),
        "router_reason": router.get("reason"),
        "message": resp.get("message"),
        # Parameters (unified)
        "param_asset": params.get("asset"),
        "param_asset_label": params.get("asset_label"),
        "param_asset1": params.get("asset1"),
        "param_asset1_label": params.get("asset1_label"),
        "param_asset2": params.get("asset2"),
        "param_asset2_label": params.get("asset2_label"),
        "param_date": params.get("date"),
        "param_start_date": params.get("start_date"),
        "param_end_date": params.get("end_date"),
        # Result (unified)
        "result_date": get_first("date_actual", "end_date_actual", "start_date_actual", "date"),
        "start_value": result.get("start_value"),
        "end_value": result.get("end_value"),
        "absolute_change": result.get("absolute_change"),
        "pct_change": result.get("pct_change"),
        "total_return_pct": result.get("total_return_pct"),
        "asset1_value": result.get("asset1_value"),
        "asset2_value": result.get("asset2_value"),
        "spread": result.get("spread"),
        # suggestions (stringified json if present)
        "suggestions": json.dumps(resp.get("suggestions", []), ensure_ascii=False)
            if "suggestions" in resp else None,
    }
    return out

def make_html_report(summary_df: pd.DataFrame,
                     details_df: pd.DataFrame,
                     stats_df: pd.DataFrame,
                     md: MarketData,
                     html_path: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    wb = md.path
    last_date = _iso(md.last_ts)

    def assets_display(row) -> str:
        if pd.notna(row.get("param_asset")):
            return row.get("param_asset_label") or row.get("param_asset")
        a1 = row.get("param_asset1_label") or row.get("param_asset1") or ""
        a2 = row.get("param_asset2_label") or row.get("param_asset2") or ""
        if a1 or a2:
            return f"{a1} vs {a2}".strip()
        return ""
    view = summary_df.copy()
    view["assets"] = view.apply(assets_display, axis=1)
    cols = [
        "id", "used_function", "router_function", "router_confidence",
        "prompt", "rewritten_prompt", "rewriter_function", "rewriter_asset_class",
        "assets",
        "param_date", "param_start_date", "param_end_date",
        "result_date", "start_value", "end_value",
        "absolute_change", "pct_change", "total_return_pct", "spread",
        "message"
    ]
    cols = [c for c in cols if c in view.columns]
    view = view[cols]

    def df_to_html(df: pd.DataFrame) -> str:
        return df.to_html(index=False, escape=True, border=0)

    stats_html = df_to_html(stats_df)
    summary_html = df_to_html(view)

    details_blocks = []
    for _, r in details_df.iterrows():
        rid = r["id"]
        prompt = html_mod.escape(str(r["prompt"]))
        try:
            obj = json.loads(r["response_json"])
            pretty = json.dumps(obj, indent=2, ensure_ascii=False)
        except Exception:
            pretty = str(r["response_json"])
        pretty_html = html_mod.escape(pretty)
        details_blocks.append(
            f"<details><summary><b>#{rid}</b> — {prompt}</summary>"
            f"<pre>{pretty_html}</pre></details>"
        )
    details_html = "\n".join(details_blocks)

    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 24px; }
    h1 { margin-top: 0; }
    table { border-collapse: collapse; width: 100%; margin: 12px 0 24px 0; font-size: 14px; }
    th, td { border: 1px solid #ddd; padding: 6px 8px; vertical-align: top; }
    th { background: #f3f3f3; }
    details { margin: 8px 0; }
    details > summary { cursor: pointer; padding: 6px 0; }
    pre { background: #fafafa; padding: 10px; border: 1px solid #eee; overflow-x: auto; }
    .meta { color: #333; font-size: 14px; }
    .badge { display: inline-block; padding: 2px 6px; border-radius: 4px; border: 1px solid #ccc; background: #f8f8f8; margin-left: 6px; }
    """

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Market Monitor Chatbot — Batch Report</title>
<style>{css}</style>
</head>
<body>
<h1>Market Monitor Chatbot — Batch Report</h1>
<div class="meta">
  <div>Generated: <b>{html_mod.escape(ts)}</b></div>
  <div>Workbook: <b>{html_mod.escape(str(wb))}</b></div>
  <div>Last data date: <b>{html_mod.escape(str(last_date))}</b></div>
  <div>Prompts evaluated: <b>{len(summary_df)}</b></div>
</div>

<h2>Stats</h2>
{stats_html}

<h2>Summary</h2>
{summary_html}

<h2>Details (full JSON responses)</h2>
{details_html}

</body>
</html>"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_doc)

def run_batch(bot: "MarketMonitorChatbot", input_path: str,
              out_xlsx: str = "batch_results.xlsx",
              out_jsonl: Optional[str] = "batch_results.jsonl",
              out_html: Optional[str] = "batch_results.html",
              limit: Optional[int] = None) -> None:
    prompts = load_prompts_file(input_path)
    if limit is not None:
        prompts = prompts[:int(limit)]

    summary_rows: List[Dict[str, Any]] = []
    detail_records: List[Dict[str, Any]] = []

    print(f"Running batch on {len(prompts)} prompts...")
    for row_id, q in prompts:
        try:
            resp = bot.answer(q)
        except Exception as e:
            resp = {
                "used_function": "error",
                "message": f"Exception: {type(e).__name__}: {e}",
                "router_decision": {"function": None, "confidence": 0, "reason": "exception"},
            }
        summary_rows.append(flatten_for_summary(row_id, q, resp))
        detail_records.append({
            "id": row_id,
            "prompt": q,
            "response_json": json.dumps(resp, ensure_ascii=False)
        })

    summary_df = pd.DataFrame(summary_rows)
    details_df = pd.DataFrame(detail_records)
    stats_df = summary_df.groupby("used_function", dropna=False).size() \
                         .reset_index(name="count") \
                         .sort_values("count", ascending=False)

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        summary_df.to_excel(writer, index=False, sheet_name="Summary")
        details_df.to_excel(writer, index=False, sheet_name="Details")
        stats_df.to_excel(writer, index=False, sheet_name="Stats")

    if out_jsonl:
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for rec in detail_records:
                obj = {
                    "id": rec["id"],
                    "prompt": rec["prompt"],
                    "response": json.loads(rec["response_json"])
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    if out_html:
        make_html_report(summary_df, details_df, stats_df, bot.md, out_html)

    print(f"✓ Wrote Excel report to: {out_xlsx}")
    if out_jsonl:
        print(f"✓ Wrote JSONL to: {out_jsonl}")
    if out_html:
        print(f"✓ Wrote HTML report to: {out_html}")

# ----------------------- Manual Calculator UI ------------------------

def render_manual_calculator(md: MarketData, bot: MarketMonitorChatbot):
    import streamlit as st  # local import to avoid issues if streamlit isn't installed

    st.subheader("🔧 Manual calculator")

    # --- Function selector (top) ---
    fn_labels = {
        "price": "Price (asset, single date)",
        "price_change": "Price change (asset, start/end)",
        "return": "Return (asset, start/end)",
        "comparison": "Comparison (two assets, single date)",
    }
    fn_key_order = ["price", "price_change", "return", "comparison"]
    function = st.selectbox(
        "Function",
        options=fn_key_order,
        format_func=lambda x: fn_labels[x],
        key="manual_fn",
    )

    # --- Ticker selectors ---
    tickers = list(md.df_prices.columns)
    ticker_labels = {t: f"{t} — {md.ticker_to_name.get(t, '')}" for t in tickers}

    def fmt_ticker(t: str) -> str:
        return ticker_labels.get(t, t)

    # Asset 1 (always enabled)
    asset1 = st.selectbox(
        "Asset 1",
        options=tickers,
        format_func=fmt_ticker,
        key="manual_asset1",
    )

    # Asset 2 (disabled for single‑asset functions)
    asset2_disabled = function in ("price", "price_change", "return")
    asset2 = st.selectbox(
        "Asset 2",
        options=tickers,
        format_func=fmt_ticker,
        key="manual_asset2",
        disabled=asset2_disabled,
    )

    # --- Dates ---
    min_date = md.available_dates.min().date()
    max_date = md.available_dates.max().date()
    ref_date = md.last_date
    default_start = date(ref_date.year, 1, 1)

    is_single_date_fn = function in ("price", "comparison")

    # Start date (greyed out for price / comparison)
    start_disabled = is_single_date_fn
    start_date_val = st.date_input(
        "Start date",
        value=default_start,
        min_value=min_date,
        max_value=max_date,
        key="manual_start_date",
        disabled=start_disabled,
    )

    # End date (or single date)
    end_label = "Date" if is_single_date_fn else "End date"
    end_date_val = st.date_input(
        end_label,
        value=ref_date,
        min_value=min_date,
        max_value=max_date,
        key="manual_end_date",
    )

    # --- Calculate button ---
    calculate_pressed = st.button("Calculate", key="manual_calc_btn", use_container_width=True)

    if calculate_pressed:
        try:
            if function == "price":
                params = PriceParams(asset=asset1, date=end_date_val)
                res = bot.calc_price(params)
                summary = {
                    "function": "price",
                    "asset": fmt_ticker(asset1),
                    "date_actual": _iso(res["date_actual"]),
                    "price": res["price"],
                }

            elif function == "price_change":
                params = PriceChangeParams(
                    asset=asset1,
                    start_date=start_date_val,
                    end_date=end_date_val,
                    raw_range_text=None,
                )
                res = bot.calc_price_change(params)
                summary = {
                    "function": "price_change",
                    "asset": fmt_ticker(asset1),
                    "start_date_actual": _iso(res["start_date_actual"]),
                    "end_date_actual": _iso(res["end_date_actual"]),
                    "start_value": res["start_value"],
                    "end_value": res["end_value"],
                    "absolute_change": res["absolute_change"],
                    "pct_change": res["pct_change"],
                }

            elif function == "return":
                params = ReturnParams(
                    asset=asset1,
                    start_date=start_date_val,
                    end_date=end_date_val,
                    raw_range_text=None,
                )
                res = bot.calc_return(params)
                summary = {
                    "function": "return",
                    "asset": fmt_ticker(asset1),
                    "start_date_actual": _iso(res["start_date_actual"]),
                    "end_date_actual": _iso(res["end_date_actual"]),
                    "total_return_pct": res["total_return_pct"],
                }

            elif function == "comparison":
                if asset2_disabled or not asset2:
                    st.warning("Please select Asset 2 for a comparison.")
                    summary = None
                else:
                    params = ComparisonParams(
                        asset1=asset1,
                        asset2=asset2,
                        date=end_date_val,
                        raw_date_text=None,
                    )
                    res = bot.calc_comparison(params)
                    summary = {
                        "function": "comparison",
                        "asset1": fmt_ticker(asset1),
                        "asset2": fmt_ticker(asset2),
                        "date_actual": _iso(res["date_actual"]),
                        "asset1_value": res["asset1_value"],
                        "asset2_value": res["asset2_value"],
                        "spread": res["spread"],
                    }
            else:
                summary = {"error": f"Unsupported function: {function}"}

            st.session_state["manual_last_result"] = summary

        except Exception as e:
            st.session_state["manual_last_result"] = {"error": str(e)}

    # --- Result at the bottom of the panel ---
    if "manual_last_result" in st.session_state and st.session_state["manual_last_result"]:
        st.markdown("---")
        st.markdown("**Result**")
        st.json(st.session_state["manual_last_result"])


# ----------------------- Streamlit UI ------------------------

def run_streamlit_app():
    try:
        import streamlit as st
    except Exception as e:
        print("Streamlit is not installed. Please `pip install streamlit`.")
        raise

    st.set_page_config(page_title="Market Monitor Chatbot", page_icon="📈", layout="wide")
    st.title("📈 Market Monitor Chatbot")
    st.caption("Simple chat UI for end users. Batch runner remains available via CLI.")

    # Global CSS for hover tooltips (Tips + Tickers)
    CUSTOM_CSS = """
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
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Instantiate bot & metadata (cache across reruns)
    @st.cache_resource
    def _load_bot():
        md = MarketData(EXCEL_PATH)
        bot = MarketMonitorChatbot(md)
        return bot

    bot = _load_bot()
    md = bot.md

    # Header info (top)
    with st.container():
        col1, col2, col3 = st.columns([3, 3, 2])
        with col1:
            st.markdown(f"**Workbook**: `{md.path}`")
        with col2:
            st.markdown(f"**Last data date**: `{_iso(md.last_ts)}`")
        with col3:
            # Tickers tooltip with scrollable list
            tickers_in_prices = [str(t) for t in md.df_prices.columns]
            rows_html = []
            for t in tickers_in_prices:
                desc = md.ticker_to_name.get(t, "")
                rows_html.append(
                    f"<div><code>{html_mod.escape(t)}</code> — {html_mod.escape(desc)}</div>"
                )
            tickers_inner = "\n".join(rows_html) or "<div><em>No tickers loaded.</em></div>"
            tickers_tooltip_html = f"""
            <div class="mm-hover-wrapper">
              <span class="mm-hover-label">Tickers in data: {len(tickers_in_prices)} ▾</span>
              <div class="mm-tooltip tickers">
                <strong>Tickers in workbook ({len(tickers_in_prices)}):</strong><br/>
                <div class="mm-tickers-list">
                  {tickers_inner}
                </div>
              </div>
            </div>
            """
            st.markdown(tickers_tooltip_html, unsafe_allow_html=True)

    # Sidebar options
    with st.sidebar:
        st.header("Options")
        show_debug = st.checkbox("Show debug info (rewriter/router/candidates)", value=False)
        st.markdown("---")

        # Tips title with hover for extra examples
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
        extra_tips_items = "".join(
            f"<li>{html_mod.escape(q)}</li>" for q in extra_tips_questions
        )
        tips_tooltip_html = f"""
        <div class="mm-hover-wrapper">
          <span class="mm-hover-label">Tips ▾</span>
          <div class="mm-tooltip tips">
            <strong>More example questions:</strong>
            <ul>
              {extra_tips_items}
            </ul>
          </div>
        </div>
        """
        st.markdown(tips_tooltip_html, unsafe_allow_html=True)

        # Always-visible short tips list
        st.markdown(
            "- Ask things like:\n"
            "  - *How's the market doing?*\n"
            "  - *Spread between US 10Y swap rate and Treasury rate*\n"
            "  - *Canada BBB 10Y corporate yield change YTD*\n"
            "  - *How much higher is US 3M vs Canada?*\n"
        )

    # Layout: main chat on the left, manual calculator on the right
    main_col, calc_col = st.columns([3, 2], gap="large")

    # --- Left: chat UI ---
    with main_col:
        # Session history
        if "history" not in st.session_state:
            st.session_state["history"] = []  # list of dicts: {"q": str, "resp": dict}

        # Chat input
        with st.form(key="chat_form", clear_on_submit=True):
            q = st.text_input("Ask a question", value="", placeholder="e.g., How's the market doing?")
            submitted = st.form_submit_button("Send")
        if submitted and q.strip():
            resp = bot.answer(q.strip())
            st.session_state["history"].append({"q": q.strip(), "resp": resp})

        # Render history
        for turn in st.session_state["history"]:
            st.markdown(f"**You:** {turn['q']}")
            resp = turn["resp"]
            used_fn = resp.get("used_function")
            params = resp.get("parameters", {})
            result = resp.get("result", {})
            msg = resp.get("message", "")
            rewriter = resp.get("rewriter", {}) or {}
            router = resp.get("router_decision", {}) or {}

            # Compact card
            with st.container():
                st.markdown("> **Assistant:**")
                if used_fn in ("price", "price_change", "return", "comparison"):
                    # Build a concise summary
                    lines = []
                    rq = rewriter.get("final_query")
                    if rq and rq != turn["q"]:
                        lines.append(f"- **Rewritten:** {rq}")

                    lines.append(f"- **Function:** `{used_fn}`")

                    # Assets
                    if used_fn in ("price", "price_change", "return"):
                        asset_label = params.get("asset_label") or params.get("asset")
                        lines.append(f"- **Asset:** {asset_label}")
                    else:
                        a1 = params.get("asset1_label") or params.get("asset1")
                        a2 = params.get("asset2_label") or params.get("asset2")
                        lines.append(f"- **Assets:** {a1}  vs  {a2}")

                    # Dates and results
                    if used_fn == "price":
                        lines.append(f"- **Date:** {params.get('date')}")
                        lines.append(f"- **Price:** {result.get('price')}")
                    elif used_fn in ("price_change", "return"):
                        lines.append(f"- **Start → End:** {params.get('start_date')} → {params.get('end_date')}")
                        if used_fn == "price_change":
                            lines.append(
                                f"- **Δ (abs / %):** {result.get('absolute_change')} / {result.get('pct_change')}%"
                            )
                        else:
                            lines.append(f"- **Return:** {result.get('total_return_pct')}%")
                    elif used_fn == "comparison":
                        lines.append(f"- **Date:** {params.get('date')}")
                        lines.append(
                            f"- **Values:** {result.get('asset1_value')} vs {result.get('asset2_value')}"
                        )
                        lines.append(f"- **Spread:** {result.get('spread')}")

                    if msg:
                        lines.append(f"- **Note:** {msg}")

                    st.markdown("\n".join(lines))
                else:
                    st.warning(msg or "Your prompt did not match an available function.")

                if show_debug:
                    with st.expander("Debug (rewriter / router / candidates)"):
                        st.json(
                            {
                                "rewriter": rewriter,
                                "router": router,
                                "resolution_debug": resp.get("resolution_debug"),
                            }
                        )
            st.markdown("---")

    # --- Right: always‑ready manual calculator ---
    with calc_col:
        render_manual_calculator(md, bot)


# ----------------------- CLI / main --------------------------

EXAMPLE_QUERIES = [
    "What's the YTD return for SPX?",
    "What’s the YTD return for S&P 500?",
    "SPX price on 2025-06-14",
    "Change in USGG10YR Index from 2025-01-02 to 2025-02-15",
    "What is the spread between the Canada 10Y BBB corporate yield and Gov yield?",
    "Compare USGG2YR Index and USGG10YR Index, latest",
    "Spread between US 10Y swap rate and Treasury rate",
    "How's the market doing?"
]

def _pp(obj: Dict[str, Any]):
    import pprint
    pprint.PrettyPrinter(indent=2, width=100).pprint(obj)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Market Monitor Chatbot")
    parser.add_argument("--batch-input", type=str, default=None,
                        help="Path to CSV/TXT file with prompts (CSV preferred; column 'prompt', optional 'id').")
    parser.add_argument("--out-xlsx", type=str, default="batch_results.xlsx",
                        help="Excel output file for the batch report.")
    parser.add_argument("--out-jsonl", type=str, default="batch_results.jsonl",
                        help="Optional JSONL output (set to '' to skip).")
    parser.add_argument("--out-html", type=str, default="batch_results.html",
                        help="Optional HTML report (set to '' to skip).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional limit on number of prompts to process from the input file.")
    parser.add_argument("--ui", action="store_true", help="Run Streamlit UI inside this process (for streamlit).")
    args = parser.parse_args()

    if args.ui:
        # Streamlit will run this module and pass --ui through; call the UI function.
        run_streamlit_app()
    else:
        md = MarketData(EXCEL_PATH)
        bot = MarketMonitorChatbot(md)

        if args.batch_input:
            print("Using workbook:", md.path)
            print("Tickers in BBG Data:", len(md.df_prices.columns))
            print("Market Monitor Chatbot ready. Data last date:", _iso(md.last_ts))
            out_jsonl = args.out_jsonl if args.out_jsonl else None
            out_html = args.out_html if args.out_html else None
            run_batch(bot, args.batch_input, args.out_xlsx, out_jsonl, out_html, limit=args.limit)
        else:
            # interactive console demo mode
            print("Using workbook:", md.path)
            print("Tickers in BBG Data:", len(md.df_prices.columns))
            print("Market Monitor Chatbot ready. Data last date:", _iso(md.last_ts))
            print("Type a question, or press Enter to run a few examples...\n")
            try:
                q = input("> ").strip()
            except EOFError:
                q = ""

            if not q:
                for e in EXAMPLE_QUERIES:
                    print(f"\nQ: {e}")
                    _pp(bot.answer(e))
            else:
                _pp(bot.answer(q))
