#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Valuation: DCF & DDM", page_icon="💹", layout="wide")

# ============================== Data helpers ==============================

def format_with_commas(x, decimals=0):
    try:
        if x is None or not np.isfinite(float(x)):
            return ""
        if decimals == 0:
            return f"{float(x):,.0f}"
        return f"{float(x):,.{decimals}f}"
    except Exception:
        return ""

def parse_number(s: str, default=None):
    if s is None:
        return default
    s = s.strip().replace(",", "")
    if s == "":
        return default
    try:
        return float(s)
    except Exception:
        return default
    
def fcf_series_from_yahoo(ticker: str) -> pd.Series | None:
    """
    Build an ANNUAL FCF series, most-recent-first, indexed by year.
    Preference order:
      (a) 'freecashflow' row from annual / quarterly cashflow tables
      (b) fallback: OCF - CapEx from the same tables
    """
    t = yf.Ticker(ticker)

    def _series_from_row(df: pd.DataFrame, row_key: str) -> pd.Series | None:
        if df is None or df.empty:
            return None
        cf = _norm_index(df.copy())
        if row_key not in cf.index:
            return None
        s = pd.to_numeric(cf.loc[row_key], errors="coerce").dropna()
        if s.empty:
            return None
        # Yahoo uses columns as dates/periods
        try:
            idx = pd.to_datetime(s.index)
        except Exception:
            idx = s.index
        return pd.Series(s.values, index=idx)

    def _ocf_minus_capex(df: pd.DataFrame) -> pd.Series | None:
        if df is None or df.empty:
            return None
        cf = _norm_index(df.copy())
        ocf = capex = None
        for k in ["totalcashfromoperatingactivities", "operatingcashflow"]:
            if k in cf.index:
                ocf = pd.to_numeric(cf.loc[k], errors="coerce")
                break
        for k in ["capitalexpenditures", "investmentsppecapex"]:
            if k in cf.index:
                capex = pd.to_numeric(cf.loc[k], errors="coerce")
                break
        if ocf is None or capex is None:
            return None
        s = (ocf - capex).dropna()
        if s.empty:
            return None
        try:
            idx = pd.to_datetime(s.index)
        except Exception:
            idx = s.index
        return pd.Series(s.values, index=idx)

    # Try annual 'freecashflow' row
    s = _series_from_row(t.cashflow, "freecashflow")
    # Else quarterly 'freecashflow' row
    if s is None:
        s = _series_from_row(getattr(t, "quarterly_cashflow", None), "freecashflow")
    # Else annual OCF - CapEx
    if s is None:
        s = _ocf_minus_capex(t.cashflow)
    # Else quarterly OCF - CapEx
    if s is None:
        s = _ocf_minus_capex(getattr(t, "quarterly_cashflow", None))

    if s is None or s.empty:
        return None

    s = s.sort_index(ascending=False)
    # roll up to years (sum quarters or keep annual)
    s_year = s.groupby(s.index.year).sum()
    return s_year

def infer_fcf_avg_from_ticker(ticker: str, years: int,
                              market_cap: float | None,
                              revenue: float | None = None) -> float | None:
    """
    Average of last N *scaled* annual FCF values.
    Each year's value is individually scaled vs revenue to correct thousand/million mixups.
    """
    ser = fcf_series_from_yahoo(ticker)
    if ser is None or ser.empty:
        return None

    # scale each year first, then average
    vals = []
    for v in ser.head(years).astype(float).values:
        vv, _ = scale_if_thousands(float(v), market_cap, revenue)
        vals.append(vv)

    if not vals:
        return None
    return float(np.mean(vals))

# ---- heuristics to fix Yahoo "values in thousands" issue ----
def approx_market_cap(price, shares):
    try:
        if price and shares:
            return float(price) * float(shares)
    except Exception:
        pass
    return None

def fetch_revenue(ticker: str):
    try:
        t = yf.Ticker(ticker)
        fin = t.financials
        if fin is not None and not fin.empty:
            fin = _norm_index(fin)
            if "totalrevenue" in fin.index:
                s = pd.to_numeric(fin.loc["totalrevenue"], errors="coerce").dropna()
                if not s.empty:
                    return float(s.iloc[0])
    except Exception:
        pass
    return None

def scale_if_thousands(val, mktcap=None, revenue=None):
    """
    Fix Yahoo 'values in thousands' problems using revenue as the primary anchor.
    We scale PER VALUE (used for series and single numbers).
    """
    if val is None or not np.isfinite(val) or val == 0:
        return val, False

    # Scale UP if clearly too small
    if revenue and abs(val) < 0.005 * revenue:      # <0.5% of revenue
        return val * 1000.0, True
    if mktcap and abs(val) < 0.0005 * (mktcap or 0): # <0.05% of market cap
        return val * 1000.0, True

    # Scale DOWN if clearly too large (use revenue only; avoid cap-based false positives)
    if revenue and abs(val) > 0.80 * revenue:       # >80% of revenue is suspicious for FCF
        return val / 1000.0, True

    return val, False

def get_company_name(ticker: str):
    try:
        info = yf.Ticker(ticker).get_info()
        return info.get("longName") or info.get("shortName")
    except Exception:
        return None
    
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yahoo_all(ticker: str):
    t = yf.Ticker(ticker)

    # Price
    price = None
    try:
        price = t.fast_info.get("last_price")
    except Exception:
        pass
    if price is None:
        try:
            price = t.info.get("currentPrice")
        except Exception:
            pass
    if price is None:
        hist = t.history(period="5d")
        if not hist.empty:
            price = float(hist["Close"].iloc[-1])

    # Shares
    shares = None
    try:
        shares = t.fast_info.get("shares")
    except Exception:
        pass
    if shares is None:
        try:
            shares = t.info.get("sharesOutstanding")
        except Exception:
            pass

    # Sector / industry (for auto-model hint)
    sector = None
    industry = None
    try:
        info = t.get_info()
        sector = info.get("sector")
        industry = info.get("industry")
    except Exception:
        pass

    # Statements
    cfs = None
    bs = None
    try:
        cfs = t.cashflow
    except Exception:
        pass
    try:
        bs = t.balance_sheet
    except Exception:
        pass

    # Dividends series
    div = None
    try:
        div = t.dividends
    except Exception:
        pass

    return {"price": price, "shares": shares, "sector": sector, "industry": industry,
            "cashflow": cfs, "balancesheet": bs, "dividends": div}


def _norm_index(df: pd.DataFrame):
    idx = [str(x).strip().lower().replace(" ", "").replace("_", "") for x in df.index]
    df = df.copy()
    df.index = idx
    return df


def infer_fcf(cashflow_df: pd.DataFrame, market_cap: float = None, revenue: float = None):
    if cashflow_df is None or cashflow_df.empty:
        return None
    cf = _norm_index(cashflow_df)

    # Try Free Cash Flow row
    for key in ["freecashflow", "freecashflowttm", "freecashflow(annual)"]:
        if key in cf.index:
            s = pd.to_numeric(cf.loc[key], errors="coerce").dropna()
            if not s.empty:
                raw = float(s.iloc[0])
                val, scaled = scale_if_thousands(raw, market_cap, revenue)
                return val

    # Try OCF - CapEx
    ocf = capex = None
    for k in ["totalcashfromoperatingactivities", "operatingcashflow"]:
        if k in cf.index:
            ocf = pd.to_numeric(cf.loc[k], errors="coerce").dropna()
            break
    for k in ["capitalexpenditures", "investmentsppecapex"]:
        if k in cf.index:
            capex = pd.to_numeric(cf.loc[k], errors="coerce").dropna()
            break
    if ocf is not None and capex is not None and not ocf.empty and not capex.empty:
        raw = float((ocf - capex).iloc[0])
        val, scaled = scale_if_thousands(raw, market_cap, revenue)
        return val

    return None

def infer_fcf_avg(cashflow_df: pd.DataFrame, years: int = 5, market_cap: float = None, revenue: float = None):
    if cashflow_df is None or cashflow_df.empty:
        return None
    cf = _norm_index(cashflow_df)

    ocf = capex = None
    for k in ["totalcashfromoperatingactivities", "operatingcashflow"]:
        if k in cf.index:
            ocf = pd.to_numeric(cf.loc[k], errors="coerce").dropna()
            break
    for k in ["capitalexpenditures", "investmentsppecapex"]:
        if k in cf.index:
            capex = pd.to_numeric(cf.loc[k], errors="coerce").dropna()
            break

    if ocf is not None and capex is not None:
        fcf_series = (ocf - capex).dropna().astype(float)
        if not fcf_series.empty:
            avg = fcf_series.head(years).mean()
            val, scaled = scale_if_thousands(avg, market_cap, revenue)
            return val
    return None

def infer_cash_plus_sti(balance_df: pd.DataFrame, market_cap: float = None):
    if balance_df is None or balance_df.empty:
        return None
    bs = _norm_index(balance_df)

    cash = None
    for k in ["cashandcashequivalents", "cashandcashequivalentsatcarryingvalue", "cash"]:
        if k in bs.index:
            s = pd.to_numeric(bs.loc[k], errors="coerce").dropna()
            if not s.empty:
                cash = float(s.iloc[0]); break
    sti = 0.0
    for k in ["shortterminvestments", "marketablesecuritiescurrent"]:
        if k in bs.index:
            s = pd.to_numeric(bs.loc[k], errors="coerce").dropna()
            if not s.empty:
                sti = float(s.iloc[0]); break

    if cash is None and sti == 0.0:
        return None
    raw = (cash or 0.0) + sti
    val, scaled = scale_if_thousands(raw, market_cap)
    return val


def infer_total_debt(balance_df: pd.DataFrame, market_cap: float = None):
    """
    Use interest-bearing debt only: ST + LT.
    Avoid Yahoo's 'totaldebt' if it looks inflated (could include liabilities).
    """
    if balance_df is None or balance_df.empty:
        return None
    bs = _norm_index(balance_df)

    # Build from components first
    st_debt = 0.0
    lt_debt = 0.0
    for k in ["shortlongtermdebt", "shorttermdebt", "currentportionoflongtermdebt"]:
        if k in bs.index:
            s = pd.to_numeric(bs.loc[k], errors="coerce").dropna()
            if not s.empty:
                st_debt = float(s.iloc[0]); break
    for k in ["longtermdebt", "longtermdebtnoncurrent"]:
        if k in bs.index:
            s = pd.to_numeric(bs.loc[k], errors="coerce").dropna()
            if not s.empty:
                lt_debt = float(s.iloc[0]); break
    comp = st_debt + lt_debt

    # If components missing, cautiously try 'totaldebt'
    if comp == 0.0 and "totaldebt" in bs.index:
        s = pd.to_numeric(bs.loc["totaldebt"], errors="coerce").dropna()
        if not s.empty:
            raw = float(s.iloc[0])
            val, scaled = scale_if_thousands(raw, market_cap)
            return val

    if comp > 0:
        val, scaled = scale_if_thousands(comp, market_cap)
        return val

    return None

def dividends_ttm(div_series: pd.Series):
    """Return trailing-12M dividends per share, fallback to last FY."""
    if div_series is None or div_series.empty:
        return None
    # Yahoo dividends are a time series of per-share cash amounts (same currency as price)
    cutoff = div_series.index.max() - pd.DateOffset(years=1)
    dps_ttm = float(div_series[div_series.index > cutoff].sum())
    if dps_ttm > 0:
        return dps_ttm
    # fallback: last calendar-year sum
    try:
        return float(div_series.resample("Y").sum().iloc[-1])
    except Exception:
        return None


# ============================== Valuation helpers ==============================

def fmt_money(x):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    try:
        return f"${x:,.2f}"
    except Exception:
        return str(x)


# ---- DCF (3-stage, 20 years) ----
def dcf_3stage_table(fcf0, g1, g2, g3, r, years=(5, 5, 10)):
    """Return table (Year, Projected FCF, Discount Factor, Discounted Value) and PV sum."""
    y1, y2, y3 = years
    total = y1 + y2 + y3
    growths = ([g1] * y1) + ([g2] * y2) + ([g3] * y3)
    rows = []
    fcf = fcf0
    for t in range(1, total + 1):
        fcf = fcf * (1 + growths[t - 1])
        disc = (1 + r) ** t
        rows.append({
            "Year": t,
            "Projected FCF": fcf,
            "Discount Factor": 1 / disc,
            "Discounted Value": fcf / disc
        })
    df = pd.DataFrame(rows)
    return df, float(df["Discounted Value"].sum())


# ---- DDM (Gordon & Two-stage) ----
def ddm_gordon(dps0, ke, g):
    """Fair value per share using Gordon Growth: D1/(ke-g)."""
    if dps0 is None or dps0 <= 0:
        return None
    if ke <= g:
        return None
    d1 = dps0 * (1 + g)
    return d1 / (ke - g)


def ddm_two_stage(dps0, ke, g1, n1, gt):
    """Two-stage DDM: grow DPS at g1 for n1 years, then terminal at gt."""
    if dps0 is None or dps0 <= 0:
        return None
    if ke <= gt:
        return None
    pv = 0.0
    d = dps0
    # Stage 1
    for t in range(1, n1 + 1):
        d *= (1 + g1)
        pv += d / ((1 + ke) ** t)
    # Terminal at end of stage 1
    d_terminal = d * (1 + gt)
    tv = d_terminal / (ke - gt)
    pv += tv / ((1 + ke) ** n1)
    return pv


# ============================== UI ==============================

st.title("💹 Valuation Toolkit — DCF (3-stage) & DDM")
    
with st.sidebar:
    st.header("General")
    ticker = st.text_input("Ticker (Yahoo format)", value="FTNT")

    # Fetch here to propose a default model
    hint = fetch_yahoo_all(ticker.strip()) if ticker.strip() else None
    sector = (hint or {}).get("sector") or ""
    industry = (hint or {}).get("industry") or ""
    # Show company name under the main title
    company_name = get_company_name(ticker.strip())
    st.subheader(f"📌 {company_name} ({ticker.upper()})" if company_name else f"📌 {ticker.upper()}")

    is_financial_like = any(x for x in [sector, industry] if x) and (
        "bank" in (industry or "").lower()
        or "financial" in (sector or "").lower()
        or "reit" in (industry or "").lower()
        or "reit" in (sector or "").lower()
    )

    default_model_index = 1 if is_financial_like else 0
    model = st.radio("Valuation model",
                     ["DCF (3-stage, 20y) — for non-financials",
                      "DDM — for banks/REITs"],
                     index=default_model_index)

# ============================== DDM path ==============================

if model == "DDM — for banks/REITs":
    data = hint or fetch_yahoo_all(ticker.strip())
    price = data["price"]
    dps0_auto = dividends_ttm(data["dividends"])
    st.subheader("🏦 Dividend Discount Model (DDM)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Price", fmt_money(price))
    c2.metric("DPS (TTM)", fmt_money(dps0_auto))
    c3.metric("Sector", data.get("sector") or "—")
    c4.metric("Industry", data.get("industry") or "—")

    st.divider()
    st.write("**Inputs**")
    col = st.columns(3)
    ke = col[0].number_input("Cost of equity (kₑ)", min_value=0.06, max_value=0.20, value=0.10, step=0.005, format="%.3f")
    gt = col[1].number_input("Terminal growth (g)", min_value=0.00, max_value=0.06, value=0.03, step=0.005, format="%.3f")
    use_two = col[2].checkbox("Two-stage DDM", value=True)

    default_core = 2.10 if ticker.strip().upper() == "D05.SI" else float(dps0_auto or 0.0)
    dps0 = st.number_input("Dividend per share (core, exclude special)", min_value=0.0,
                       value=default_core, step=0.01, format="%.2f")

    if not use_two:
        fv = ddm_gordon(dps0, ke, gt)
        st.metric("Fair Value (DDM)", fmt_money(fv))
        if fv and price:
            st.metric("Discount/Premium vs Price", f"{(fv - price) / price * 100:+.1f}%")
    else:
        c = st.columns(3)
        n1 = c[0].slider("Stage-1 years (n₁)", min_value=1, max_value=10, value=5, step=1)
        g1 = c[1].number_input("Stage-1 dividend growth (g₁)", min_value=0.00, max_value=0.20, value=0.06, step=0.005, format="%.3f")
        fv = ddm_two_stage(dps0, ke, g1, n1, gt)
        st.metric("Fair Value (Two-stage DDM)", fmt_money(fv))
        if fv and price:
            st.metric("Discount/Premium vs Price", f"{(fv - price) / price * 100:+.1f}%")

    st.caption("Tip: For SG banks (e.g., D05.SI), DDM/Residual-Income is generally more appropriate than FCF-DCF.")
    st.stop()

# ============================== DCF path ==============================

st.subheader("🏭 3-Stage DCF (20 years) — Excel-style")

with st.sidebar:
    st.header("DCF inputs")
    g1 = st.number_input("FCF Growth Yr 1–5", min_value=0.0, max_value=0.50, value=0.158, step=0.001, format="%.3f")
    g2 = st.number_input("FCF Growth Yr 6–10", min_value=0.0, max_value=0.50, value=0.108, step=0.001, format="%.3f")
    g3 = st.number_input("FCF Growth Yr 11–20", min_value=0.0, max_value=0.20, value=0.043, step=0.001, format="%.3f")
    r  = st.number_input("Discount rate (r)", min_value=0.03, max_value=0.20, value=0.063, step=0.001, format="%.3f")

    st.divider()
    st.subheader("FCF / Cash / Debt / Shares")
    
    fcf_source = st.radio(
        "FCF basis",
        ["Latest (TTM)", "5-year average", "Manual override"],
        horizontal=False
    )
    
   # --- Manual override with Yahoo prefills + comma-friendly text inputs ---
    if fcf_source == "Manual override":
        # Pull Yahoo to prefill (same as before)
        data_prefill = fetch_yahoo_all(ticker.strip())
        price_pref   = data_prefill["price"]
        shares_pref  = data_prefill["shares"]
        mktcap_pref  = approx_market_cap(price_pref, shares_pref)

        fcf_pref  = infer_fcf(data_prefill["cashflow"], mktcap_pref) or 0.0
        cash_pref = infer_cash_plus_sti(data_prefill["balancesheet"], mktcap_pref) or 0.0
        debt_pref = infer_total_debt(data_prefill["balancesheet"], mktcap_pref) or 0.0
        shrs_pref = float(shares_pref or 0.0)

        st.caption("You can paste numbers with commas (e.g., 2,470,000,000). Decimals are OK.")

        # Show as text_input so commas are allowed; prefill with nicely formatted strings
        fcf0_in_str = st.text_input("Current FCF ($)", value=format_with_commas(fcf_pref, 0))
        cash_in_str = st.text_input("Cash + short-term investments ($)", value=format_with_commas(cash_pref, 0))
        debt_in_str = st.text_input("Total debt ($)", value=format_with_commas(debt_pref, 0))
        shrs_in_str = st.text_input("Shares outstanding", value=format_with_commas(shrs_pref, 0))

        # Convert back to floats; if user clears a field, fall back to the prefill
        fcf0_in = parse_number(fcf0_in_str, default=fcf_pref)
        cash_in = parse_number(cash_in_str, default=cash_pref)
        debt_in = parse_number(debt_in_str, default=debt_pref)
        shrs_in = parse_number(shrs_in_str, default=shrs_pref)
    else:
        fcf0_in = cash_in = debt_in = shrs_in = None

    run_btn = st.button("Run valuation", type="primary", use_container_width=True)

if not run_btn:
    st.info("Choose model assumptions and click **Run valuation**. If FCF is negative, consider switching to DDM or enter a normalized manual FCF.")
    st.stop()

# Data fetch & inference
data = fetch_yahoo_all(ticker.strip())

t = yf.Ticker(ticker.strip())
company_name = None
try:
    info = t.get_info()
    company_name = info.get("longName") or info.get("shortName")
except Exception:
    company_name = None
    
price = data["price"]
shares = data["shares"] if shrs_in is None else shrs_in
mktcap = approx_market_cap(price, shares)
revenue = fetch_revenue(ticker.strip())

if fcf_source == "Latest (TTM)":
    # Keep your current logic (annual row or OCF–CapEx from annual table)
    fcf0 = infer_fcf(data["cashflow"], mktcap,revenue)
elif fcf_source == "5-year average":
    # New robust path: builds annual series from annual OR quarterly tables
    fcf0 = infer_fcf_avg_from_ticker(ticker.strip(), years=5, market_cap=mktcap,revenue=revenue)
elif fcf_source == "Manual override":
    fcf0 = fcf0_in

with st.expander("Debug (FCF scaling)", expanded=False):
    st.write(f"Market cap (approx): {fmt_money(mktcap)}")
    st.write(f"Revenue (latest): {fmt_money(revenue)}")
    ser_dbg = fcf_series_from_yahoo(ticker.strip())
    if ser_dbg is not None and not ser_dbg.empty:
        raw = ser_dbg.head(5).rename("FCF raw/yr")
        scaled_vals = [scale_if_thousands(float(v), mktcap, revenue)[0] for v in raw.values]
        scaled = pd.Series(scaled_vals, index=raw.index, name="FCF scaled/yr")
        st.dataframe(pd.concat([raw, scaled], axis=1).style.format("${:,.0f}"))
    st.write(f"Base FCF used (after scaling/avg if chosen): {fmt_money(fcf0)}")

cash = infer_cash_plus_sti(data["balancesheet"], mktcap) if cash_in is None else cash_in
debt = infer_total_debt(data["balancesheet"], mktcap) if debt_in is None else debt_in

if fcf_source == "5-year average":
    ser_dbg = fcf_series_from_yahoo(ticker.strip())
    if ser_dbg is None or ser_dbg.empty:
        st.warning("Yahoo has no usable cash-flow for this ticker. Falling back to Net Income proxy or please use Manual override.")
    else:
        st.caption("FCF used = 5-yr avg of annual FCF (OCF − CapEx). Raw series below (pre-scale):")
        st.dataframe(ser_dbg.head(6).to_frame("Annual FCF (raw)").style.format("${:,.0f}"), use_container_width=True)
                    



if mktcap and cash and cash > mktcap:
    st.warning("⚠️ Cash > Market Cap. Yahoo scaling may still be off; consider Manual override.")
if mktcap and debt and debt > 2 * mktcap:
    st.warning("⚠️ Debt ≫ Market Cap. Yahoo may be returning total liabilities; use Manual override.")
if shares and shares > 10_000_000_000:
    st.info("ℹ️ Shares outstanding looks very high. If this seems wrong, use Manual override for shares.")
    
missing = []
if price is None:  missing.append("price")
if shares is None: missing.append("shares")
if fcf0 is None:   missing.append("FCF")
if cash is None:   missing.append("cash")
if debt is None:   missing.append("debt")
if missing:
    st.error(f"Missing from Yahoo: {', '.join(missing)}. Use **Manual override** for the missing fields.")
    st.stop()

if fcf0 <= 0:
    st.warning("⚠️ Base FCF is non-positive. DCF on negative FCF is not meaningful. Consider a multiples model or use a normalized/manual FCF.")

# Build 20-year projections
proj_df, pv_fcfs = dcf_3stage_table(fcf0, g1, g2, g3, r, years=(5, 5, 10))

# Equity value = PV(20y FCF) + Net cash
net_cash = (cash - debt)
equity_value = pv_fcfs + net_cash
intrinsic = equity_value / shares if shares else None
mos_pct = (intrinsic - price) / price * 100 if (intrinsic and price) else None

# Top metrics
m = st.columns(4)
m[0].metric("Current Price", fmt_money(price))
m[1].metric("PV of 20Y FCF", fmt_money(pv_fcfs))
m[2].metric("Net Cash (Cash − Debt)", fmt_money(net_cash))
m[3].metric("Intrinsic Value / Share", fmt_money(intrinsic))
st.caption("Method: Sum of discounted FCFs over 20 years (3 growth bands) + net cash, divided by shares.")

with st.expander("Details & Assumptions", expanded=True):
    c1, c2, c3 = st.columns(3)
    c1.write(f"**Ticker**: `{ticker}`")
    c1.write(f"**Company**: {company_name or '—'}")
    c1.write(f"**Shares Outstanding**: {shares:,.0f}")
    c1.write(f"**Discount Rate (r)**: {r:.2%}")

    c2.write(f"**Base FCF (used)**: {fmt_money(fcf0)}")
    c2.write(f"**Growth Yr 1–5**: {g1:.2%}")
    c2.write(f"**Growth Yr 6–10**: {g2:.2%}")
    c2.write(f"**Growth Yr 11–20**: {g3:.2%}")

    c3.write(f"**Cash + STI**: {fmt_money(cash)}")
    c3.write(f"**Total Debt**: {fmt_money(debt)}")
    if mos_pct is not None:
        c3.write(f"**Discount/Premium vs Price**: {mos_pct:+.1f}%")

st.subheader("Projection Table (Excel-style)")
pretty = proj_df.copy()
pretty["Projected FCF"] = pretty["Projected FCF"].map(fmt_money)
pretty["Discount Factor"] = pretty["Discount Factor"].map(lambda x: f"{x:.2f}")
pretty["Discounted Value"] = pretty["Discounted Value"].map(fmt_money)
st.dataframe(pretty, use_container_width=True)

st.write(f"**Sum of Discounted FCFs (20y):** {fmt_money(pv_fcfs)}")
st.caption("Note: Do **not** use the DCF path for banks/REITs. Use the DDM tab instead.")