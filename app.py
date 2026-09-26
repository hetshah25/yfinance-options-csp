import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, date
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Cash-Secured Put Screener", layout="wide")

# ---------------- Model functions ----------------

LEVERAGE = {
    "SOXL": 3, "SOXS": 3, "TQQQ": 3, "SQQQ": 3, "SPXL": 3,
    "SPXS": 3, "TNA": 3, "TZA": 3, "LABU": 3, "LABD": 3,
    "QLD": 2, "SSO": 2, "UVXY": 2,
}

@st.cache_data(ttl=3600)
def get_risk_free_rate():
    try:
        irx = yf.Ticker("^IRX").history(period="5d")["Close"].iloc[-1]
        return irx / 100
    except Exception:
        return 0.045


def get_dividend_yield(tk):
    try:
        dy = tk.info.get("dividendYield", 0) or 0
        if dy > 1:
            dy = dy / 100
        return min(max(dy, 0), 0.15)
    except Exception:
        return 0.0


def historical_volatility(close_prices, lookback_days=30):
    if len(close_prices) < lookback_days + 1:
        return np.nan
    log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
    return log_returns[-lookback_days:].std() * np.sqrt(252)


def days_to_expiry(expiry_str):
    exp_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
    return (exp_date - date.today()).days


def get_next_earnings_date(tk):
    try:
        cal = tk.calendar
        earnings_dates = cal.get("Earnings Date") if cal else None
        if earnings_dates:
            future = [d for d in earnings_dates if d >= date.today()]
            return min(future) if future else None
    except Exception:
        pass
    try:
        ed = tk.get_earnings_dates(limit=8)
        if ed is not None and not ed.empty:
            idx_dates = [d.date() if hasattr(d, "date") else d for d in ed.index]
            future = [d for d in idx_dates if d >= date.today()]
            if future:
                return min(future)
    except Exception:
        pass
    return None


def bs_put_metrics(S, K, T, r, q, sigma):
    if T <= 0 or sigma is None or sigma <= 0 or np.isnan(sigma):
        return np.nan, np.nan
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = np.exp(-q * T) * (norm.cdf(d1) - 1)
    win_prob = norm.cdf(d2)
    return delta, win_prob


@st.cache_data(ttl=900)
def scan_ticker_for_csp(ticker_symbol, risk_free_rate, min_dte, max_dte,
                          win_prob_min, win_prob_max, min_oi, max_spread_pct, top_n):
    tk = yf.Ticker(ticker_symbol)

    try:
        hist = tk.history(period="1y")
        spot = hist["Close"].iloc[-1]
    except Exception:
        return pd.DataFrame(), None

    hv = historical_volatility(hist["Close"], lookback_days=30)
    low_52w = hist["Close"].min()
    sma_50 = hist["Close"].tail(50).mean()
    dividend_yield = get_dividend_yield(tk)
    next_earnings = get_next_earnings_date(tk)
    leverage_multiplier = LEVERAGE.get(ticker_symbol, 1)

    candidates = []

    try:
        expirations = tk.options
    except Exception:
        return pd.DataFrame(), spot

    for exp in expirations:
        dte = days_to_expiry(exp)
        if dte < min_dte or dte > max_dte:
            continue

        try:
            chain = tk.option_chain(exp)
        except Exception:
            continue

        puts = chain.puts.copy()
        if puts.empty:
            continue

        T = dte / 365.0
        exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
        earnings_in_window = next_earnings is not None and date.today() <= next_earnings <= exp_date

        for _, row in puts.iterrows():
            K = row["strike"]
            iv = row["impliedVolatility"]
            bid = row.get("bid", 0) or 0
            ask = row.get("ask", 0) or 0
            last = row.get("lastPrice", 0) or 0
            oi = row.get("openInterest", 0)
            oi = 0 if pd.isna(oi) else oi
            vol = row.get("volume", 0)
            vol = 0 if pd.isna(vol) else vol

            premium = bid if bid > 0 else last
            if premium <= 0 or K >= spot:
                continue
            # Yahoo's free feed frequently reports openInterest as 0/blank even
            # when the contract is actively trading, so fall back to volume.
            liquidity = oi if oi > 0 else vol
            if liquidity < min_oi:
                continue

            spread_pct = np.nan
            if ask > 0:
                spread_pct = (ask - bid) / ask * 100
                if spread_pct > max_spread_pct:
                    continue

            delta, win_prob = bs_put_metrics(spot, K, T, risk_free_rate, dividend_yield, iv)
            if np.isnan(win_prob):
                continue

            win_prob_pct = win_prob * 100
            if win_prob_pct < win_prob_min or win_prob_pct > win_prob_max:
                continue

            yield_pct = (premium / K) * 100
            annualized_yield = yield_pct * (365 / dte)
            score = (win_prob_pct / 100) * annualized_yield / leverage_multiplier
            iv_hv_ratio = (iv / hv) if hv and not np.isnan(hv) and hv > 0 else np.nan
            exit_target_70pct = premium * 0.3
            exit_target_50pct = premium * 0.5

            candidates.append({
                "Ticker": ticker_symbol,
                "Spot Price": round(spot, 2),
                "Expiration": exp,
                "DTE": dte,
                "Strike": K,
                "Premium (Bid)": round(premium, 2),
                "IV %": round(iv * 100, 1),
                "HV % (30d)": round(hv * 100, 1) if not np.isnan(hv) else np.nan,
                "IV/HV": round(iv_hv_ratio, 2) if not np.isnan(iv_hv_ratio) else np.nan,
                "Delta": round(delta, 3),
                "Est. Win Prob %": round(win_prob_pct, 1),
                "Yield %": round(yield_pct, 2),
                "Annualized Yield %": round(annualized_yield, 1),
                "Score": round(score, 2),
                "Lev": leverage_multiplier,
                "Breakeven": round(K - premium, 2),
                "Capital Req. $": round(K * 100, 0),
                "Premium $": round(premium * 100, 0),
                "Exit Target ($)": f"${exit_target_70pct:.2f}–${exit_target_50pct:.2f}",
                "Next Earnings": next_earnings.strftime("%Y-%m-%d") if next_earnings else "—",
                "Earnings Alert": "⚠️ In Window" if earnings_in_window else "",
                "Open Interest": int(oi),
                "Volume": int(vol),
                "Spread %": round(spread_pct, 1) if not np.isnan(spread_pct) else np.nan,
                "50D SMA": round(sma_50, 2),
                "52W Low": round(low_52w, 2),
            })

    if not candidates:
        return pd.DataFrame(), spot

    df = pd.DataFrame(candidates)
    df = df.sort_values(by="Score", ascending=False)
    return df.head(top_n).reset_index(drop=True), spot


# ---------------- UI ----------------

st.title("Cash-Secured Put Screener")
st.caption("Live yfinance data. Not financial advice — verify pricing and liquidity with your broker before trading.")

with st.sidebar:
    st.header("Settings")
    tickers_input = st.text_input("Tickers (comma-separated)", "NVDA, SOXL, AAPL, AMD, INTC")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    st.subheader("Expiration window")
    min_dte, max_dte = st.slider("Days to expiration", 1, 90, (30, 45))

    st.subheader("Win probability target")
    win_prob_min, win_prob_max = st.slider("Win probability %", 50, 99, (75, 85))

    st.subheader("Liquidity filters")
    min_oi = st.number_input(
        "Minimum open interest",
        min_value=0, value=300, step=5,
        help="Falls back to volume when Yahoo reports open interest as 0/blank for a contract.",
    )
    max_spread_pct = st.number_input("Max bid-ask spread %", min_value=1, value=15, step=1)

    top_n = st.number_input("Top N per ticker", min_value=1, max_value=10, value=3)

    run_button = st.button("Run scan", type="primary", use_container_width=True)

if run_button:
    if not tickers:
        st.warning("Add at least one ticker.")
    else:
        rf_rate = get_risk_free_rate()
        st.info(f"Risk-free rate in use (13-week T-bill): {rf_rate*100:.2f}%")

        all_results = []
        progress = st.progress(0.0, text="Scanning...")

        for i, ticker in enumerate(tickers):
            df, spot = scan_ticker_for_csp(
                ticker, rf_rate, min_dte, max_dte,
                win_prob_min, win_prob_max, min_oi, max_spread_pct, top_n
            )
            if df.empty:
                st.warning(f"{ticker}: no candidates found in the current window (spot: {spot}).")
            else:
                all_results.append(df)
            progress.progress((i + 1) / len(tickers), text=f"Scanned {ticker}")

        progress.empty()

        if all_results:
            summary = pd.concat(all_results, ignore_index=True)

            display_cols = ["Expiration", "DTE", "Strike", "Premium (Bid)", "Delta",
                             "Est. Win Prob %", "Annualized Yield %", "Score", "Lev", "IV/HV",
                             "Exit Target ($)", "Next Earnings", "Earnings Alert",
                             "Breakeven", "Capital Req. $", "Open Interest", "Spread %"]

            st.subheader("All candidates (sorted by Score within each ticker)")
            st.dataframe(summary[["Ticker"] + display_cols], use_container_width=True, hide_index=True)

            st.divider()
            st.subheader("Per-ticker breakdown")

            for ticker in tickers:
                sub = summary[summary["Ticker"] == ticker]
                if sub.empty:
                    continue
                st.markdown(
                    f"**{ticker}** — spot ${sub['Spot Price'].iloc[0]} · "
                    f"50D SMA ${sub['50D SMA'].iloc[0]} · 52W Low ${sub['52W Low'].iloc[0]}"
                )
                st.dataframe(sub[display_cols], use_container_width=True, hide_index=True)

            with st.expander("How to read this"):
                st.markdown("""
- **Est. Win Prob %** — model probability the put expires worthless and you keep the full premium. 70-85% is the conventional target band.
- **IV/HV** — implied vol vs. 30-day realized vol. Above ~1.2 generally means premium is rich relative to how the stock has actually been moving.
- **Score** — win probability times annualized yield, divided by the underlying's leverage multiplier (see **Lev**). Use it to shortlist, then check win probability itself before deciding — a high score can come from either a genuinely strong trade or a risky one with a fat premium masking the risk.
- **Lev** — leverage multiplier applied to the underlying (e.g. 3 for SOXL/TQQQ, 2 for SSO/QLD, 1 for unleveraged names). Score is divided by this so leveraged ETFs aren't overrated relative to their real risk.
- **Exit Target ($)** — the buy-to-close premium range that locks in 50-70% of max profit (70% target price is the lower number, 50% target is the higher number). A common approach is closing in this range rather than holding to expiration, since late-stage premium decays slowest relative to the tail risk of holding on.
- **Next Earnings / Earnings Alert** — the ticker's next known earnings date, with a ⚠️ flag if it falls inside the option's expiration window. A high-scoring put with this flag set may be an earnings bet in disguise — check the date before trading.
                """)
else:
    st.info("Set your filters in the sidebar and click **Run scan** to pull live options data.")
