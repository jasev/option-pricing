"""
market_data.py
==============
Fetches live market data from Yahoo Finance.
Computes annualised historical volatility from log-returns.
"""

import numpy as np
import yfinance as yf
from datetime import datetime

# Approximate US 10-yr risk-free rate (update or swap with FRED API if desired)
DEFAULT_RISK_FREE_RATE = 0.0525


def fetch_market_data(ticker: str, period: str = "1y") -> dict:
    """
    Download OHLCV data for `ticker` and compute key option inputs.

    Parameters
    ----------
    ticker : str   e.g. "AAPL", "TSLA", "SPY"
    period : str   lookback window for volatility e.g. "6mo", "1y", "2y"

    Returns
    -------
    dict with: ticker, current_price, volatility, dividend_yield,
               risk_free_rate, data_period, last_updated
    """
    t = yf.Ticker(ticker)
    hist = t.history(period=period)

    if hist.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. "
                         "Check the symbol and your internet connection.")

    closes = hist["Close"].dropna().values
    if len(closes) < 20:
        raise ValueError(f"Insufficient price history ({len(closes)} days) "
                         "to compute reliable volatility.")

    # Annualised historical volatility (σ = std of log-returns × √252)
    log_returns = np.log(closes[1:] / closes[:-1])
    volatility  = float(np.std(log_returns) * np.sqrt(252))

    current_price   = float(closes[-1])

    # Dividend yield from metadata (fallback to 0)
    info = t.info
    div_yield = float(info.get("dividendYield") or 0.0)

    return {
        "ticker":         ticker.upper(),
        "current_price":  current_price,
        "volatility":     volatility,
        "dividend_yield": div_yield,
        "risk_free_rate": DEFAULT_RISK_FREE_RATE,
        "data_period":    period,
        "n_obs":          len(closes),
        "last_updated":   datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def format_market_data(md: dict) -> str:
    """Pretty-print market data summary."""
    lines = [
        f"  Ticker         : {md['ticker']}",
        f"  Current Price  : ${md['current_price']:.2f}",
        f"  Hist. Volatility: {md['volatility']*100:.1f}% (ann.)",
        f"  Dividend Yield : {md['dividend_yield']*100:.2f}%",
        f"  Risk-Free Rate : {md['risk_free_rate']*100:.2f}%",
        f"  Data Period    : {md['data_period']}  ({md['n_obs']} sessions)",
        f"  Last Updated   : {md['last_updated']}",
    ]
    return "\n".join(lines)
