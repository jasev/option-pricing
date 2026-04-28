#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║          Monte Carlo Options Pricer  —  main.py              ║
║    CUDA-accelerated · Live Market Data · Greeks Dashboard    ║
╚══════════════════════════════════════════════════════════════╝

Usage
-----
    python main.py                        # interactive mode
    python main.py --ticker AAPL \\
                   --strike 200 \\
                   --expiry 0.5 \\
                   --type call \\
                   --sims 1000000
"""

import argparse
import sys
import os

# ── colour helpers ────────────────────────────────────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    GREEN  = "\033[92m"
    CYAN   = "\033[96m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    BLUE   = "\033[94m"
    MAGENTA= "\033[95m"
    WHITE  = "\033[97m"

def clr(text, *codes):
    return "".join(codes) + str(text) + C.RESET

def banner():
    print(clr("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║      ██████╗ ██████╗ ████████╗██╗ ██████╗ ███╗  ██╗             ║
║     ██╔═══██╗██╔══██╗╚══██╔══╝██║██╔═══██╗████╗ ██║             ║
║     ██║   ██║██████╔╝   ██║   ██║██║   ██║██╔██╗██║             ║
║     ██║   ██║██╔═══╝    ██║   ██║██║   ██║██║╚████║             ║
║     ╚██████╔╝██║        ██║   ██║╚██████╔╝██║ ╚███║             ║
║      ╚═════╝ ╚═╝        ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚══╝             ║
║                                                                  ║
║         Monte Carlo Options Pricer  ·  CUDA Accelerated          ║
╚══════════════════════════════════════════════════════════════════╝
""", C.CYAN, C.BOLD))

def section(title):
    print(f"\n{clr('─' * 66, C.DIM)}")
    print(clr(f"  {title}", C.YELLOW, C.BOLD))
    print(clr('─' * 66, C.DIM))

def kv(label, value, unit="", colour=C.WHITE):
    label_str = clr(f"  {label:<26}", C.DIM)
    value_str = clr(str(value), colour, C.BOLD)
    unit_str  = clr(f" {unit}", C.DIM) if unit else ""
    print(f"{label_str}{value_str}{unit_str}")

def spinner_msg(msg):
    print(clr(f"\n  ⟳  {msg}…", C.CYAN), flush=True)

# ── main logic ────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser(description="Monte Carlo Options Pricer (CUDA)")
    p.add_argument("--ticker", type=str,   default=None)
    p.add_argument("--strike", type=float, default=None)
    p.add_argument("--expiry", type=float, default=None,
                   help="Time to expiry in years (e.g. 0.5 = 6 months)")
    p.add_argument("--type",   type=str,   default=None, choices=["call","put"])
    p.add_argument("--sims",   type=int,   default=1_000_000)
    p.add_argument("--steps",  type=int,   default=252)
    p.add_argument("--vol-period", type=str, default="1y",
                   help="Historical window for volatility: 6mo, 1y, 2y")
    return p.parse_args()


def prompt(label, default=None, cast=str):
    hint = f" [{default}]" if default is not None else ""
    raw  = input(clr(f"  ▸ {label}{hint}: ", C.CYAN)).strip()
    if raw == "" and default is not None:
        return default
    try:
        return cast(raw)
    except ValueError:
        print(clr(f"    Invalid input — please enter a valid {cast.__name__}.", C.RED))
        return prompt(label, default, cast)


def run():
    banner()

    args = get_args()

    # ── imports (after banner so errors appear cleanly) ──
    from pricer      import run_pricer, CUDA_AVAILABLE
    from market_data import fetch_market_data, format_market_data

    # ── Device info ──────────────────────────────────────
    section("Device")
    if CUDA_AVAILABLE:
        try:
            from numba import cuda as _cuda
            dev = _cuda.get_current_device()
            kv("Backend", "CUDA GPU", colour=C.GREEN)
            kv("Device name", dev.name.decode() if isinstance(dev.name, bytes) else dev.name, colour=C.GREEN)
        except Exception:
            kv("Backend", "CUDA GPU", colour=C.GREEN)
    else:
        kv("Backend", "CPU / NumPy  (no CUDA device found)", colour=C.YELLOW)
        print(clr("  ℹ  Install CUDA + numba[cuda] and run on a GPU machine for full acceleration.", C.DIM))

    # ── Market data inputs ────────────────────────────────
    section("Market Data")
    ticker = args.ticker or prompt("Ticker symbol (e.g. AAPL)", "AAPL")
    vol_period = args.vol_period

    spinner_msg(f"Fetching live data for {ticker.upper()}")
    try:
        md = fetch_market_data(ticker, period=vol_period)
    except Exception as e:
        print(clr(f"\n  ✗ Error fetching data: {e}", C.RED))
        sys.exit(1)

    print()
    print(format_market_data(md))

    S     = md["current_price"]
    sigma = md["volatility"]
    r     = md["risk_free_rate"]

    # ── Option parameters ─────────────────────────────────
    section("Option Contract")
    K            = args.strike or prompt(f"Strike price (S={S:.2f})", round(S, 0), float)
    T            = args.expiry or prompt("Time to expiry (years, e.g. 0.5)", 0.25, float)
    option_type  = args.type   or prompt("Option type [call/put]", "call")
    n_sims       = args.sims
    n_steps      = args.steps

    moneyness = "ATM" if abs(S - K) / S < 0.02 else ("ITM" if (option_type=="call" and S>K) or (option_type=="put" and S<K) else "OTM")

    kv("Underlying",    f"${S:.2f}")
    kv("Strike",        f"${K:.2f}  ({moneyness})")
    kv("Expiry",        f"{T:.4f} yrs  ({T*365:.0f} days)")
    kv("Type",          option_type.upper(), colour=C.MAGENTA)
    kv("Volatility σ",  f"{sigma*100:.1f}%")
    kv("Risk-free r",   f"{r*100:.2f}%")
    kv("Simulations",   f"{n_sims:,}")
    kv("Time steps",    f"{n_steps}")

    # ── Run pricer ────────────────────────────────────────
    section("Running Monte Carlo Simulation")
    spinner_msg("Simulating price paths")

    result = run_pricer(S, K, T, r, sigma, option_type, n_sims, n_steps)

    # ── Results ───────────────────────────────────────────
    section("Results")

    price_colour = C.GREEN if result["price"] > 0 else C.RED
    kv("MC Option Price",   f"${result['price']:.4f}",  colour=price_colour)
    kv("95% CI  (±2σ)",     f"±${2*result['std_err']:.4f}")
    kv("Black-Scholes Ref", f"${result['bs_price']:.4f}", colour=C.CYAN)
    kv("MC Error vs BS",    f"${abs(result['price']-result['bs_price']):.4f}")
    kv("Simulations run",   f"{result['n_sims']:,}")
    kv("Device",            result["device"])
    kv("MC elapsed",        f"{result['elapsed_mc']*1000:.1f} ms")

    if result["speedup"] is not None:
        kv("CPU baseline",  f"{result['elapsed_cpu']*1000:.1f} ms")
        kv("GPU Speedup",   f"{result['speedup']:.1f}×", colour=C.GREEN)

    # ── Greeks ────────────────────────────────────────────
    section("Option Greeks")
    g = result["greeks"]
    kv("Delta  Δ", f"{g['delta']:+.4f}",  colour=C.CYAN)
    kv("Gamma  Γ", f"{g['gamma']:+.6f}",  colour=C.CYAN)
    kv("Vega   ν", f"{g['vega']:+.4f}  (per 1% σ move)", colour=C.CYAN)
    kv("Theta  Θ", f"{g['theta']:+.4f}  (per calendar day)", colour=C.CYAN)

    # ── Payoff bar chart (ASCII) ──────────────────────────
    section("Moneyness Summary")
    intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
    time_val  = max(result["price"] - intrinsic, 0)

    bar_total = 40
    if result["price"] > 0:
        iv_frac = intrinsic / result["price"]
        iv_bars = int(iv_frac * bar_total)
        tv_bars = bar_total - iv_bars
        bar = clr("█" * iv_bars, C.GREEN) + clr("░" * tv_bars, C.BLUE)
        mc_price_str = f"${result['price']:.2f}"
        print(f"  [Intrinsic {clr(f'${intrinsic:.2f}',C.GREEN)}] + "
              f"[Time {clr(f'${time_val:.2f}',C.BLUE)}]  =  "
              f"{clr(mc_price_str, C.WHITE, C.BOLD)}")
        print(f"  |{bar}|")
    else:
        print(clr("  Option is worthless at current parameters.", C.RED))

    print(f"\n{clr('  Done.', C.GREEN, C.BOLD)}\n")


if __name__ == "__main__":
    # run from project dir so relative imports work
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run()
