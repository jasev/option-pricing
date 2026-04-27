"""
Monte Carlo Options Pricer
==========================
GPU-accelerated (CUDA via Numba) with automatic CPU fallback.
Prices European Call/Put options using Geometric Brownian Motion.
"""

import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

# ── CUDA detection ────────────────────────────────────────────────────────────
try:
    from numba import cuda
    from numba import float32
    import numba as nb
    CUDA_AVAILABLE = cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False

# ── Black-Scholes analytical reference ───────────────────────────────────────
from scipy.stats import norm

def black_scholes(S, K, T, r, sigma, option_type="call"):
    """Closed-form Black-Scholes price for reference."""
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

# ── Greeks via finite differences ────────────────────────────────────────────
def compute_greeks(S, K, T, r, sigma, option_type="call"):
    eps_s = S * 0.01
    eps_v = 0.001
    eps_t = 1 / 365.0

    price  = black_scholes(S, K, T, r, sigma, option_type)
    price_up   = black_scholes(S + eps_s, K, T, r, sigma, option_type)
    price_down = black_scholes(S - eps_s, K, T, r, sigma, option_type)
    price_vol_up = black_scholes(S, K, T, r, sigma + eps_v, option_type)
    price_t_down = black_scholes(S, K, max(T - eps_t, 1e-6), r, sigma, option_type)

    delta = (price_up - price_down) / (2 * eps_s)
    gamma = (price_up - 2 * price + price_down) / (eps_s ** 2)
    vega  = (price_vol_up - price) / eps_v / 100  # per 1% vol move
    theta = (price_t_down - price) / eps_t / 365   # per calendar day

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}

# ── CUDA Kernel ───────────────────────────────────────────────────────────────
if CUDA_AVAILABLE:
    from numba import cuda
    from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float32

    @cuda.jit
    def mc_kernel(rng_states, S, K, T, r, sigma, results, n_steps):
        """Each GPU thread simulates one price path."""
        idx = cuda.grid(1)
        if idx >= results.shape[0]:
            return

        dt = T / n_steps
        price = S
        for _ in range(n_steps):
            z = xoroshiro128p_normal_float32(rng_states, idx)
            price *= (1.0 + r * dt + sigma * (dt ** 0.5) * z)

        results[idx] = price


def price_cuda(S, K, T, r, sigma, option_type, n_sims, n_steps=252):
    """Run Monte Carlo on GPU using CUDA."""
    threads_per_block = 256
    blocks = (n_sims + threads_per_block - 1) // threads_per_block
    actual_sims = blocks * threads_per_block

    rng_states = create_xoroshiro128p_states(actual_sims, seed=42)
    d_results = cuda.device_array(actual_sims, dtype=np.float32)

    t0 = time.perf_counter()
    mc_kernel[blocks, threads_per_block](
        rng_states, np.float32(S), np.float32(K),
        np.float32(T), np.float32(r), np.float32(sigma),
        d_results, np.int32(n_steps)
    )
    cuda.synchronize()
    elapsed = time.perf_counter() - t0

    final_prices = d_results.copy_to_host()
    if option_type == "call":
        payoffs = np.maximum(final_prices - K, 0)
    else:
        payoffs = np.maximum(K - final_prices, 0)

    price = np.exp(-r * T) * np.mean(payoffs)
    std_err = np.exp(-r * T) * np.std(payoffs) / np.sqrt(actual_sims)
    return price, std_err, elapsed, actual_sims

# ── CPU Monte Carlo ────────────────────────────────────────────────────────────
def price_cpu(S, K, T, r, sigma, option_type, n_sims, n_steps=252):
    """Vectorised NumPy Monte Carlo (CPU baseline)."""
    rng = np.random.default_rng(42)
    dt = T / n_steps

    t0 = time.perf_counter()
    Z = rng.standard_normal((n_sims, n_steps)).astype(np.float32)
    log_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    final_prices = S * np.exp(log_returns.sum(axis=1))

    if option_type == "call":
        payoffs = np.maximum(final_prices - K, 0)
    else:
        payoffs = np.maximum(K - final_prices, 0)

    elapsed = time.perf_counter() - t0
    price   = np.exp(-r * T) * np.mean(payoffs)
    std_err = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_sims)
    return price, std_err, elapsed, n_sims


# ── Public interface ──────────────────────────────────────────────────────────
def run_pricer(S, K, T, r, sigma, option_type="call", n_sims=1_000_000, n_steps=252):
    """
    Price an option via Monte Carlo.
    Automatically uses GPU if CUDA is available, falls back to CPU.

    Returns
    -------
    dict with keys: price, std_err, bs_price, greeks,
                    elapsed_mc, device, speedup (if both ran)
    """
    bs_price = black_scholes(S, K, T, r, sigma, option_type)
    greeks   = compute_greeks(S, K, T, r, sigma, option_type)

    if CUDA_AVAILABLE:
        # Warm-up pass (JIT compile)
        price_cuda(S, K, T, r, sigma, option_type, n_sims=1024, n_steps=n_steps)
        mc_price, std_err, elapsed_gpu, actual_sims = price_cuda(
            S, K, T, r, sigma, option_type, n_sims, n_steps)
        # CPU run for benchmark comparison
        _, _, elapsed_cpu, _ = price_cpu(
            S, K, T, r, sigma, option_type, min(n_sims, 500_000), n_steps)
        speedup = elapsed_cpu / elapsed_gpu
        device  = "CUDA GPU"
    else:
        mc_price, std_err, elapsed_gpu, actual_sims = price_cpu(
            S, K, T, r, sigma, option_type, n_sims, n_steps)
        elapsed_cpu = elapsed_gpu
        speedup = None
        device  = "CPU (NumPy)"

    return {
        "price":        mc_price,
        "std_err":      std_err,
        "bs_price":     bs_price,
        "greeks":       greeks,
        "elapsed_mc":   elapsed_gpu,
        "elapsed_cpu":  elapsed_cpu,
        "device":       device,
        "n_sims":       actual_sims,
        "speedup":      speedup,
    }
