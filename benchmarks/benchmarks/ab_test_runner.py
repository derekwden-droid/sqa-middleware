"""
benchmarks/ab_test_runner.py
=============================
SQA vs Classical SA A/B Benchmark Suite — Rev 1.4
Designed for Ryzen multi-core Docker environment.

Run directly:
    python -m benchmarks.ab_test_runner
    python -m benchmarks.ab_test_runner --trials 200 --spins 64

Or via Docker:
    docker compose -f docker-compose.bench.yml up --build
"""

import os
import sys
import time
import json
import argparse
import threading
from dataclasses import dataclass, asdict
from typing import List, Tuple

import numpy as np
from scipy import stats
import psutil

# Support both package import and direct script run
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sqa_middleware.core import SQAMiddleware, SQAConfig


# ─────────────────────────────────────────────────────────────────────────────
#  Console colours (works on all platforms with ANSI support)
# ─────────────────────────────────────────────────────────────────────────────

class C:
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    DIM    = "\033[2m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"

def _green(s):  return f"{C.GREEN}{s}{C.RESET}"
def _red(s):    return f"{C.RED}{s}{C.RESET}"
def _dim(s):    return f"{C.DIM}{s}{C.RESET}"
def _bold(s):   return f"{C.BOLD}{s}{C.RESET}"
def _cyan(s):   return f"{C.CYAN}{s}{C.RESET}"
def _yellow(s): return f"{C.YELLOW}{s}{C.RESET}"


# ─────────────────────────────────────────────────────────────────────────────
#  Problem generators
# ─────────────────────────────────────────────────────────────────────────────

def generate_sk_instance(n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sherrington-Kirkpatrick (SK) spin-glass: fully-connected graph with
    Gaussian couplings N(0, 1/√N). NP-hard; local minima density ∝ exp(N).
    Canonical benchmark for annealing algorithms.
    """
    rng = np.random.default_rng(seed)
    J   = rng.standard_normal((n, n)) / np.sqrt(n)
    J   = (J + J.T) / 2.0
    np.fill_diagonal(J, 0.0)
    h   = rng.standard_normal(n) * 0.1
    return J.astype(np.float32), h.astype(np.float32)


def generate_routing_instance(n: int, seed: int, sparsity: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sparse graph mimicking real agent decision routing: n variables,
    sparsity fraction of edges active. Uniform biases simulate logit distribution.
    """
    rng  = np.random.default_rng(seed)
    mask = rng.random((n, n)) < sparsity
    J    = rng.standard_normal((n, n)) * mask
    J    = (J + J.T) / 2.0
    np.fill_diagonal(J, 0.0)
    h    = rng.uniform(-1.0, 1.0, n)
    return J.astype(np.float32), h.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Classical SA baseline (single-threaded)
# ─────────────────────────────────────────────────────────────────────────────

def classical_sa_optimize(
    J: np.ndarray,
    h: np.ndarray,
    n_sweeps:  int   = 300,
    T_init:    float = 1.5,
    T_final:   float = 0.05,
) -> dict:
    """
    Single-threaded classical Simulated Annealing.
    Escape counting identical to SQA: improvement after >= 10 stalled sweeps.
    """
    n     = J.shape[0]
    rng   = np.random.default_rng(42)
    spins = rng.choice([-1, 1], size=n).astype(np.float32)
    temps = np.geomspace(T_init, T_final, n_sweeps)

    best_E               = np.inf
    local_minima_escapes = 0
    stall_count          = 0
    t0                   = time.perf_counter()

    for T in temps:
        for i in rng.permutation(n):
            local_field = J[i] @ spins + h[i]
            delta_E     = 2.0 * spins[i] * local_field
            if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
                spins[i] *= -1

        E = float(-0.5 * spins @ J @ spins - h @ spins)
        if E < best_E:
            if stall_count >= 10:
                local_minima_escapes += 1
            best_E      = E
            stall_count = 0
        else:
            stall_count += 1

    return {
        "energy":      best_E,
        "latency_ms":  (time.perf_counter() - t0) * 1000.0,
        "n_sweeps":    n_sweeps,
        "escapes":     local_minima_escapes,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  CPU monitor helper
# ─────────────────────────────────────────────────────────────────────────────

def run_with_cpu_sample(fn, *args, **kwargs):
    """Run fn(*args) while sampling per-core CPU%. Returns (result, cpu_array)."""
    cpu_samples = []
    stop        = threading.Event()

    def monitor():
        while not stop.is_set():
            cpu_samples.append(psutil.cpu_percent(interval=0.01, percpu=True))

    t = threading.Thread(target=monitor, daemon=True)
    t.start()
    result = fn(*args, **kwargs)
    stop.set()
    t.join(timeout=1.0)
    arr = np.array(cpu_samples) if cpu_samples else np.zeros((1, os.cpu_count() or 1))
    return result, arr


# ─────────────────────────────────────────────────────────────────────────────
#  Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrialResult:
    agent:        str
    problem_type: str
    n_vars:       int
    trial:        int
    energy:       float
    latency_ms:   float
    n_sweeps:     int
    escapes:      int
    cpu_pct_avg:  float
    cpu_pct_std:  float


# ─────────────────────────────────────────────────────────────────────────────
#  Benchmark suite
# ─────────────────────────────────────────────────────────────────────────────

class ABBenchmarkSuite:

    def __init__(self, n_trials: int, n_spins: int, n_cores: int, n_sweeps: int):
        self.n_trials = n_trials
        self.n_spins  = n_spins
        self.n_cores  = n_cores
        self.n_sweeps = n_sweeps
        self.results: List[TrialResult] = []

        sqa_config = SQAConfig(
            n_spins      = n_spins,
            n_replicas   = n_cores,
            n_sweeps     = n_sweeps,
            gamma_init   = 4.0,
            gamma_final  = 1e-3,
            temp_init    = 1.5,
            temp_final   = 0.05,
            pin_threads  = True,
        )
        self.sqa = SQAMiddleware(sqa_config)

    def _run_trial(self, prob_type: str, n: int, trial: int) -> Tuple[TrialResult, TrialResult]:
        seed = trial * 1000 + n
        if prob_type == "sk":
            J, h = generate_sk_instance(n, seed)
        else:
            J, h = generate_routing_instance(n, seed)

        # SQA
        sqa_raw, sqa_cpu = run_with_cpu_sample(self.sqa.optimize, J, h)
        sqa_r = TrialResult(
            agent="sqa", problem_type=prob_type, n_vars=n, trial=trial,
            energy=sqa_raw["energy"], latency_ms=sqa_raw["latency_ms"],
            n_sweeps=sqa_raw["n_sweeps"], escapes=sqa_raw["escapes"],
            cpu_pct_avg=float(np.mean(sqa_cpu)),
            cpu_pct_std=float(np.std(sqa_cpu)),
        )

        # Classical SA
        sa_raw, sa_cpu = run_with_cpu_sample(
            classical_sa_optimize, J, h, n_sweeps=self.n_sweeps
        )
        sa_r = TrialResult(
            agent="classical_sa", problem_type=prob_type, n_vars=n, trial=trial,
            energy=sa_raw["energy"], latency_ms=sa_raw["latency_ms"],
            n_sweeps=sa_raw["n_sweeps"], escapes=sa_raw["escapes"],
            cpu_pct_avg=float(np.mean(sa_cpu)),
            cpu_pct_std=float(np.std(sa_cpu)),
        )
        return sqa_r, sa_r

    def _print_header(self):
        w = 104
        print("\n" + "═" * w)
        print(_bold(_cyan(
            f"  {'PROBLEM':<16} {'N':>4}  "
            f"{'SQA Energy':>18}  {'SA Energy':>18}  "
            f"{'Gain':>7}  {'SQA ms':>9}  {'SA ms':>9}  "
            f"{'Esc×':>6}  {'p-val':>8}"
        )))
        print("─" * w)

    def _print_row(
        self, prob: str, n: int,
        sqa_E, sa_E, sqa_L, sa_L, sqa_esc, sa_esc
    ):
        mean_gain = (np.mean(sa_E) - np.mean(sqa_E)) / abs(np.mean(sa_E)) * 100
        _, p_val  = stats.ttest_ind(sqa_E, sa_E, equal_var=False)
        esc_ratio = (np.mean(sqa_esc) / max(np.mean(sa_esc), 0.01))

        gain_str = f"{mean_gain:+.1f}%"
        gain_fmt = _green(gain_str) if (mean_gain > 0 and p_val < 0.05) else _dim(gain_str)

        p_str = f"{p_val:.4f}"
        p_fmt = _green(p_str) if p_val < 0.05 else _dim(p_str)

        esc_str = f"{esc_ratio:.1f}×"
        esc_fmt = _green(esc_str) if esc_ratio > 1.5 else _dim(esc_str)

        print(
            f"  {prob:<16} {n:>4}  "
            f"{np.mean(sqa_E):>8.2f}±{np.std(sqa_E):<7.2f}  "
            f"{np.mean(sa_E):>8.2f}±{np.std(sa_E):<7.2f}  "
            f"{gain_fmt:>7}  "
            f"{np.mean(sqa_L):>7.1f}ms  "
            f"{np.mean(sa_L):>7.1f}ms  "
            f"{esc_fmt:>6}  "
            f"{p_fmt:>8}"
        )

    def run(self):
        configs = [
            ("sk",      self.n_spins),
            ("routing", self.n_spins),
        ]

        phys  = psutil.cpu_count(logical=False) or self.n_cores
        logic = psutil.cpu_count(logical=True)  or self.n_cores

        print("\n" + "═" * 60)
        print(_bold("  SQA Middleware Benchmark — Rev 1.4"))
        print("─" * 60)
        print(f"  Cores (physical / logical) : {phys} / {logic}")
        print(f"  SQA replicas (P)           : {self.n_cores}")
        print(f"  Problem size (N)            : {self.n_spins} spins")
        print(f"  Trials per config           : {self.n_trials}")
        print(f"  Sweeps per call             : {self.n_sweeps}")
        print("═" * 60)

        self._print_header()

        for prob_type, n in configs:
            sqa_E, sa_E   = [], []
            sqa_L, sa_L   = [], []
            sqa_esc, sa_esc = [], []

            for trial in range(self.n_trials):
                # Progress dot every 10 trials
                if trial % 10 == 0:
                    print(
                        f"  {_dim(f'[{prob_type.upper()} N={n}]')} "
                        f"trial {trial+1}/{self.n_trials}...",
                        end="\r", flush=True
                    )

                sqa_r, sa_r = self._run_trial(prob_type, n, trial)
                self.results.extend([sqa_r, sa_r])

                sqa_E.append(sqa_r.energy);    sa_E.append(sa_r.energy)
                sqa_L.append(sqa_r.latency_ms); sa_L.append(sa_r.latency_ms)
                sqa_esc.append(sqa_r.escapes); sa_esc.append(sa_r.escapes)

            print(" " * 60, end="\r")  # Clear progress line
            self._print_row(
                prob_type, n,
                sqa_E, sa_E, sqa_L, sa_L, sqa_esc, sa_esc
            )

        print("─" * 104)
        print(_dim("  Green = statistically significant (p < 0.05) · Esc× = SQA escapes ÷ SA escapes"))
        print()

    def save(self, path: str = "benchmark_results.json"):
        with open(path, "w") as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
        print(f"  Results saved → {path}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SQA vs Classical SA A/B Benchmark"
    )
    parser.add_argument("--trials",   type=int, default=500,
                        help="Trials per configuration (default: 500)")
    parser.add_argument("--spins",    type=int, default=64,
                        help="Problem size N (default: 64)")
    parser.add_argument("--replicas", type=int, default=0,
                        help="SQA replicas P; 0 = auto-detect physical cores")
    parser.add_argument("--sweeps",   type=int, default=300,
                        help="MC sweeps per call (default: 300)")
    parser.add_argument("--fast",     action="store_true",
                        help="Quick run: 100 trials, N=32, 150 sweeps")
    parser.add_argument("--output",   type=str, default="benchmark_results.json",
                        help="Output JSON file path")
    args = parser.parse_args()

    if args.fast:
        args.trials, args.spins, args.sweeps = 100, 32, 150

    n_cores = args.replicas if args.replicas > 0 else (
        psutil.cpu_count(logical=False) or os.cpu_count() or 4
    )

    suite = ABBenchmarkSuite(
        n_trials = args.trials,
        n_spins  = args.spins,
        n_cores  = n_cores,
        n_sweeps = args.sweeps,
    )
    suite.run()
    suite.save(args.output)


if __name__ == "__main__":
    main()
