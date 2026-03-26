"""
sqa_middleware/core.py
======================
Simulated Quantum Annealing Middleware — Rev 1.4
Double-buffered Trotter decomposition on commodity multi-core CPUs.

Mathematical foundation: Suzuki-Trotter decomposition of the transverse-field
Ising model. Each Trotter replica maps 1:1 to a physical CPU core via OS thread
affinity. Inter-replica coupling is exchanged via double-buffered shared-memory
arrays — no locks in the hot path, barrier synchronization only.

Usage:
    from sqa_middleware.core import SQAMiddleware, SQAConfig
    import numpy as np

    config = SQAConfig(n_spins=64, n_replicas=12)
    sqa    = SQAMiddleware(config)
    result = sqa.optimize(J, h)   # J: [N,N] coupling, h: [N] bias
    print(result["energy"], result["spins"], result["escapes"])
"""

import os
import math
import time
import threading
from dataclasses import dataclass
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  Configuration dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SQAConfig:
    """All tunable parameters for one SQA optimizer instance."""
    n_spins:         int            # N: number of decision variables
    n_replicas:      int   = 12     # P: set to physical core count
    n_sweeps:        int   = 300    # MC sweeps per optimize() call
    temp_init:       float = 1.5    # T_0: initial temperature
    temp_final:      float = 0.05   # T_f: final temperature
    gamma_init:      float = 4.0    # Γ_0: initial transverse field
    gamma_final:     float = 1e-3   # Γ_f: final transverse field
    pin_threads:     bool  = True   # Enable os.sched_setaffinity (Linux only)
    seed: Optional[int]    = None   # RNG seed (None = random)


# ─────────────────────────────────────────────────────────────────────────────
#  Core middleware class
# ─────────────────────────────────────────────────────────────────────────────

class SQAMiddleware:
    """
    Simulated Quantum Annealing Middleware.

    Thread-safety model
    -------------------
    Two threading.Barrier(P) calls per sweep epoch:
      Barrier 1 — after all P workers write their updated spins to write_buf.
                  Guarantees write_buf is fully committed before any read.
      Barrier 2 — after replica-0 flips the active_buf pointer.
                  Guarantees the flip is visible to all threads before the
                  next epoch's read_buf assignment.

    No mutexes anywhere in the hot path. The per-replica best-tracking uses
    purely local variables; a single min() reduction runs after the thread
    pool exits (no shared state during execution).
    """

    def __init__(self, config: SQAConfig):
        self.cfg = config
        self.P   = config.n_replicas
        self.N   = config.n_spins
        self.barrier = threading.Barrier(self.P)
        self.rng     = np.random.default_rng(config.seed)

        # Pre-compute full geometric annealing schedules once at init.
        # Γ decays faster than T to enforce quantum→classical phase transition
        # before thermal convergence completes.
        self.gamma_schedule = np.geomspace(
            config.gamma_init, config.gamma_final, config.n_sweeps
        )
        self.temp_schedule = np.geomspace(
            config.temp_init, config.temp_final, config.n_sweeps
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _compute_j_tunnel(self, gamma: float, temp: float) -> float:
        """
        Suzuki-Trotter inter-replica coupling constant.

            J_tunnel = (P·T/2) · ln( coth(Γ / (P·T)) )

        NumPy's cosh/sinh handle the large-x asymptote (coth→1) correctly
        via IEEE 754 overflow rules — no manual branch needed.
        The 1e-12 epsilon guards only the x≈0 degenerate case (Γ→0).
        """
        x = gamma / (self.P * temp)
        return (self.P * temp / 2.0) * np.log(
            np.cosh(x) / np.sinh(x + 1e-12)
        )

    def _compute_energy(
        self, spins: np.ndarray, J: np.ndarray, h: np.ndarray
    ) -> float:
        """H = -0.5·σᵀJσ - hᵀσ  (fully vectorized, no Python loop)"""
        return float(-0.5 * spins @ J @ spins - h @ spins)

    def _replica_worker(
        self,
        replica_id:       int,
        buf_a:            np.ndarray,   # Double-buffer A [P, N]
        buf_b:            np.ndarray,   # Double-buffer B [P, N]
        active_buf:       list,         # [0] = index of current READ buffer
        J:                np.ndarray,   # [N, N] coupling matrix (read-only)
        h:                np.ndarray,   # [N]    bias vector     (read-only)
        per_replica_best: list,         # Write-back slot for this replica
        rng_seed:         int,
    ):
        """
        Double-buffered replica worker.

        Race-condition proof: adjacent replica states are ALWAYS read from
        the committed "read" buffer (fully written + barrier-fenced in the
        PREVIOUS epoch). The current worker writes only to the "write" buffer
        for its own row. No other thread writes the same row in write_buf.

        Escape counting: identical logic to classical SA baseline.
        An escape is defined as an energy improvement that occurs after
        >= 10 consecutive stalled sweeps (stall = no improvement).
        This makes SQA and SA escape-rate columns directly comparable.
        """
        # Optional: pin this thread to a specific physical core
        if self.cfg.pin_threads:
            try:
                os.sched_setaffinity(0, {replica_id % os.cpu_count()})
            except (AttributeError, OSError):
                pass  # Non-Linux or insufficient privileges — silently skip

        local_rng = np.random.default_rng(rng_seed)
        p      = replica_id
        p_prev = (p - 1) % self.P   # Periodic boundary: replica P+1 ≡ replica 1
        p_next = (p + 1) % self.P
        bufs   = [buf_a, buf_b]

        # Per-replica tracking — all local, zero shared state
        local_best_energy = np.inf
        local_best_spins  = None
        local_escapes     = 0   # Incremented when energy improves after ≥10-sweep stall
        stall_count       = 0   # Consecutive sweeps without improvement

        for sweep_idx in range(self.cfg.n_sweeps):
            gamma = self.gamma_schedule[sweep_idx]
            temp  = self.temp_schedule[sweep_idx]
            j_tun = self._compute_j_tunnel(gamma, temp)

            # ── Identify read/write buffers for this epoch ────────────────
            read_buf  = bufs[active_buf[0]]       # Committed previous-sweep (READ)
            write_buf = bufs[1 - active_buf[0]]   # Fresh next-sweep buffer  (WRITE)

            # ── Snapshot adjacent replicas from the COMMITTED read buffer ─
            # Taken BEFORE the spin loop. No other thread writes to read_buf
            # during this epoch — it is the "previous" buffer by definition.
            prev_spins = read_buf[p_prev].copy()   # σ^{p-1} from last epoch
            next_spins = read_buf[p_next].copy()   # σ^{p+1} from last epoch
            spin_p     = read_buf[p].copy()        # Own state from last epoch

            # ── Monte Carlo sweep over all N spins (hot path) ─────────────
            for i in local_rng.permutation(self.N):
                # Intra-replica field: Σ_j J_ij·σ_j + h_i
                # Uses evolving spin_p (within-sweep updates are intentional)
                local_field = J[i] @ spin_p + h[i]

                # Inter-replica tunneling field — uses FROZEN prev/next snapshots
                tunnel_field = j_tun * (prev_spins[i] + next_spins[i])

                delta_E = 2.0 * spin_p[i] * (local_field + tunnel_field)

                # Metropolis acceptance criterion
                if delta_E < 0.0 or local_rng.random() < np.exp(-delta_E / temp):
                    spin_p[i] *= -1

            # ── Write result to NEXT buffer (this row is exclusively ours) ─
            write_buf[p] = spin_p

            # ── Barrier 1: all P replicas finish writing ──────────────────
            self.barrier.wait()

            # Replica 0 flips the active buffer pointer (post-barrier, no lock)
            if p == 0:
                active_buf[0] = 1 - active_buf[0]

            # ── Per-epoch best tracking + stall/escape counting ───────────
            energy = self._compute_energy(spin_p, J, h)
            if energy < local_best_energy:
                if stall_count >= 10:       # Improvement after stall = escape event
                    local_escapes += 1
                local_best_energy = energy
                local_best_spins  = spin_p.copy()
                stall_count       = 0
            else:
                stall_count += 1

            # ── Barrier 2: ensure buffer flip is visible before next read ─
            self.barrier.wait()

        # Write-back: only once, after the loop — no contention possible
        per_replica_best[p] = (local_best_energy, local_best_spins, local_escapes)

    # ── Public API ────────────────────────────────────────────────────────────

    def optimize(self, J: np.ndarray, h: np.ndarray) -> dict:
        """
        Main entry point. Called by the AI agent once per decision tick.

        Args:
            J : [N, N] symmetric float32 coupling matrix (QUBO/Ising weights)
            h : [N]    float32 local bias vector (from agent logit scores)

        Returns dict with keys:
            spins        — [N] int8 optimal spin configuration {-1, +1}
            energy       — float, Ising Hamiltonian energy of best solution
            escapes      — int, total local-minima escape events across replicas
            n_sweeps     — int
            n_replicas   — int
            latency_ms   — float, wall-clock time for this call
            gamma_schedule — (gamma_init, gamma_final) tuple
        """
        assert J.shape == (self.N, self.N), f"J must be [{self.N},{self.N}], got {J.shape}"
        assert h.shape == (self.N,),        f"h must be [{self.N}], got {h.shape}"
        assert np.allclose(J, J.T, atol=1e-5), "J must be symmetric"

        J = J.astype(np.float32, copy=True)
        h = h.astype(np.float32, copy=True)
        np.fill_diagonal(J, 0.0)   # No self-coupling in Ising model

        # ── Double-buffer allocation ──────────────────────────────────────
        # buf_a: initial random spin states (READ buffer for epoch 0)
        # buf_b: zeroed write buffer for epoch 0 results
        # active_buf[0]: index of current READ-side buffer (0=A, 1=B)
        # Replica 0 flips active_buf[0] at Barrier 1 each epoch.
        buf_a      = self.rng.choice([-1, 1], size=(self.P, self.N)).astype(np.float32)
        buf_b      = np.empty_like(buf_a)
        active_buf = [0]

        per_replica_best = [None] * self.P
        t_start = time.perf_counter()

        with ThreadPoolExecutor(max_workers=self.P) as pool:
            seeds   = self.rng.integers(0, 2**31, size=self.P)
            futures = [
                pool.submit(
                    self._replica_worker,
                    p, buf_a, buf_b, active_buf, J, h,
                    per_replica_best, int(seeds[p])
                )
                for p in range(self.P)
            ]
            for f in futures:
                f.result()  # Re-raises any exception from worker threads

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        # ── Final reduction (lock-free: all workers have exited) ──────────
        best_energy, best_config, _ = min(per_replica_best, key=lambda x: x[0])
        total_escapes = sum(r[2] for r in per_replica_best)

        return {
            "spins":          best_config.astype(np.int8),
            "energy":         best_energy,
            "escapes":        total_escapes,
            "n_sweeps":       self.cfg.n_sweeps,
            "n_replicas":     self.P,
            "latency_ms":     round(latency_ms, 3),
            "gamma_schedule": (self.cfg.gamma_init, self.cfg.gamma_final),
        }
