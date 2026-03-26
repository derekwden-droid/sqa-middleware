"""
sqa_middleware/agent_adapter.py
================================
Thin adapter that translates between agent-native data formats and the
SQA Ising model representation. Zero-copy where possible.

Usage:
    adapter = AgentSQAAdapter(n_decisions=64, coupling_matrix=J, n_cores=12)
    actions, latency_ms, energy = adapter.optimize_action(logit_scores)
"""

import os
import numpy as np
from .core import SQAMiddleware, SQAConfig


class AgentSQAAdapter:
    """
    Connects any AI agent that emits logit/confidence scores to the SQA
    optimizer. Handles the logit→Ising-bias translation and binary decode.

    The mapping is:
        h_i = -log(confidence_i)   (high confidence → low field energy)
        σ_i ∈ {-1, +1}            → action mask ∈ {0, 1} on decode
    """

    def __init__(
        self,
        n_decisions:     int,
        coupling_matrix: np.ndarray,
        n_cores:         int = 0,      # 0 = auto-detect physical cores
    ):
        """
        Args:
            n_decisions:     Number of binary decision variables (N).
            coupling_matrix: [N, N] symmetric matrix encoding pairwise
                             decision correlations (pre-trained or identity).
            n_cores:         Physical core count. 0 = auto-detect via os.cpu_count().
        """
        if n_cores <= 0:
            n_cores = os.cpu_count() or 4

        config = SQAConfig(
            n_spins      = n_decisions,
            n_replicas   = n_cores,
            n_sweeps     = 300,
            gamma_init   = 4.0,
            gamma_final  = 1e-3,
            temp_init    = 1.5,
            temp_final   = 0.05,
            pin_threads  = True,
        )
        self.sqa = SQAMiddleware(config)

        # Enforce symmetry on the caller-supplied coupling matrix
        J = coupling_matrix.astype(np.float32)
        self.J = (J + J.T) / 2.0

    def optimize_action(
        self,
        logit_scores: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        """
        Takes raw agent logit scores and returns an optimized binary action mask.

        Args:
            logit_scores: [N] float array of raw logits or softmax probabilities.

        Returns:
            actions    : [N] int8 array, values in {0, 1}
            latency_ms : float, wall-clock time for this optimization call
            energy     : float, Ising Hamiltonian energy of the returned solution
        """
        # Clip to avoid log(0); negate so high confidence → low energy bias
        scores = np.clip(logit_scores.astype(np.float32), 1e-9, None)
        h = -np.log(scores)

        result  = self.sqa.optimize(self.J, h)
        actions = ((result["spins"] + 1) // 2).astype(np.int8)  # {-1,+1} → {0,1}
        return actions, result["latency_ms"], result["energy"]
