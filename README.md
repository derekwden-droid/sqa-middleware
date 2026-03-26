[README (1).md](https://github.com/user-attachments/files/26257419/README.1.md)
# SQA Middleware — Simulated Quantum Annealing Optimizer for Autonomous AI Agents

**Ryzen-optimized, double-buffered Trotter decomposition middleware** that turns any LLM/agent action vector into a provably better decision using quantum tunneling simulation on commodity hardware.

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rev](https://img.shields.io/badge/Rev-1.4-green)]()

---

### Why this exists

Classical simulated annealing gets stuck in local minima.
This middleware uses **Suzuki-Trotter decomposition + double-buffered inter-replica state exchange** to simulate quantum tunneling — on your existing Ryzen CPU in a Docker container. No GPUs. No quantum hardware. No cloud.

Full technical research report (Rev 1.4) with USPTO-style patent claims is in `/docs`.

---

### Features

- Bijective Trotter replica → OS thread mapping (`P` = physical core count, auto-detected)
- Double-buffered shared-memory arrays — race-condition free, no locks in hot path
- Dual geometric annealing schedules (`Γ` decays faster than `T` — enforces quantum→classical phase transition)
- Agent logit scores → Ising bias translation (`h_i = -log(confidence_i)`)
- Escape-rate tracking identical to classical SA baseline (comparable across both)
- Full A/B benchmarking suite: escape rate, solution quality, latency vs classical SA
- Docker + Ryzen CPU pinning ready (`--cpuset-cpus`)

---

### Repo structure

```
sqa-middleware/
├── sqa_middleware/
│   ├── __init__.py
│   ├── core.py            ← SQAMiddleware + SQAConfig (main optimizer)
│   └── agent_adapter.py   ← Thin adapter: logits → Ising → binary actions
├── benchmarks/
│   └── ab_test_runner.py  ← A/B benchmark suite with live console table
├── docs/
│   └── SQA_Research_Report.html  ← Full Rev 1.4 technical report + patent claims
├── docker-compose.bench.yml
├── Dockerfile
├── requirements.txt
└── NOTICE
```

---

### Quick Start — Bare Python (recommended for local Ryzen)

```bash
# 1. Clone
git clone https://github.com/derekwden-droid/sqa-middleware.git
cd sqa-middleware

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the benchmark (auto-detects your core count)
python -m benchmarks.ab_test_runner

# Fast mode (100 trials, N=32, 150 sweeps — ~2 min on Ryzen 9)
python -m benchmarks.ab_test_runner --fast

# Custom parameters
python -m benchmarks.ab_test_runner --trials 500 --spins 64 --sweeps 300
```

---

### Quick Start — Docker (isolated CPU pinning, patent-grade reproducibility)

```bash
# Build and run benchmarks with cores 0-11 pinned to SQA container
docker compose -f docker-compose.bench.yml up --build
```

---

### Use in your own agent

```python
import numpy as np
from sqa_middleware import AgentSQAAdapter

# J: your pre-trained [N, N] decision correlation matrix
# Can be identity (np.eye(N)) to start, then train from agent history
N = 64
J = np.eye(N, dtype=np.float32)   # replace with your correlation matrix

adapter = AgentSQAAdapter(n_decisions=N, coupling_matrix=J)

# logit_scores: raw logits or softmax probabilities from your LLM/agent
logit_scores = np.random.rand(N).astype(np.float32)

actions, latency_ms, energy = adapter.optimize_action(logit_scores)
print(f"Optimal actions: {actions}")
print(f"Latency: {latency_ms:.2f} ms  |  Energy: {energy:.4f}")
```

---

### Benchmark output format

```
════════════════════════════════════════════════════════════════════════════════════════════════════════
  PROBLEM             N     SQA Energy           SA Energy           Gain    SQA ms     SA ms    Esc×    p-val
────────────────────────────────────────────────────────────────────────────────────────────────────────
  sk                 64   -41.87±0.62       -38.95±1.21          +7.5%    5.8ms      5.2ms    3.8×   <0.001
  routing            64   -21.55±0.39       -20.11±0.71          +7.2%    5.6ms      5.0ms    3.1×   <0.001
```

Green values = statistically significant (p < 0.05). `Esc×` = SQA escape events ÷ SA escape events.

---

### Patent notice

See `NOTICE` file. Methods implemented in this software are the subject of a pending USPTO provisional patent application.

---

### License

MIT — see `LICENSE`.
