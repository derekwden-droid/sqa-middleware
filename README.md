# SQA Middleware — Simulated Quantum Annealing Optimizer for Autonomous AI Agents

**Ryzen-optimized, double-buffered Trotter decomposition middleware** that turns any LLM or autonomous agent action vector into a provably better decision using quantum tunneling simulation on commodity hardware.

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

### Why this exists
Classical simulated annealing gets stuck in local minima.  
This middleware uses **Suzuki-Trotter decomposition + double-buffered inter-replica state exchange** to simulate quantum tunneling — on your existing Ryzen CPU in a Docker container. No GPUs, no quantum hardware required.

Full technical research report (Rev 1.4) with architecture details, benchmark results, and USPTO-style patent claims is included in `/docs`.

### Status
- Open-source release under MIT license
- Provisional patent application pending (filed to protect core claims)
- Ready for local testing on Ryzen multi-core systems

### Features
- Bijective Trotter replica → OS thread mapping (`P =` physical cores)
- Double-buffered shared-memory arrays (race-condition free)
- Dual geometric annealing schedules (Γ decays faster than T)
- Agent logit scores → Ising bias translation (`h_i = -log(confidence)`)
- Full A/B benchmarking suite vs classical SA (energy, latency, escape rate)
- Docker + Ryzen CPU pinning ready

### Quick Start

#### Option 1: Direct Python
```bash
git clone https://github.com/derekwden-droid/sqa-middleware.git
cd sqa-middleware

pip install numpy scipy psutil
python -m benchmarks.ab_test_runner
