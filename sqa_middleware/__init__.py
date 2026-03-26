"""
sqa_middleware
==============
Simulated Quantum Annealing middleware for autonomous AI agents.
Ryzen-optimized, double-buffered Trotter decomposition — Rev 1.4.
"""
from .core import SQAMiddleware, SQAConfig
from .agent_adapter import AgentSQAAdapter

__all__ = ["SQAMiddleware", "SQAConfig", "AgentSQAAdapter"]
__version__ = "1.4.0"
