# src/sse_core/__init__.py
"""
SSE Core: A High-Performance Stochastic Single Electronics Simulator.
"""

__version__ = "1.0.0"

# Expose the primary user-facing API directly at the root level
from sse_core.compiler.builder import SSECompiler

# We will expose the GillespieSolver here in Phase 3!
from sse_core.solvers.gillespie import GillespieSolver

__all__ = [
    "SSECompiler",
    "GillespieSolver",
]
