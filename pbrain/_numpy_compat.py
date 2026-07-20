"""NumPy 1.x / 2.x compatibility.

NumPy 2.0 renamed ``np.trapz`` to ``np.trapezoid``: the new name does not exist
below 2.0, and the old one is deprecated above it. p-Brain declares
``numpy>=1.24``, so both are in scope — and this is not hypothetical, because the
Apple-GPU environment is a NumPy 1.x environment *by construction*:
``tensorflow-metal`` only loads on TensorFlow <= 2.16, which in turn pins
``numpy<2``. Calling ``np.trapezoid`` there raises ``AttributeError`` deep inside
a kinetic fit, several stages into a run that has already done its expensive work.

The two spellings are the same function with the same signature, so binding
whichever one exists keeps every call site identical across both versions.
"""

from __future__ import annotations

try:                                       # NumPy >= 2.0
    from numpy import trapezoid
except ImportError:                        # NumPy < 2.0 — same function, former name
    from numpy import trapz as trapezoid   # type: ignore[no-redef]

__all__ = ["trapezoid"]
