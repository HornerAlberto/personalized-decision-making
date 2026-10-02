"""Formatting of point estimates, intervals and head counts."""
import numpy as np


def fmt_interval(bounds, tol: float) -> str:
    """Point estimate when the bounds collapse, interval otherwise."""
    low, high = float(np.min(bounds)), float(np.max(bounds))
    if high - low < tol:
        return f"{(low + high) / 2:.3f}"
    return f"[{low:.3f}, {high:.3f}]"


def fmt(quantity, tol: float) -> str:
    """Scalar for identified quantities, interval for partially identified ones."""
    if np.ndim(quantity) == 0:
        return f"{float(quantity):.3f}"
    return fmt_interval(quantity, tol)


def fmt_count(bounds, n: int) -> str:
    """The same estimate expressed as a head count out of the n patients sampled."""
    low, high = float(np.min(bounds)), float(np.max(bounds))
    if round(low * n) == round(high * n):
        return f"n ≈ {round(low * n):,}"
    return f"n ≈ {round(low * n):,}–{round(high * n):,}"
