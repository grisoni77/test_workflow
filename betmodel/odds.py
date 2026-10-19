"""Quote, probabilità implicite, margine del bookmaker e dimensionamento delle puntate."""

from __future__ import annotations

import numpy as np


def implied(odds) -> np.ndarray:
    return 1.0 / np.asarray(odds, dtype=float)


def overround(odds) -> float:
    """Margine del bookmaker: 0.05 = 5%."""
    return float(implied(odds).sum() - 1)


def fair_probs(odds, method: str = "power") -> np.ndarray:
    """Probabilità "eque" togliendo il margine.

    - ``proportional``: divide per la somma (semplice, sovrastima i favoriti).
    - ``power``: trova k tale che sum(p_i^k) = 1. Corregge meglio il
      favourite-longshot bias (il margine è caricato soprattutto sugli sfavoriti).
    """
    p = implied(odds)
    if method == "proportional":
        return p / p.sum()
    if method != "power":
        raise ValueError(f"metodo sconosciuto: {method}")
    lo, hi = 1.0, 3.0
    if p.sum() < 1:  # quote già "sotto margine" (es. migliori quote di mercato)
        lo, hi = 0.3, 1.0
    for _ in range(100):
        k = (lo + hi) / 2
        # sum(p^k) decresce con k: se è ancora > 1 serve un k più grande
        if (p ** k).sum() > 1:
            lo = k
        else:
            hi = k
    q = p ** ((lo + hi) / 2)
    return q / q.sum()


def edge(prob: float, odds: float) -> float:
    """Valore atteso per unità puntata: 0.05 = +5%."""
    return prob * odds - 1


def kelly(prob: float, odds: float, fraction: float = 0.25, cap: float = 0.05) -> float:
    """Frazione del bankroll da puntare (Kelly frazionato, con tetto massimo)."""
    b = odds - 1
    if b <= 0:
        return 0.0
    f = (prob * b - (1 - prob)) / b
    return float(min(max(f * fraction, 0.0), cap))
