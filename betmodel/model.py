"""Modello Dixon-Coles con decadimento temporale.

Gol casa  ~ Poisson(attack[casa] * defence[ospite] * home_adv)
Gol ospite ~ Poisson(attack[ospite] * defence[casa])

con la correzione ``tau`` di Dixon-Coles sui risultati bassi (0-0, 1-0,
0-1, 1-1) e partite pesate con ``exp(-xi * giorni_fa)``.

Il fit dei parametri di Poisson usa l'algoritmo iterativo di Maher (che
converge allo stimatore di massima verosimiglianza), poi ``rho`` è scelto
con una ricerca a griglia. Non servono dipendenze oltre a numpy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MAX_GOALS = 10


@dataclass
class Prediction:
    home_win: float
    draw: float
    away_win: float
    over25: float
    under25: float
    btts: float
    exp_home_goals: float
    exp_away_goals: float

    def as_dict(self) -> dict[str, float]:
        return {"H": self.home_win, "D": self.draw, "A": self.away_win,
                "O25": self.over25, "U25": self.under25}


def _tau(hg: np.ndarray, ag: np.ndarray, lam: np.ndarray, mu: np.ndarray, rho: float) -> np.ndarray:
    tau = np.ones_like(lam, dtype=float)
    m00 = (hg == 0) & (ag == 0)
    m10 = (hg == 1) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m11 = (hg == 1) & (ag == 1)
    tau[m00] = 1 - lam[m00] * mu[m00] * rho
    tau[m10] = 1 + mu[m10] * rho
    tau[m01] = 1 + lam[m01] * rho
    tau[m11] = 1 - rho
    return tau


class DixonColes:
    def __init__(self, xi: float = 0.0019, max_iter: int = 200, tol: float = 1e-8):
        # xi=0.0019/giorno: una partita di un anno fa pesa circa la metà
        self.xi = xi
        self.max_iter = max_iter
        self.tol = tol
        self.teams: list[str] = []
        self.attack: dict[str, float] = {}
        self.defence: dict[str, float] = {}
        self.matches_played: dict[str, float] = {}
        self.home_adv = 1.0
        self.rho = 0.0

    def fit(self, matches: pd.DataFrame, as_of: pd.Timestamp | None = None,
            target: tuple[str, str] = ("hg", "ag")) -> "DixonColes":
        """Stima i parametri sulle partite passate (colonne date, home, away, hg, ag).

        ``target`` sono le colonne usate per stimare attacco/difesa: di default i
        gol, ma possono essere "gol attesi" non interi (es. da tiri in porta).
        ``rho`` è sempre stimato sui gol reali.
        """
        if as_of is None:
            as_of = matches["date"].max()
        days = (as_of - matches["date"]).dt.days.to_numpy(dtype=float)
        w = np.exp(-self.xi * np.clip(days, 0, None))

        teams = sorted(set(matches["home"]) | set(matches["away"]))
        idx = {t: i for i, t in enumerate(teams)}
        h = matches["home"].map(idx).to_numpy()
        a = matches["away"].map(idx).to_numpy()
        hg = matches[target[0]].to_numpy(dtype=float)
        ag = matches[target[1]].to_numpy(dtype=float)
        n = len(teams)

        # Gol fatti/subiti pesati: costanti durante le iterazioni
        scored = np.bincount(h, w * hg, n) + np.bincount(a, w * ag, n)
        conceded = np.bincount(h, w * ag, n) + np.bincount(a, w * hg, n)

        att = np.ones(n)
        dfn = np.ones(n)
        gamma = 1.0
        for _ in range(self.max_iter):
            old = np.concatenate([att, dfn, [gamma]])
            att_den = np.bincount(h, w * dfn[a] * gamma, n) + np.bincount(a, w * dfn[h], n)
            att = scored / np.where(att_den > 0, att_den, 1)
            att = np.where(att > 0, att, 1e-3)  # squadra senza gol: evita parametri nulli
            att /= att.mean()
            def_den = np.bincount(h, w * att[a], n) + np.bincount(a, w * att[h] * gamma, n)
            dfn = conceded / np.where(def_den > 0, def_den, 1)
            dfn = np.where(dfn > 0, dfn, 1e-3)
            gamma = (w * hg).sum() / (w * att[h] * dfn[a]).sum()
            new = np.concatenate([att, dfn, [gamma]])
            if np.max(np.abs(new - old)) < self.tol:
                break

        lam = att[h] * dfn[a] * gamma
        mu = att[a] * dfn[h]
        self.rho = self._fit_rho(matches["hg"].to_numpy(dtype=int), matches["ag"].to_numpy(dtype=int),
                                 lam, mu, w)

        self.teams = teams
        self.attack = dict(zip(teams, att))
        self.defence = dict(zip(teams, dfn))
        self.matches_played = dict(zip(teams, np.bincount(h, minlength=n) + np.bincount(a, minlength=n)))
        self.home_adv = float(gamma)
        return self

    @staticmethod
    def _fit_rho(hg, ag, lam, mu, w) -> float:
        low = (hg <= 1) & (ag <= 1)
        # rho deve tenere tau > 0 per tutte le partite
        best_rho, best_ll = 0.0, -np.inf
        for rho in np.linspace(-0.25, 0.25, 101):
            tau = _tau(hg[low], ag[low], lam[low], mu[low], rho)
            if np.any(tau <= 0):
                continue
            ll = (w[low] * np.log(tau)).sum()
            if ll > best_ll:
                best_rho, best_ll = float(rho), ll
        return best_rho

    def knows(self, team: str, min_matches: int = 0) -> bool:
        return team in self.attack and self.matches_played.get(team, 0) >= min_matches

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        lam = self.attack[home] * self.defence[away] * self.home_adv
        mu = self.attack[away] * self.defence[home]
        return lam, mu

    def score_matrix(self, home: str, away: str) -> np.ndarray:
        """Matrice P[gol_casa, gol_ospite] fino a MAX_GOALS."""
        lam, mu = self.expected_goals(home, away)
        goals = np.arange(MAX_GOALS + 1)
        ph = _poisson_pmf(goals, lam)
        pa = _poisson_pmf(goals, mu)
        m = np.outer(ph, pa)
        m[0, 0] *= 1 - lam * mu * self.rho
        m[1, 0] *= 1 + mu * self.rho
        m[0, 1] *= 1 + lam * self.rho
        m[1, 1] *= 1 - self.rho
        return m / m.sum()

    def predict(self, home: str, away: str) -> Prediction:
        m = self.score_matrix(home, away)
        totals = np.add.outer(np.arange(MAX_GOALS + 1), np.arange(MAX_GOALS + 1))
        lam, mu = self.expected_goals(home, away)
        over = float(m[totals >= 3].sum())
        return Prediction(
            home_win=float(np.tril(m, -1).sum()),
            draw=float(np.trace(m)),
            away_win=float(np.triu(m, 1).sum()),
            over25=over,
            under25=1 - over,
            btts=float(m[1:, 1:].sum()),
            exp_home_goals=lam,
            exp_away_goals=mu,
        )


def _poisson_pmf(k: np.ndarray, lam: float) -> np.ndarray:
    from math import lgamma

    log_fact = np.array([lgamma(x + 1) for x in k])
    return np.exp(k * np.log(lam) - lam - log_fact)
