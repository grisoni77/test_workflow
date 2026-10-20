"""Valutazione della qualità delle previsioni, indipendente dalle scommesse.

Il modello genera probabilità walk-forward (ogni giornata vede solo il
passato) e le confrontiamo con quelle implicite nelle quote Pinnacle tramite
log loss. La domanda chiave: mescolare modello e Pinnacle riduce la log loss
rispetto a Pinnacle da solo? Se no, il modello non aggiunge informazione.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model import DixonColes
from .odds import fair_probs

OUTCOMES = ["H", "D", "A"]


@dataclass
class ModelConfig:
    # 1.0 = solo gol; 0.0 = solo tiri in porta convertiti in gol; valori intermedi = media
    goals_weight: float = 1.0
    xi: float = 0.0019
    train_days: int = 730
    min_matches: int = 10
    warmup_days: int = 180


def _targets(past: pd.DataFrame, goals_weight: float) -> pd.DataFrame:
    """Aggiunge le colonne th/ta: gol "corretti" con i tiri in porta."""
    past = past.copy()
    if goals_weight >= 1:
        past["th"], past["ta"] = past["hg"], past["ag"]
        return past
    has = past["hsot"].notna() & past["asot"].notna()
    sot = past.loc[has, ["hsot", "asot"]].to_numpy().sum()
    goals = past.loc[has, ["hg", "ag"]].to_numpy().sum()
    conv = goals / sot if sot > 0 else 0.0  # gol per tiro in porta nella finestra
    xh = np.where(has, past["hsot"].fillna(0) * conv, past["hg"])
    xa = np.where(has, past["asot"].fillna(0) * conv, past["ag"])
    past["th"] = goals_weight * past["hg"] + (1 - goals_weight) * xh
    past["ta"] = goals_weight * past["ag"] + (1 - goals_weight) * xa
    return past


def walk_forward(matches: pd.DataFrame, cfg: ModelConfig) -> pd.DataFrame:
    """Probabilità del modello per ogni partita, stimate solo con le partite precedenti."""
    matches = matches.sort_values("date", kind="stable").reset_index(drop=True)
    first = matches["date"].min() + pd.Timedelta(days=cfg.warmup_days)
    rows = []
    for day, today in matches.groupby(matches["date"].dt.normalize()):
        if day < first:
            continue
        past = matches[(matches["date"] < day) &
                       (matches["date"] >= day - pd.Timedelta(days=cfg.train_days))]
        if len(past) < 50:
            continue
        model = DixonColes(xi=cfg.xi).fit(_targets(past, cfg.goals_weight), as_of=day, target=("th", "ta"))
        for i, r in today.iterrows():
            if not (model.knows(r["home"], cfg.min_matches) and model.knows(r["away"], cfg.min_matches)):
                continue
            p = model.predict(r["home"], r["away"])
            rows.append({"idx": i, "m_H": p.home_win, "m_D": p.draw, "m_A": p.away_win,
                         "m_O25": p.over25, "m_U25": p.under25})
    probs = pd.DataFrame(rows).set_index("idx")
    return matches.join(probs, how="inner")


def add_market_probs(df: pd.DataFrame, prefix: str, name: str) -> pd.DataFrame:
    """Aggiunge <name>_H/D/A: probabilità senza margine dalle quote <prefix>_H/D/A."""
    df = df.copy()
    cols = [f"{prefix}_{o}" for o in OUTCOMES]
    ok = df[cols].notna().all(axis=1) & (df[cols] > 1).all(axis=1)
    fair = np.full((len(df), 3), np.nan)
    fair[ok.to_numpy()] = np.vstack([fair_probs(r) for r in df.loc[ok, cols].to_numpy()]) if ok.any() else []
    for k, o in enumerate(OUTCOMES):
        df[f"{name}_{o}"] = fair[:, k]
    return df


def result_index(df: pd.DataFrame) -> np.ndarray:
    return np.select([df["hg"] > df["ag"], df["hg"] == df["ag"]], [0, 1], 2)


def log_loss(df: pd.DataFrame, name: str) -> float:
    p = df[[f"{name}_{o}" for o in OUTCOMES]].to_numpy()
    y = result_index(df)
    return float(-np.log(np.clip(p[np.arange(len(df)), y], 1e-12, 1)).mean())


def blend(df: pd.DataFrame, a: str, b: str, w: float, name: str) -> pd.DataFrame:
    """Media logaritmica (geometrica normalizzata): w peso di ``a``."""
    df = df.copy()
    pa = df[[f"{a}_{o}" for o in OUTCOMES]].to_numpy()
    pb = df[[f"{b}_{o}" for o in OUTCOMES]].to_numpy()
    q = np.exp(w * np.log(np.clip(pa, 1e-12, 1)) + (1 - w) * np.log(np.clip(pb, 1e-12, 1)))
    q /= q.sum(axis=1, keepdims=True)
    for k, o in enumerate(OUTCOMES):
        df[f"{name}_{o}"] = q[:, k]
    return df


def information_test(df: pd.DataFrame, weights=np.linspace(0, 0.5, 11)) -> pd.DataFrame:
    """Log loss di Pinnacle mescolato al modello, per diversi pesi del modello.

    Se il minimo è a peso 0, il modello non aggiunge nulla a Pinnacle.
    Il peso migliore è scelto sugli stessi dati: il guadagno è leggermente ottimistico.
    """
    df = add_market_probs(df, "odds_ps", "ps").dropna(subset=["ps_H", "m_H"])
    rows = []
    for w in weights:
        ll = log_loss(blend(df, "m", "ps", w, "b"), "b")
        rows.append({"model_weight": round(float(w), 2), "log_loss": ll})
    return pd.DataFrame(rows)
