"""Strategia "sharp vs soft": nessun modello statistico.

La probabilità di riferimento è quella implicita nelle quote Pinnacle
(il bookmaker con il margine più basso e le quote più precise), senza
margine. Si punta sul bookmaker "morbido" che offre la quota migliore
quando ``prob_pinnacle * quota_soft - 1 >= min_edge``.

Le quote di football-data.co.uk sono raccolte nello stesso momento per
tutti i bookmaker, quindi il confronto non usa informazioni future.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .backtest import MARKETS, Result, _clv, _won, summarize
from .data import SOFT_BOOKS
from .odds import fair_probs, kelly


@dataclass
class SharpConfig:
    books: tuple[str, ...] = tuple(b.lower() for b in SOFT_BOOKS)
    markets: tuple[str, ...] = ("1x2", "ou25")
    min_edge: float = 0.02
    min_odds: float = 1.3
    max_odds: float = 6.0
    fair_method: str = "power"
    staking: str = "flat"
    kelly_fraction: float = 0.25
    stake_cap: float = 0.05
    start_bankroll: float = 100.0
    # Filtro con il modello: punta solo se anche prob_modello * quota - 1 > model_filter.
    # None = nessun filtro. Richiede le colonne m_<esito> (vedi evaluate.walk_forward).
    model_filter: float | None = None
    # Con il filtro attivo, le occasioni con edge >= override_edge passano comunque
    override_edge: float | None = None
    goals_weight: float = 0.5


def _soft_columns(market: str, outcome: str, books: tuple[str, ...]) -> list[str]:
    # Nei CSV storici l'Over/Under c'è solo per Bet365: le altre colonne mancano e vengono ignorate
    return [f"odds_{b}_{outcome}" for b in books]


def find_bets(matches: pd.DataFrame, cfg: SharpConfig) -> list[dict]:
    out = []
    for _, row in matches.iterrows():
        for market in cfg.markets:
            outcomes = MARKETS[market]
            ps = [row.get(f"odds_ps_{o}") for o in outcomes]
            if any(pd.isna(x) or x <= 1 for x in ps):  # quote mancanti o errate (es. 0)
                continue
            fair = dict(zip(outcomes, fair_probs(ps, cfg.fair_method)))
            candidates = []
            for o in outcomes:
                cols = [c for c in _soft_columns(market, o, cfg.books) if c in row and pd.notna(row[c])]
                if not cols:
                    continue
                best = max(cols, key=lambda c: row[c])
                price = float(row[best])
                if not (cfg.min_odds <= price <= cfg.max_odds):
                    continue
                e = fair[o] * price - 1
                override = cfg.override_edge is not None and e >= cfg.override_edge
                if cfg.model_filter is not None and not override:
                    pm = row.get(f"m_{o}")
                    if pd.isna(pm) or pm * price - 1 <= cfg.model_filter:
                        continue
                if e >= cfg.min_edge:
                    book = best.split("_")[1]
                    candidates.append((e, o, price, book))
            if candidates:
                e, o, price, book = max(candidates)
                out.append({"date": row["date"], "home": row["home"], "away": row["away"],
                            "market": market, "pick": o, "odds": price, "book": book,
                            "prob": fair[o], "edge": e, "_row": row})
    return out


def run(matches: pd.DataFrame, cfg: SharpConfig) -> Result:
    if cfg.model_filter is not None and "m_H" not in matches:
        from .evaluate import ModelConfig, walk_forward
        matches = walk_forward(matches, ModelConfig(goals_weight=cfg.goals_weight))
    matches = matches.sort_values("date", kind="stable").reset_index(drop=True)
    bankroll = cfg.start_bankroll
    records = []
    for _, today in matches.groupby(matches["date"].dt.normalize()):
        start_of_day = bankroll
        for b in find_bets(today, cfg):
            row = b.pop("_row")
            if cfg.staking == "kelly":
                stake = start_of_day * kelly(b["prob"], b["odds"], cfg.kelly_fraction, cfg.stake_cap)
            else:
                stake = 1.0
            if stake <= 0:
                continue
            won = _won(b["pick"], row["hg"], row["ag"])
            profit = stake * (b["odds"] - 1) if won else -stake
            bankroll += profit
            records.append({**b, "stake": stake, "won": won, "profit": profit,
                            "score": f"{row['hg']}-{row['ag']}",
                            "clv": _clv(row, b["market"], b["pick"], b["odds"]),
                            "bankroll": bankroll})
    bets = pd.DataFrame(records)
    return Result(bets=bets, config=cfg, summary=summarize(bets, cfg))
