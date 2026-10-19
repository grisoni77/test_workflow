"""Backtest walk-forward: per ogni giornata il modello vede solo le partite precedenti."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .model import DixonColes
from .odds import edge, fair_probs, kelly

MARKETS = {"1x2": ["H", "D", "A"], "ou25": ["O25", "U25"]}


@dataclass
class Config:
    markets: tuple[str, ...] = ("1x2", "ou25")
    odds_source: str = "avg"          # b365 | ps | avg | max
    min_edge: float = 0.05            # valore atteso minimo per puntare
    min_odds: float = 1.5
    max_odds: float = 5.0
    market_weight: float = 0.0        # 0 = solo modello; 0.5 = media modello/mercato
    staking: str = "flat"             # flat | kelly
    kelly_fraction: float = 0.25
    stake_cap: float = 0.05
    start_bankroll: float = 100.0
    xi: float = 0.0019
    train_days: int = 730             # finestra di dati usata per il fit
    min_matches: int = 10             # partite minime per squadra nella finestra
    warmup_days: int = 180            # nessuna scommessa nei primi N giorni di dati
    one_bet_per_market: bool = True   # al massimo una puntata per mercato e partita


@dataclass
class Result:
    bets: pd.DataFrame
    config: Config
    summary: dict = field(default_factory=dict)


def model_probs(model: DixonColes, home: str, away: str) -> dict[str, float]:
    return model.predict(home, away).as_dict()


def _blend(p_model: dict, row: pd.Series, outcomes: list[str], src: str, w: float) -> dict:
    if w <= 0:
        return p_model
    odds = [row.get(f"odds_{src}_{o}") for o in outcomes]
    if any(pd.isna(x) for x in odds):
        return p_model
    fair = fair_probs(odds)
    return {o: (1 - w) * p_model[o] + w * f for o, f in zip(outcomes, fair)}


def select_bets(model: DixonColes, matches: pd.DataFrame, cfg: Config) -> list[dict]:
    """Restituisce le value bet su un insieme di partite (non ancora giocate)."""
    out = []
    for _, row in matches.iterrows():
        home, away = row["home"], row["away"]
        if not (model.knows(home, cfg.min_matches) and model.knows(away, cfg.min_matches)):
            continue
        p_all = model_probs(model, home, away)
        for market in cfg.markets:
            outcomes = MARKETS[market]
            p = _blend(p_all, row, outcomes, cfg.odds_source, cfg.market_weight)
            candidates = []
            for o in outcomes:
                price = row.get(f"odds_{cfg.odds_source}_{o}")
                if pd.isna(price) or not (cfg.min_odds <= price <= cfg.max_odds):
                    continue
                e = edge(p[o], price)
                if e >= cfg.min_edge:
                    candidates.append((e, o, price))
            if cfg.one_bet_per_market and candidates:
                candidates = [max(candidates)]
            for e, o, price in candidates:
                out.append({
                    "date": row["date"], "home": home, "away": away,
                    "market": market, "pick": o, "odds": float(price),
                    "prob": p[o], "edge": e,
                    "_row": row,
                })
    return out


def _won(pick: str, hg: int, ag: int) -> bool:
    return {
        "H": hg > ag, "D": hg == ag, "A": hg < ag,
        "O25": hg + ag > 2, "U25": hg + ag <= 2,
    }[pick]


def _clv(row: pd.Series, market: str, pick: str, odds: float) -> float:
    """Closing line value rispetto alla quota Pinnacle di chiusura senza margine."""
    outcomes = MARKETS[market]
    close = [row.get(f"close_ps_{o}") for o in outcomes]
    if any(pd.isna(x) for x in close):
        return float("nan")
    fair = dict(zip(outcomes, fair_probs(close)))
    return odds * fair[pick] - 1


def run(matches: pd.DataFrame, cfg: Config) -> Result:
    matches = matches.sort_values("date", kind="stable").reset_index(drop=True)
    first_bet_day = matches["date"].min() + pd.Timedelta(days=cfg.warmup_days)
    bankroll = cfg.start_bankroll
    records = []

    for day, today in matches.groupby(matches["date"].dt.normalize()):
        if day < first_bet_day:
            continue
        past = matches[(matches["date"] < day) &
                       (matches["date"] >= day - pd.Timedelta(days=cfg.train_days))]
        if len(past) < 50:
            continue
        model = DixonColes(xi=cfg.xi).fit(past, as_of=day)
        day_bets = select_bets(model, today, cfg)
        # le puntate della giornata sono decise con il bankroll di inizio giornata
        start_of_day = bankroll
        for b in day_bets:
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


def summarize(bets: pd.DataFrame, cfg: Config) -> dict:
    if bets.empty:
        return {"bets": 0}
    cum = bets["profit"].cumsum()
    if cfg.staking == "kelly":
        equity = cfg.start_bankroll + cum
        peak = np.maximum.accumulate(np.concatenate([[cfg.start_bankroll], equity]))[1:]
        max_dd = float(((peak - equity) / peak).max())
    else:
        peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
        max_dd = float((peak - cum).max())  # in unità puntate
    staked = bets["stake"].sum()
    profit = bets["profit"].sum()
    s = {
        "bets": int(len(bets)),
        "staked": float(staked),
        "profit": float(profit),
        "roi": float(profit / staked),
        "hit_rate": float(bets["won"].mean()),
        "avg_odds": float(bets["odds"].mean()),
        "avg_edge": float(bets["edge"].mean()),
        "avg_clv": float(bets["clv"].mean()) if bets["clv"].notna().any() else float("nan"),
        "clv_positive_share": float((bets["clv"] > 0).mean()) if bets["clv"].notna().any() else float("nan"),
        "max_drawdown": max_dd,
        # ROI / errore standard: |z| < 2 significa che il risultato può essere solo fortuna
        "z_score": _z_score(bets),
    }
    if cfg.staking == "kelly":
        s["final_bankroll"] = float(cfg.start_bankroll + profit)
    s["by_market"] = {
        m: {"bets": int(len(g)), "roi": float(g["profit"].sum() / g["stake"].sum())}
        for m, g in bets.groupby("market")
    }
    return s


def _z_score(bets: pd.DataFrame) -> float:
    """Test grezzo: il profitto per unità è distinguibile da zero?"""
    r = bets["profit"] / bets["stake"]
    if len(r) < 2 or r.std() == 0:
        return float("nan")
    return float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r))))
