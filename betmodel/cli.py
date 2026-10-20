"""Riga di comando: download, backtest, compare, predict, sharp, sharp-scan."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

import pandas as pd

from . import backtest, data, sharp
from .model import DixonColes

DEFAULT_DATA = Path("data")


def _years(spec: str) -> list[int]:
    """'2015-2024' -> [2015..2024]; '2023' -> [2023]. Anno = inizio stagione."""
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(spec)]


def _add_config_args(p: argparse.ArgumentParser) -> None:
    d = backtest.Config()
    p.add_argument("--markets", default=",".join(d.markets), help="1x2,ou25")
    p.add_argument("--odds", dest="odds_source", default=d.odds_source,
                   choices=["b365", "ps", "avg", "max"], help="quote su cui puntare")
    p.add_argument("--min-edge", type=float, default=d.min_edge)
    p.add_argument("--min-odds", type=float, default=d.min_odds)
    p.add_argument("--max-odds", type=float, default=d.max_odds)
    p.add_argument("--market-weight", type=float, default=d.market_weight,
                   help="peso delle quote di mercato nella probabilità finale (0-1)")
    p.add_argument("--staking", default=d.staking, choices=["flat", "kelly"])
    p.add_argument("--kelly-fraction", type=float, default=d.kelly_fraction)
    p.add_argument("--xi", type=float, default=d.xi, help="decadimento temporale per giorno")
    p.add_argument("--train-days", type=int, default=d.train_days)
    p.add_argument("--min-matches", type=int, default=d.min_matches)


def _config(args) -> backtest.Config:
    kw = {f.name: getattr(args, f.name) for f in fields(backtest.Config) if hasattr(args, f.name)}
    kw["markets"] = tuple(m.strip() for m in args.markets.split(","))
    return backtest.Config(**kw)


def cmd_download(args) -> None:
    for league in args.league.split(","):
        for path in data.download(league, _years(args.seasons), args.data_dir):
            print(f"scaricato {path}")
    if args.fixtures:
        print(f"scaricato {data.download_fixtures(args.data_dir)}")


def _print_summary(league: str, s: dict) -> None:
    print(f"\n=== {league} ({data.LEAGUES.get(league, '?')}) ===")
    if s["bets"] == 0:
        print("nessuna scommessa selezionata")
        return
    print(f"scommesse:      {s['bets']}")
    print(f"puntato:        {s['staked']:.1f}")
    print(f"profitto:       {s['profit']:+.1f}")
    print(f"ROI:            {s['roi']:+.2%}")
    print(f"z-score:        {s['z_score']:+.2f}   (|z|<2: compatibile con la fortuna)")
    print(f"vincenti:       {s['hit_rate']:.1%}   quota media {s['avg_odds']:.2f}")
    print(f"edge stimato:   {s['avg_edge']:+.2%}")
    print(f"CLV media:      {s['avg_clv']:+.2%}   CLV>0 nel {s['clv_positive_share']:.0%} dei casi")
    dd_unit = "del bankroll" if "final_bankroll" in s else "unità"
    dd = f"{s['max_drawdown']:.1%}" if "final_bankroll" in s else f"{s['max_drawdown']:.1f}"
    print(f"max drawdown:   {dd} {dd_unit}")
    if "final_bankroll" in s:
        print(f"bankroll finale: {s['final_bankroll']:.1f}")
    for m, ms in s["by_market"].items():
        print(f"  {m:5s} {ms['bets']:5d} scommesse  ROI {ms['roi']:+.2%}")


def cmd_backtest(args) -> None:
    cfg = _config(args)
    for league in args.league.split(","):
        matches = data.load_league(league, args.data_dir)
        res = backtest.run(matches, cfg)
        _print_summary(league, res.summary)
        if args.out:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            path = out.with_name(f"{out.stem}_{league}{out.suffix or '.csv'}")
            res.bets.to_csv(path, index=False)
            print(f"dettaglio scommesse: {path}")


def cmd_compare(args) -> None:
    """Stessa configurazione su più campionati: dove il modello rende meglio?"""
    cfg = _config(args)
    rows = []
    for league in args.league.split(","):
        try:
            matches = data.load_league(league, args.data_dir)
        except FileNotFoundError as e:
            print(e, file=sys.stderr)
            continue
        s = backtest.run(matches, cfg).summary
        rows.append({"league": league, "name": data.LEAGUES.get(league, "?"),
                     **{k: s.get(k) for k in ("bets", "roi", "z_score", "avg_clv", "hit_rate")}})
    df = pd.DataFrame(rows).sort_values("avg_clv", ascending=False)
    with pd.option_context("display.float_format", "{:+.3f}".format):
        print(df.to_string(index=False))


def cmd_predict(args) -> None:
    cfg = _config(args)
    fixtures_path = args.fixtures or args.data_dir / "fixtures.csv"
    fixtures = data.load_csv(Path(fixtures_path), require_result=False)
    for league in args.league.split(","):
        history = data.load_league(league, args.data_dir)
        upcoming = fixtures[fixtures["league"] == league] if "league" in fixtures else fixtures
        if upcoming.empty:
            print(f"\n{league}: nessuna partita in programma nel file fixtures")
            continue
        as_of = upcoming["date"].min()
        past = history[history["date"] < as_of]
        if past.empty:
            print(f"\n{league}: nessuna partita storica prima del {as_of:%d/%m/%Y}")
            continue
        # finestra ancorata all'ultima partita giocata, anche se i dati non sono aggiornati
        window_end = min(as_of, past["date"].max() + pd.Timedelta(days=1))
        past = past[past["date"] >= window_end - pd.Timedelta(days=cfg.train_days)]
        model = DixonColes(xi=cfg.xi).fit(past, as_of=window_end)

        print(f"\n=== {league} — probabilità del modello ===")
        for _, r in upcoming.iterrows():
            if not (model.knows(r["home"]) and model.knows(r["away"])):
                print(f"{r['home']} - {r['away']}: squadra senza storico, salto")
                continue
            p = model.predict(r["home"], r["away"])
            print(f"{r['date']:%d/%m} {r['home']:>15s} - {r['away']:<15s} "
                  f"1 {p.home_win:.0%}  X {p.draw:.0%}  2 {p.away_win:.0%}  "
                  f"O2.5 {p.over25:.0%}  xG {p.exp_home_goals:.2f}-{p.exp_away_goals:.2f}")

        bets = backtest.select_bets(model, upcoming, cfg)
        print(f"\n--- value bet (edge >= {cfg.min_edge:.0%}, quote '{cfg.odds_source}') ---")
        if not bets:
            print("nessuna")
        for b in sorted(bets, key=lambda b: -b["edge"]):
            stake = ""
            if cfg.staking == "kelly":
                from .odds import kelly
                stake = f"  puntata {kelly(b['prob'], b['odds'], cfg.kelly_fraction, cfg.stake_cap):.1%} bankroll"
            print(f"{b['date']:%d/%m} {b['home']} - {b['away']}: {b['pick']} @ {b['odds']:.2f}  "
                  f"prob {b['prob']:.1%}  edge {b['edge']:+.1%}{stake}")


def _add_sharp_args(p: argparse.ArgumentParser) -> None:
    d = sharp.SharpConfig()
    p.add_argument("--books", default=",".join(d.books),
                   help="bookmaker su cui puntare, es. b365,bw,wh (default: tutti)")
    p.add_argument("--markets", default=",".join(d.markets), help="1x2,ou25")
    p.add_argument("--min-edge", type=float, default=d.min_edge)
    p.add_argument("--min-odds", type=float, default=d.min_odds)
    p.add_argument("--max-odds", type=float, default=d.max_odds)
    p.add_argument("--staking", default=d.staking, choices=["flat", "kelly"])
    p.add_argument("--kelly-fraction", type=float, default=d.kelly_fraction)


def _sharp_config(args) -> sharp.SharpConfig:
    return sharp.SharpConfig(
        books=tuple(b.strip().lower() for b in args.books.split(",")),
        markets=tuple(m.strip() for m in args.markets.split(",")),
        min_edge=args.min_edge, min_odds=args.min_odds, max_odds=args.max_odds,
        staking=args.staking, kelly_fraction=args.kelly_fraction,
    )


def cmd_sharp(args) -> None:
    cfg = _sharp_config(args)
    rows = []
    for league in args.league.split(","):
        try:
            matches = data.load_league(league, args.data_dir)
        except FileNotFoundError as e:
            print(e, file=sys.stderr)
            continue
        res = sharp.run(matches, cfg)
        _print_summary(league, res.summary)
        if res.summary["bets"]:
            years = (matches["date"].max() - matches["date"].min()).days / 365.25
            print(f"scommesse/anno: {res.summary['bets'] / years:.0f}")
            rows.append({"league": league, "bets": res.summary["bets"],
                         "roi": res.summary["roi"], "avg_clv": res.summary["avg_clv"]})
        if args.out and not res.bets.empty:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            res.bets.to_csv(out.with_name(f"{out.stem}_{league}{out.suffix or '.csv'}"), index=False)
    if len(rows) > 1:
        print("\n" + pd.DataFrame(rows).sort_values("avg_clv", ascending=False)
              .to_string(index=False, float_format="{:+.3f}".format))


def cmd_sharp_scan(args) -> None:
    """Opportunità sulle prossime partite (quote del file fixtures)."""
    cfg = _sharp_config(args)
    fixtures = data.load_csv(Path(args.fixtures or args.data_dir / "fixtures.csv"), require_result=False)
    if args.league:
        fixtures = fixtures[fixtures["league"].isin(args.league.split(","))]
    if fixtures["odds_ps_H"].isna().all():
        print("il file fixtures non contiene quote Pinnacle: servono quote in tempo reale "
              "(es. The Odds API) per usare questa strategia sulle prossime partite")
        return
    bets = sharp.find_bets(fixtures, cfg)
    if not bets:
        print("nessuna opportunità con le quote attuali")
    for b in sorted(bets, key=lambda b: -b["edge"]):
        row = b["_row"]
        print(f"{row.get('league', '')} {b['date']:%d/%m} {b['home']} - {b['away']}: "
              f"{b['pick']} @ {b['odds']:.2f} su {data.SOFT_BOOKS.get(b['book'].upper(), b['book'])}  "
              f"prob Pinnacle {b['prob']:.1%}  edge {b['edge']:+.1%}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="betmodel", description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("download", help="scarica i CSV da football-data.co.uk")
    p.add_argument("--league", default="I1", help="es. I1,I2,E0")
    p.add_argument("--seasons", default="2012-2025", help="anni di inizio stagione, es. 2012-2025")
    p.add_argument("--fixtures", action="store_true", help="scarica anche le prossime partite")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("backtest", help="simula la strategia sullo storico")
    p.add_argument("--league", default="I1")
    p.add_argument("--out", help="CSV con il dettaglio delle scommesse")
    _add_config_args(p)
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("compare", help="confronta i campionati con la stessa strategia")
    p.add_argument("--league", default=",".join(data.LEAGUES))
    _add_config_args(p)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("predict", help="probabilità e value bet sulle prossime partite")
    p.add_argument("--league", default="I1")
    p.add_argument("--fixtures", help="CSV fixtures (default: <data-dir>/fixtures.csv)")
    _add_config_args(p)
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("sharp", help="backtest: quote Pinnacle come riferimento, puntate sui bookmaker morbidi")
    p.add_argument("--league", default="I1")
    p.add_argument("--out", help="CSV con il dettaglio delle scommesse")
    _add_sharp_args(p)
    p.set_defaults(func=cmd_sharp)

    p = sub.add_parser("sharp-scan", help="opportunità sharp vs soft sulle prossime partite")
    p.add_argument("--league", help="filtra i campionati, es. I1,I2 (default: tutti)")
    p.add_argument("--fixtures", help="CSV fixtures (default: <data-dir>/fixtures.csv)")
    _add_sharp_args(p)
    p.set_defaults(func=cmd_sharp_scan)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
