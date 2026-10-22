"""Quote in tempo reale da The Odds API (https://the-odds-api.com, API v4).

Le risposte vengono convertite nello stesso formato dei CSV storici
(colonne ``odds_<bookmaker>_<esito>``), così la strategia sharp vs soft
funziona senza modifiche. Pinnacle diventa ``odds_ps_*``.

Ogni chiamata a /odds consuma crediti: (numero di mercati) x (numero di regioni).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

API_URL = "https://api.the-odds-api.com/v4"

# Codici football-data -> chiavi sport di The Odds API
SPORT_KEYS = {
    "I1": "soccer_italy_serie_a",
    "I2": "soccer_italy_serie_b",
    "E0": "soccer_epl",
    "E1": "soccer_efl_champ",
    "E2": "soccer_england_league1",
    "SP1": "soccer_spain_la_liga",
    "SP2": "soccer_spain_segunda_division",
    "D1": "soccer_germany_bundesliga",
    "D2": "soccer_germany_bundesliga2",
    "F1": "soccer_france_ligue_one",
    "F2": "soccer_france_ligue_two",
    "N1": "soccer_netherlands_eredivisie",
    "P1": "soccer_portugal_primeira_liga",
    "B1": "soccer_belgium_first_div",
    "T1": "soccer_turkey_super_league",
    "G1": "soccer_greece_super_league",
    "SC0": "soccer_spl",
}

SHARP_BOOK = "pinnacle"
# Exchange: le quote non includono la commissione, quindi non sono confrontabili
EXCHANGES = {"betfair_ex_eu", "betfair_ex_uk", "betfair_ex_au", "matchbook", "smarkets"}


def load_api_key(env_file: Path = Path(".env")) -> str:
    key = os.environ.get("ODDS_API_KEY")
    if not key and env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("ODDS_API_KEY="):
                key = line.split("=", 1)[1].strip()
    if not key:
        raise RuntimeError("chiave mancante: imposta ODDS_API_KEY o crea il file .env")
    return key


def fetch_odds(sport_key: str, api_key: str, regions: str = "eu",
               markets: str = "h2h,totals") -> tuple[list[dict], dict]:
    """Restituisce (eventi, quota) dove quota riporta i crediti usati e rimanenti."""
    import requests

    resp = requests.get(
        f"{API_URL}/sports/{sport_key}/odds",
        params={"apiKey": api_key, "regions": regions, "markets": markets,
                "oddsFormat": "decimal", "dateFormat": "iso"},
        timeout=30,
    )
    resp.raise_for_status()
    quota = {k: resp.headers.get(f"x-requests-{k}") for k in ("used", "remaining", "last")}
    return resp.json(), quota


def _book_code(key: str) -> str:
    # niente "_" nel codice: le colonne sono odds_<codice>_<esito>
    return "ps" if key == SHARP_BOOK else key.replace("_", "")


def events_to_frame(events: list[dict], league: str | None = None) -> pd.DataFrame:
    """Converte la risposta di /odds nel formato normalizzato (una riga per partita)."""
    rows = []
    for ev in events:
        row = {
            "date": pd.Timestamp(ev["commence_time"]).tz_convert(None),
            "home": ev["home_team"],
            "away": ev["away_team"],
            "league": league,
        }
        for bm in ev.get("bookmakers", []):
            if bm["key"] in EXCHANGES:
                continue
            code = _book_code(bm["key"])
            for market in bm.get("markets", []):
                for out in market.get("outcomes", []):
                    if market["key"] == "h2h":
                        o = {ev["home_team"]: "H", ev["away_team"]: "A", "Draw": "D"}.get(out["name"])
                    elif market["key"] == "totals" and out.get("point") == 2.5:
                        o = {"Over": "O25", "Under": "U25"}.get(out["name"])
                    else:
                        o = None
                    if o:
                        row[f"odds_{code}_{o}"] = float(out["price"])
        rows.append(row)
    return pd.DataFrame(rows)


def soft_books(frame: pd.DataFrame) -> tuple[str, ...]:
    """Bookmaker presenti nella risposta, escluso Pinnacle."""
    books = {c.split("_")[1] for c in frame.columns if c.startswith("odds_")}
    return tuple(sorted(books - {"ps"}))


def save_snapshot(events: list[dict], sport_key: str, out_dir: Path) -> Path:
    """Salva la risposta grezza: servirà per misurare la CLV delle giocate."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{sport_key}_{stamp}.json"
    path.write_text(json.dumps(events))
    return path
