"""Download e normalizzazione dei CSV di football-data.co.uk.

I file storici hanno colonne che cambiano nel tempo (es. ``BbMxH`` nelle
stagioni vecchie, ``MaxH`` in quelle recenti). Qui le riportiamo a un
formato unico:

    date, home, away, hg, ag,
    odds_<fonte>_<esito>   con fonte in {b365, ps, avg, max}
                           ed esito in {H, D, A, O25, U25}
    close_ps_<esito>       quote Pinnacle di chiusura (per la CLV)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE_URL = "https://football-data.co.uk/mmz4281"
FIXTURES_URL = "https://football-data.co.uk/fixtures.csv"

# Codici dei campionati principali su football-data.co.uk
LEAGUES = {
    "I1": "Serie A",
    "I2": "Serie B",
    "E0": "Premier League",
    "E1": "Championship",
    "E2": "League One",
    "E3": "League Two",
    "SP1": "La Liga",
    "SP2": "La Liga 2",
    "D1": "Bundesliga",
    "D2": "2. Bundesliga",
    "F1": "Ligue 1",
    "F2": "Ligue 2",
    "N1": "Eredivisie",
    "B1": "Jupiler League",
    "P1": "Primeira Liga",
    "T1": "Süper Lig",
    "G1": "Super League Grecia",
    "SC0": "Scottish Premiership",
}

# Per ogni colonna normalizzata, i possibili nomi nei CSV (in ordine di preferenza)
ODDS_ALIASES = {
    "odds_b365_H": ["B365H"], "odds_b365_D": ["B365D"], "odds_b365_A": ["B365A"],
    "odds_ps_H": ["PSH", "PH"], "odds_ps_D": ["PSD", "PD"], "odds_ps_A": ["PSA", "PA"],
    "odds_avg_H": ["AvgH", "BbAvH"], "odds_avg_D": ["AvgD", "BbAvD"], "odds_avg_A": ["AvgA", "BbAvA"],
    "odds_max_H": ["MaxH", "BbMxH"], "odds_max_D": ["MaxD", "BbMxD"], "odds_max_A": ["MaxA", "BbMxA"],
    "odds_b365_O25": ["B365>2.5"], "odds_b365_U25": ["B365<2.5"],
    "odds_ps_O25": ["P>2.5"], "odds_ps_U25": ["P<2.5"],
    "odds_avg_O25": ["Avg>2.5", "BbAv>2.5"], "odds_avg_U25": ["Avg<2.5", "BbAv<2.5"],
    "odds_max_O25": ["Max>2.5", "BbMx>2.5"], "odds_max_U25": ["Max<2.5", "BbMx<2.5"],
    "close_ps_H": ["PSCH"], "close_ps_D": ["PSCD"], "close_ps_A": ["PSCA"],
    "close_ps_O25": ["PC>2.5"], "close_ps_U25": ["PC<2.5"],
}

STATS_COLUMNS = {"hshots": "HS", "ashots": "AS", "hsot": "HST", "asot": "AST"}

# Bookmaker "morbidi" con quote 1X2 nei CSV (prefisso della colonna -> nome).
# Le colonne compaiono e spariscono negli anni: se manca, il valore è NaN.
SOFT_BOOKS = {
    "B365": "Bet365", "BW": "bwin", "WH": "William Hill", "IW": "Interwetten",
    "VC": "BetVictor", "LB": "Ladbrokes", "GB": "Gamebookers", "SJ": "Stan James",
    "BS": "Blue Square", "1XB": "1xBet", "BMGM": "BetMGM", "BV": "BetVictor (nuovo)",
    "CL": "Coral",
}
for _book in SOFT_BOOKS:
    if _book != "B365":  # Bet365 è già in ODDS_ALIASES
        for _o in "HDA":
            ODDS_ALIASES[f"odds_{_book.lower()}_{_o}"] = [f"{_book}{_o}"]


def season_code(start_year: int) -> str:
    """2024 -> '2425' (stagione 2024/25)."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_url(league: str, start_year: int) -> str:
    return f"{BASE_URL}/{season_code(start_year)}/{league}.csv"


def download(league: str, start_years: list[int], data_dir: Path) -> list[Path]:
    """Scarica i CSV delle stagioni richieste in ``data_dir/<league>/``."""
    import requests

    out_dir = data_dir / league
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for year in start_years:
        url = season_url(league, year)
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        path = out_dir / f"{season_code(year)}.csv"
        path.write_bytes(resp.content)
        paths.append(path)
    return paths


def download_fixtures(data_dir: Path) -> Path:
    import requests

    data_dir.mkdir(parents=True, exist_ok=True)
    resp = requests.get(FIXTURES_URL, timeout=30)
    resp.raise_for_status()
    path = data_dir / "fixtures.csv"
    path.write_bytes(resp.content)
    return path


def normalize(raw: pd.DataFrame, require_result: bool = True) -> pd.DataFrame:
    """Converte un CSV grezzo di football-data nel formato normalizzato."""
    raw = raw.dropna(how="all")
    out = pd.DataFrame({
        "date": pd.to_datetime(raw["Date"], dayfirst=True, format="mixed"),
        "home": raw["HomeTeam"].str.strip(),
        "away": raw["AwayTeam"].str.strip(),
    })
    if "Div" in raw:
        out["league"] = raw["Div"]
    if require_result:
        out["hg"] = pd.to_numeric(raw["FTHG"], errors="coerce")
        out["ag"] = pd.to_numeric(raw["FTAG"], errors="coerce")
    # Statistiche di partita (non presenti in tutte le stagioni/campionati)
    for col, src in STATS_COLUMNS.items():
        out[col] = pd.to_numeric(raw[src], errors="coerce") if src in raw.columns else float("nan")
    for col, aliases in ODDS_ALIASES.items():
        src = next((a for a in aliases if a in raw.columns), None)
        out[col] = pd.to_numeric(raw[src], errors="coerce") if src else float("nan")
    if require_result:
        out = out.dropna(subset=["hg", "ag"])
        out[["hg", "ag"]] = out[["hg", "ag"]].astype(int)
    return out.sort_values("date", kind="stable").reset_index(drop=True)


def load_csv(path: Path, require_result: bool = True) -> pd.DataFrame:
    # I file recenti sono UTF-8 con BOM, alcuni storici latin-1; ci sono righe con virgole in eccesso
    try:
        raw = pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
    except UnicodeDecodeError:
        raw = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
    raw.columns = raw.columns.str.replace("\ufeff", "").str.replace("ï»¿", "").str.strip()
    return normalize(raw, require_result=require_result)


def load_league(league: str, data_dir: Path) -> pd.DataFrame:
    """Carica e concatena tutte le stagioni scaricate di un campionato."""
    files = sorted((data_dir / league).glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"Nessun CSV in {data_dir / league}. Esegui prima il comando 'download'."
        )
    frames = [load_csv(f) for f in files]
    return pd.concat(frames, ignore_index=True).sort_values("date", kind="stable").reset_index(drop=True)
