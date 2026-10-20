# betmodel

Tool da riga di comando per cercare **value bet** sul calcio (Serie A e altri
campionati) con un modello **Dixon-Coles** e verificarle con un **backtest
walk-forward** sullo storico.

> Il tool non indovina chi vince. Stima le probabilità e le confronta con le
> quote: segnala una scommessa solo se `probabilità × quota − 1 ≥ soglia`.
> Il backtest serve a capire **se** il metodo funziona davvero: nella maggior
> parte delle configurazioni il risultato onesto sarà negativo.

## Installazione

```bash
pip install -r requirements.txt
```

## Uso

```bash
# 1. Scarica lo storico (anno = inizio stagione) e le prossime partite
python -m betmodel download --league I1,I2 --seasons 2012-2025 --fixtures

# 2. Backtest: ogni giornata il modello vede solo le partite precedenti
python -m betmodel backtest --league I1 --odds avg --min-edge 0.05 --out results/bets.csv

# 3. Confronta i campionati con la stessa strategia (ordinati per CLV)
python -m betmodel compare --league I1,I2,E0,E1,SP2,D2,F2

# 4. Probabilità e value bet sulle prossime partite
python -m betmodel predict --league I1 --staking kelly
```

Opzioni principali (valide per `backtest`, `compare`, `predict`):

| Opzione | Default | Significato |
|---|---|---|
| `--markets` | `1x2,ou25` | mercati: 1X2 e Over/Under 2.5 |
| `--odds` | `avg` | quote su cui puntare: `b365`, `ps` (Pinnacle), `avg`, `max` |
| `--min-edge` | `0.05` | valore atteso minimo (5%) |
| `--min-odds` / `--max-odds` | `1.5` / `5.0` | intervallo di quote accettate |
| `--market-weight` | `0` | quanto pesare le quote di mercato nella probabilità (0–1) |
| `--staking` | `flat` | `flat` (1 unità) o `kelly` (Kelly frazionato) |
| `--xi` | `0.0019` | decadimento temporale per giorno |
| `--train-days` | `730` | finestra di storico usata per il fit |

## Strategia "sharp vs soft"

Non usa il modello: prende le quote Pinnacle (margine basso, molto precise)
come stima della probabilità vera e punta sul bookmaker "morbido" con la quota
migliore quando `prob_Pinnacle × quota − 1 ≥ min-edge`.

```bash
# backtest su più campionati, solo bookmaker scelti
python -m betmodel sharp --league I1,I2,SP2,T1 --books b365,bw,wh --min-edge 0.02

# opportunità sulle prossime partite (servono quote Pinnacle nel file fixtures)
python -m betmodel sharp-scan --league I1,I2
```

Con `--model-filter 0` si punta solo quando anche il modello Dixon-Coles
(stimato su gol e tiri in porta, `--goals-weight 0.5`) vede valore nella quota.
Sullo storico dimezza le scommesse ma alza la CLV media (da circa +3% a +4/5%).

```bash
python -m betmodel sharp --league I1,I2 --model-filter 0

# compromesso: modello d'accordo, oppure edge Pinnacle >= 3% anche senza modello
python -m betmodel sharp --league I1,I2 --model-filter 0 --override-edge 0.03

# il modello da solo aggiunge informazione alle quote Pinnacle? (log loss)
python -m betmodel evaluate --league I1
```

Limiti da tenere presenti:
- le quote dei CSV sono una fotografia (di solito il venerdì): con quote in
  tempo reale le occasioni sono di più ma durano poco;
- i bookmaker morbidi limitano rapidamente i conti che vincono;
- le quote dei siti italiani (.it) possono differire da quelle internazionali
  presenti nei CSV, e non tutti i bookmaker sono autorizzati in Italia (ADM).

## Come leggere il backtest

- **ROI**: profitto / puntato. Il margine del bookmaker parte da −4/−8%.
- **z-score**: se `|z| < 2` il risultato può essere solo fortuna. Servono
  centinaia o migliaia di scommesse per conclusioni solide.
- **CLV** (closing line value): quanto la tua quota batte la quota Pinnacle
  di chiusura senza margine. Una CLV media positiva e stabile è il segnale
  più affidabile di un vantaggio reale; un ROI positivo con CLV negativa è
  quasi sempre fortuna.
- **Quote `max`**: sono le migliori del mercato, difficili da ottenere davvero.
  `avg` o `b365` sono più realistiche.

## Struttura

- `betmodel/data.py` – download e normalizzazione dei CSV di football-data.co.uk
- `betmodel/model.py` – modello Dixon-Coles (numpy puro)
- `betmodel/odds.py` – margine, probabilità eque, edge, Kelly
- `betmodel/backtest.py` – backtest walk-forward e metriche
- `betmodel/cli.py` – comandi da terminale
- `tests/` – test su campionati simulati

```bash
python -m pytest
```

Gioca responsabilmente: il gioco è vietato ai minori e può causare dipendenza.
