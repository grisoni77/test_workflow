"""Test su dati simulati: verificano che il codice sia corretto, non che la strategia guadagni."""

import numpy as np
import pandas as pd
import pytest

from betmodel import backtest, data
from betmodel.model import DixonColes
from betmodel.odds import fair_probs, kelly, overround


def simulate_league(n_teams=20, seasons=3, seed=0, home_adv=1.3):
    rng = np.random.default_rng(seed)
    teams = [f"T{i:02d}" for i in range(n_teams)]
    att = np.exp(rng.normal(0, 0.3, n_teams))
    att /= att.mean()
    dfn = np.exp(rng.normal(0, 0.3, n_teams))
    rows = []
    day = pd.Timestamp("2020-08-20")
    for _ in range(seasons):
        pairs = [(h, a) for h in range(n_teams) for a in range(n_teams) if h != a]
        rng.shuffle(pairs)
        for k in range(0, len(pairs), n_teams // 2):
            for h, a in pairs[k:k + n_teams // 2]:
                lam, mu = att[h] * dfn[a] * home_adv, att[a] * dfn[h]
                rows.append({"date": day, "home": teams[h], "away": teams[a],
                             "hg": int(rng.poisson(lam)), "ag": int(rng.poisson(mu)),
                             "lam": lam, "mu": mu})
            day += pd.Timedelta(days=7)
        day += pd.Timedelta(days=60)
    return pd.DataFrame(rows), dict(zip(teams, att)), dict(zip(teams, dfn))


def true_1x2(lam, mu, n=10):
    from math import exp, factorial
    ph = [exp(-lam) * lam ** k / factorial(k) for k in range(n + 1)]
    pa = [exp(-mu) * mu ** k / factorial(k) for k in range(n + 1)]
    m = np.outer(ph, pa)
    m /= m.sum()
    return np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()


def add_odds(df, margin=0.05, mispriced_home=False):
    df = df.copy()
    for col in ["H", "D", "A"]:
        df[f"odds_avg_{col}"] = np.nan
    for i, r in df.iterrows():
        p = np.array(true_1x2(r["lam"], r["mu"]))
        if mispriced_home:
            p = np.array([1 / 3, 1 / 3, 1 / 3])  # mercato "cieco"
        odds = 1 / (p * (1 + margin))
        df.loc[i, ["odds_avg_H", "odds_avg_D", "odds_avg_A"]] = odds
    return df


# --- quote ---

def test_fair_probs_sum_to_one_and_remove_margin():
    odds = [1.80, 3.60, 4.50]
    assert overround(odds) > 0
    for method in ("proportional", "power"):
        p = fair_probs(odds, method)
        assert p.sum() == pytest.approx(1)
        assert np.all(p < 1 / np.array(odds))


def test_power_method_shifts_margin_to_longshots():
    odds = [1.30, 5.50, 10.0]
    prop, power = fair_probs(odds, "proportional"), fair_probs(odds, "power")
    assert power[0] > prop[0]
    assert power[2] < prop[2]


def test_power_method_with_negative_overround():
    p = fair_probs([2.1, 3.9, 4.4], "power")  # somma implicite < 1
    assert p.sum() == pytest.approx(1)


def test_kelly():
    assert kelly(0.5, 1.9) == 0
    assert kelly(0.6, 2.0, fraction=1, cap=1) == pytest.approx(0.2)
    assert kelly(0.9, 3.0, fraction=1, cap=0.05) == 0.05


# --- modello ---

def test_model_recovers_true_parameters():
    df, att, dfn = simulate_league(seasons=4)
    m = DixonColes(xi=0).fit(df)
    teams = sorted(att)
    assert np.corrcoef([att[t] for t in teams], [m.attack[t] for t in teams])[0, 1] > 0.9
    assert np.corrcoef([dfn[t] for t in teams], [m.defence[t] for t in teams])[0, 1] > 0.85
    assert m.home_adv == pytest.approx(1.3, abs=0.1)


def test_prediction_is_a_distribution():
    df, *_ = simulate_league(seasons=1)
    m = DixonColes().fit(df)
    p = m.predict("T00", "T01")
    assert p.home_win + p.draw + p.away_win == pytest.approx(1)
    assert p.over25 + p.under25 == pytest.approx(1)
    assert m.score_matrix("T00", "T01").min() >= 0


# --- dati ---

def test_normalize_handles_aliases_and_dates(tmp_path):
    csv = tmp_path / "x.csv"
    csv.write_text(
        "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,BbMxH,BbMxD,BbMxA,PSCH,PSCD,PSCA\n"
        "I1,20/08/12,Juventus ,Parma,2,0,1.5,4.0,7.0,1.45,4.2,8.0\n"
        "I1,19/08/2012,Milan,Inter,1,1,2.5,3.2,3.0,2.4,3.3,3.1\n"
        ",,,,,,,,,,,\n"
    )
    df = data.load_csv(csv)
    assert list(df["home"]) == ["Milan", "Juventus"]  # ordinate per data, spazi rimossi
    assert df.loc[1, "odds_max_H"] == 1.5
    assert df.loc[1, "close_ps_A"] == 8.0
    assert df["odds_b365_H"].isna().all()


def test_season_code():
    assert data.season_code(2024) == "2425"
    assert data.season_code(1999) == "9900"


# --- backtest ---

def test_backtest_has_no_lookahead_and_finds_mispriced_market():
    df, *_ = simulate_league(seasons=3, seed=1)
    df = add_odds(df, mispriced_home=True)
    cfg = backtest.Config(markets=("1x2",), min_odds=1.0, max_odds=10, warmup_days=200)
    res = backtest.run(df, cfg)
    s = res.summary
    assert s["bets"] > 100
    assert res.bets["date"].min() >= df["date"].min() + pd.Timedelta(days=200)
    # un mercato che dà 33% a tutto è battibile: il modello deve trovarlo
    assert s["roi"] > 0.05


def test_backtest_cannot_beat_efficient_market_with_margin():
    df, *_ = simulate_league(seasons=3, seed=2)
    df = add_odds(df, margin=0.06)
    cfg = backtest.Config(markets=("1x2",), min_edge=0.0, warmup_days=200)
    s = backtest.run(df, cfg).summary
    # quote "vere" con margine: nel lungo periodo non si guadagna in modo significativo
    assert s["z_score"] < 2


# --- strategia sharp vs soft ---

def test_sharp_bets_only_when_soft_beats_pinnacle():
    from betmodel import sharp
    base = {"date": pd.Timestamp("2024-01-01"), "hg": 1, "ag": 0,
            "odds_ps_H": 2.0, "odds_ps_D": 3.5, "odds_ps_A": 4.0}
    rows = pd.DataFrame([
        {**base, "home": "A", "away": "B", "odds_b365_H": 1.90, "odds_wh_H": 1.95},  # sotto Pinnacle
        {**base, "home": "C", "away": "D", "odds_b365_H": 1.90, "odds_wh_H": 2.30},  # wh fuori linea
    ])
    cfg = sharp.SharpConfig(books=("b365", "wh"), markets=("1x2",), min_edge=0.02)
    bets = sharp.find_bets(rows, cfg)
    assert [(b["home"], b["pick"], b["book"], b["odds"]) for b in bets] == [("C", "H", "wh", 2.30)]
    res = sharp.run(rows, cfg)
    assert res.summary["bets"] == 1 and res.summary["profit"] == pytest.approx(1.30)
