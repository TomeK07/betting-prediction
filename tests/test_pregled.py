from model import picks as pk
from pipeline import pregled_cmd as pr


def _c(**kw):
    base = dict(liga="nba", game_code="g1", natprevar="PHI - CHI", tim=None, protivnik=None, igrac=None, player_code=None,
                pazar="vkupno", strana="nad", linija=230.5, mu=233.0, p_model=0.6, p_kal=0.6)
    base.update(kw)
    return pk.Candidate(**base)


def test_market_percent_is_the_devig_of_that_line_mozzart_first():
    c = _c(kvoti={"mozzart": (1.30, 3.40), "evrotip": (1.35, 3.10)})
    p, book = pr.market_p(c)
    assert book == "mozzart" and abs(p - (1 / 1.30) / (1 / 1.30 + 1 / 3.40)) < 1e-9
    assert pr.market_p(_c(kvoti={"evrotip": (1.5, 2.6)}))[1] == "evrotip"
    assert pr.market_p(_c(kvoti={"mozzart": (1.5, None)}))[0] is None          # one price only → no number


def test_levels_marks_and_own_q_capped_to_red():
    import pandas as pd
    players = pd.DataFrame({"player_code": ["7"], "team": ["PHI"], "min_bez": [33.0], "q_self": [True]})
    sure = _c(kvoti={"mozzart": (1.20, 4.50)}, p_kal=0.70)                       # market ≈ 79%
    q = _c(igrac="Embiid J. (PHI)", player_code="7", tim="PHI", pazar="poeni", linija=20.5, p_kal=0.5,
           kvoti={"mozzart": (1.20, 4.50)}, min_proj=33.0, zavisi=True)
    weak = _c(kvoti={"mozzart": (1.90, 1.90)})                                   # 50% → not shown
    out = pr.options([sure, q, weak], {}, players, frozenset())
    assert len(out) == 2
    s = next(o for o in out if o["c"] is sure)
    assert s["level"].startswith("🟢") and s["model"] == "❌"                  # model 70% vs market 79%
    o = next(o for o in out if o["c"] is q)
    assert o["level"].startswith("🔴") and "самиот е Q" in o["risks"][0]


def test_example_ticket_chance_is_product_of_different_games():
    opts = {f"g{i}": [{"c": _c(game_code=f"g{i}", natprevar=f"G{i}"), "p": 0.8, "level": pr.LEVELS[0][0],
                       "own_q": False, "kvota": 1.2}] for i in range(3)}
    text = "\n".join(pr.example_tickets(opts))
    assert "шанса на тикетот **51%**" in text and "квота 1.73" in text
