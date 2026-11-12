import pytest

from model import picks as pk


def _c(**kw):
    base = dict(liga="nba", game_code="1", natprevar="A - B", tim="A", protivnik="B", igrac="X Y. (A)", player_code="9",
                pazar="poeni", strana="pod", linija=20.5, mu=17.0, p_model=0.62, min_proj=32.0)
    base.update(kw)
    return pk.Candidate(**base)


def test_probabilities_hand_examples():
    assert pk.p_normal_over(170.0, 18.0, 170.0) == pytest.approx(0.5)
    assert pk.p_normal_over(176.0, 18.0, 170.0) == pytest.approx(0.6306, abs=1e-3)    # Φ(1/3)
    # home −6.5 with expected margin +6.5 → 50%
    assert pk.p_handicap_cover(6.5, 12.0, -6.5) == pytest.approx(0.5)
    assert pk.p_count_over(6.0, 1e6, 5.5) == pytest.approx(0.5543, abs=1e-3)


def test_order_calibrate_then_price_best_book():
    c = _c(liga="euroleague", p_model=0.70)
    pk.calibrate(c, 0.5)                                   # EL shrink: 0.5 + 0.5·0.2 = 0.60
    assert c.p_kal == pytest.approx(0.60)
    pk.price(c, "mozzart", 1.90, 1.90, nat_id="m1")
    pk.price(c, "evrotip", 1.95, 1.85, nat_id="e1")
    assert c.kladilnica == "evrotip" and c.ev == pytest.approx(0.60 * 1.95 - 1)
    assert c.p_pazar == pytest.approx((1 / 1.95) / (1 / 1.95 + 1 / 1.85))
    assert c.nat_ids == {"mozzart": "m1", "evrotip": "e1"}


def test_kelly_quarter_capped():
    # PROCES §5 example: p 0.60 at 1.90 → full 15.6% → ¼ = 3.9% → cap 2%
    assert pk.kelly_fraction(0.60, 1.90) == pytest.approx(0.02)
    assert pk.kelly_fraction(0.53, 1.90) == pytest.approx(0.25 * (0.53 * 1.9 - 1) / 0.9)
    assert pk.kelly_fraction(0.45, 1.90) == 0.0


def test_named_filters():
    c = _c(p_model=0.62, min_proj=18.0)
    pk.calibrate(c, None)
    pk.price(c, "mozzart", 1.90, 1.90)
    pk.apply_filters(c, "nba", blocked_pod=True)
    rules = " ".join(c.pravila)
    assert "МИНУТИ < 24" in rules and "правило 4" in rules
    alarm = _c(p_model=0.80)
    pk.calibrate(alarm, None)
    pk.price(alarm, "evrotip", 2.50, 1.55)
    pk.apply_filters(alarm, "nba", blocked_pod=False)
    assert any("ПРОВЕРИ_ПОДАТОЦИ" in r for r in alarm.pravila)          # error 16.10 alarm
    over = _c(liga="euroleague", strana="nad", mu=12.0, linija=10.5, p_model=0.70)   # 12.0 < 10.5·1.2 = 12.6
    pk.calibrate(over, None)
    pk.price(over, "mozzart", 1.90, 1.90)
    pk.apply_filters(over, "euroleague", blocked_pod=False)
    assert any("правило 6" in r for r in over.pravila) and any("65%" in r for r in over.pravila)


def test_ticket_rules():
    def ok(game, tim, side, pazar="poeni", ev=0.10, liga="nba"):
        c = _c(game_code=game, tim=tim, strana=side, pazar=pazar, liga=liga)
        c.p_kal, c.ev, c.kvota = 0.6, ev, 1.9
        return c
    a, b = ok("1", "A", "pod", ev=0.20), ok("1", "A", "pod", ev=0.15)          # same game → only one
    d = ok("2", "C", "nad", ev=0.12)
    e = ok("3", "D", "nad", ev=0.11)                                           # second points НАД → skipped (rule 6)
    f = ok("4", "E", "pod", pazar="asistencii", ev=0.05)
    z = ok("5", "F", "pod", ev=0.50); z.zavisi = True                          # ЗАВИСИ_ОД_СОСТАВ never in the ticket
    t = pk.build_ticket([a, b, d, e, f, z])
    assert [c.game_code for c in t] == ["1", "2", "4"]
    assert pk.build_ticket([a]) == []                                          # one pick is not a ticket


def test_bookmaker_names_rule_2():
    players = [{"player_code": "1", "first": "Aaron", "last": "Wiggins"}, {"player_code": "2", "first": "Andrew", "last": "Wiggins"},
               {"player_code": "3", "first": "CJ", "last": "McCollum"}, {"player_code": "4", "first": "ROKAS", "last": "JOKUBAITIS"}]
    assert pk.match_book_player("Aar.Wiggins", players)["player_code"] == "1"
    assert pk.match_book_player("A. Wiggins", players) is None                 # two Wiggins A. → never guess
    assert pk.match_book_player("C.J.McCollum", players)["player_code"] == "3"
    assert pk.match_book_player("CJ McCollum", players)["player_code"] == "3"
    assert pk.match_book_player("R. Jokubaitis", players)["player_code"] == "4"


def test_ev_above_15_percent_is_check_data_and_not_in_ticket(monkeypatch):
    monkeypatch.setattr(pk, "PLAYER_PROPS_PAPER", False)
    c = _c(p_model=0.62)
    pk.calibrate(c, None)
    pk.price(c, "evrotip", 1.90, None)                          # EV = 0.62·1.90 − 1 = +17.8%, no other side → no pp alarm
    pk.apply_filters(c, "nba", blocked_pod=False)
    assert any("EV +17.8% > +15%" in r for r in c.pravila) and c.odbien
    ok = _c(p_model=0.58, game_code="2")
    pk.calibrate(ok, None)
    pk.price(ok, "evrotip", 1.90, None)                         # +10.2% → passes
    pk.apply_filters(ok, "nba", blocked_pod=False)
    assert not ok.pravila
    assert pk.build_ticket([c, ok]) == []                       # the alarm pick never enters; one pick is not a ticket


def test_count_probability_uses_per_mean_dispersion_spec():
    spec = {"r_po_mu": [1.0, 2.0, 12.0, 8.0, 6.0]}
    # mean 0.8 → first range → r = 1 (wider than r = 12 → more mass above 0.5 … and at 0)
    assert pk.p_count_over(0.8, spec, 0.5) == pytest.approx(pk.p_count_over(0.8, 1.0, 0.5))
    assert pk.p_count_over(0.8, spec, 1.5) > pk.p_count_over(0.8, 12.0, 1.5)
    assert pk.p_count_over(5.0, spec, 4.5) == pytest.approx(pk.p_count_over(5.0, 8.0, 4.5))


def test_tome_30_10_filters(monkeypatch):
    from model.picks import Candidate, apply_filters, build_ticket
    monkeypatch.setattr(pk, "PLAYER_PROPS_PAPER", False)          # the 30.10 filters alone
    def cand(**kw):
        base = dict(liga="nba", game_code="g", natprevar="A - B", tim="A", protivnik="B", igrac="X", player_code="1",
                    pazar="poeni", strana="nad", linija=20.5, mu=26.0, p_model=0.6, p_kal=0.6, min_proj=30.0)
        base.update(kw)
        c = Candidate(**base)
        c.ev = kw.get("ev", 0.06)
        return c
    c = cand(ev=0.06)
    apply_filters(c, "nba", False)
    assert c.odbien and any("< 8%" in r for r in c.pravila)
    low = cand(pazar="trojki", strana="pod", linija=1.5, mu=0.8)
    apply_filters(low, "nba", False)
    assert any("ниска линија" in r for r in low.pravila)
    team = cand(igrac=None, player_code=None, pazar="vkupno", linija=220.5, mu=225.0, min_proj=None, game_code="h")
    apply_filters(team, "nba", False)
    assert not team.odbien and team.samo_hartija
    ok = cand(pazar="skokovi", strana="nad", linija=5.5, mu=6.7, game_code="k")
    apply_filters(ok, "nba", False)
    assert not ok.pravila
    assert team not in build_ticket([team, ok, cand(pazar="asistencii", linija=4.5, mu=6.0, game_code="m", ev=0.1)])


def test_player_props_paper_only_tome_31_10():
    c = _c(pazar="skokovi", strana="nad", linija=5.5, mu=6.7, p_model=0.6)
    pk.calibrate(c, None)
    pk.price(c, "mozzart", 1.85, 1.95)
    pk.apply_filters(c, "euroleague", blocked_pod=False)
    assert c.samo_hartija and not c.odbien
    assert c not in pk.build_ticket([c, _c(game_code="2", igrac=None, player_code=None, pazar="vkupno")])


def test_propose_tickets_max_two_per_player():
    from model.picks import Candidate, propose_tickets
    cs = []
    for g in range(9):
        for k, ev in enumerate((0.10, 0.08)):
            cs.append(Candidate("nba", f"G{g}", f"G{g}", f"T{g}", None, f"P{g}{k}", f"{g}{k}", "asistencii", "nad", 4.5,
                                5.0, 0.6, p_kal=0.6, kvota=1.85, ev=ev - g * 0.001))
    out = propose_tickets(cs)
    seen = {}
    for name, tk, why in out:
        assert len({c.game_code for c in tk}) == len(tk)          # 1 pick per game per ticket
        for c in tk:
            seen[c.player_code] = seen.get(c.player_code, 0) + 1
    assert max(seen.values()) <= 2
    assert [len(tk) for _, tk, _ in out] == [3, 5, 8]
    two_games = [c for c in cs if c.game_code in ("G0", "G1")]
    small, mid, big = propose_tickets(two_games)
    assert len(small[1]) == 2 and mid[1] == [] and "само 2" in mid[2] and big[1] == []
