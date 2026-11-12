from model.scenarios import PlayerContext, blocked_sides, player_scenarios, team_scenarios


def _ctx(**kw):
    base = dict(league="euroleague", position="C", starter=True, team_is_favorite=True, expected_margin=3.0,
                dvp_pct=0.0, team_pace_vs_lg=0.0, opp_pace_vs_lg=0.0, team_drtg_vs_lg=0.0, opp_drtg_vs_lg=0.0,
                fouls_per36=2.0, bench_teammate_gain=0.0, returning_teammate=False, new_teammate_absence=False,
                first_scorer_of_depleted=False)
    base.update(kw)
    return PlayerContext(**base)


def codes(c):
    return {(s.side, s.code) for s in player_scenarios(c)}


def test_minus_scenarios():
    c = _ctx(dvp_pct=-12, team_pace_vs_lg=-3, opp_pace_vs_lg=-2.5, team_drtg_vs_lg=-2, opp_drtg_vs_lg=-1,
             expected_margin=11, bench_teammate_gain=4, returning_teammate=True, fouls_per36=3.7)
    assert codes(c) >= {("pod", "DVP_SILNA"), ("pod", "ZID"), ("pod", "BLOWOUT_FAVORIT"), ("pod", "KLUPA_ZEMA"),
                        ("pod", "SOIGRAC_SE_VRAKJA"), ("pod", "FAULI")}       # EL centre: 3.7 fouls/36 with 5-foul limit


def test_plus_scenarios_and_rule_4():
    c = _ctx(dvp_pct=10, team_pace_vs_lg=3, opp_pace_vs_lg=2, new_teammate_absence=True, first_scorer_of_depleted=True)
    assert codes(c) >= {("nad", "DVP_SLABA"), ("nad", "BRZO"), ("nad", "NOVO_OTSUSTVO"), ("nad", "PRV_STRELEC")}
    assert blocked_sides(player_scenarios(c)) == {"pod"}


def test_nba_foul_threshold_and_blowout_threshold():
    assert ("pod", "FAULI") not in codes(_ctx(league="nba", fouls_per36=3.7))
    assert ("pod", "BLOWOUT_FAVORIT") not in codes(_ctx(league="nba", expected_margin=11))
    assert ("pod", "BLOWOUT_FAVORIT") in codes(_ctx(league="nba", expected_margin=12))


def test_team_scenarios():
    s = {x.code for x in team_scenarios("nba", -3, -2.5, -1, -2, 4)}
    assert s == {"ZID"}
    assert {x.code for x in team_scenarios("nba", 3, 2.5, 1, 2, 13)} == {"SUTAUT", "BLOWOUT"}
