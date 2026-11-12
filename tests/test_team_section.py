from pipeline.team_section import identity, trust


def test_trust_labels():
    assert trust("nba", "vkupno") == "backtest: без предност против пазарот"
    assert trust("nba", "hendikep") == "backtest: без предност против пазарот"
    assert trust("nba", "tim_vkupno").startswith("непроверено")
    assert trust("euroleague", "vkupno") == "непроверено (нема историски линии)"


def test_identity_from_ranks():
    b = lambda n, o: {"efikasnost": {"rang_napad": n, "rang_odbrana": o}}
    assert identity(b(18, 4), 20) == "ѕид (одбрана)"
    assert identity(b(2, 17), 20) == "шутаут (напад)"
    assert identity(b(9, 11), 20) == "среден"
    assert identity({}, 20) == "—"
