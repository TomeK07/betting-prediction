from datetime import date

import yaml

from model.team_profile import detect_changes, write_profile


def _b(coach="A", five=("P1", "P2", "P3", "P4", "P5"), players=("P1", "P2", "P3", "P4", "P5", "P6")):
    return {"natprevari": 5, "do_datum": "2026-10-28", "tempo": {"poseda_po_natprevar": 72, "rang": 1, "liga": 75},
            "efikasnost": {"ortg_surov": 110, "drtg_surov": 100, "ortg_prilagoden": 111, "drtg_prilagoden": 101,
                           "rang_napad": 3, "rang_odbrana": 2, "liga_ortg": 110},
            "dvp": {}, "nositeli": [], "rotacija": [], "startna_petorka_posledni_5": list(five), "trener": coach,
            "doma": {"n": 3, "poeni": 80, "primeni": 70}, "gosti": {"n": 2, "poeni": 75, "primeni": 72},
            "forma_posledni_5": {"pobedi": 4, "porazi": 1, "poeni": 80, "primeni": 70, "neto_rejting": 8},
            "trojki": {"3par_napad": 40, "3par_dozvoluva": 38, "3p_pct": 36, "liga_3par": 41},
            "skokovi": {"oreb_pct": 30, "dreb_pct": 70}, "igraci_vo_zapisnik": list(players)}


def test_detect_changes_coach_five_and_new_player():
    ch = detect_changes(_b(), _b(coach="B", five=("P1", "P2", "P3", "P4", "P6"), players=("P1", "P2", "P3", "P4", "P5", "P6", "P7")))
    assert any("НОВ ТРЕНЕР" in c for c in ch) and any("СТАРТНА ПЕТОРКА" in c for c in ch)
    assert any("P7" in c for c in ch)
    assert detect_changes(None, _b()) == []                        # first profile: nothing to compare


def test_write_profile_keeps_approved_description_and_appends_changes(tmp_path):
    folder = tmp_path / "euroleague" / "PAN"
    write_profile(folder, "Евролига", "PAN", _b(), date(2026, 10, 22))
    prof = yaml.safe_load((folder / "profil.yaml").read_text(encoding="utf-8"))
    prof["opis"] = {"shema_odbrana": "SWITCH", "izvor_url": "https://x", "citat": "they switch everything"}
    (folder / "profil.yaml").write_text(yaml.safe_dump(prof, allow_unicode=True), encoding="utf-8")
    write_profile(folder, "Евролига", "PAN", _b(coach="B"), date(2026, 10, 29))
    again = yaml.safe_load((folder / "profil.yaml").read_text(encoding="utf-8"))
    assert again["opis"]["shema_odbrana"] == "SWITCH"            # numbers refresh never touches OPIS
    log = (folder / "promeni.md").read_text(encoding="utf-8")
    assert "2026-10-29" in log and "НОВ ТРЕНЕР: A → B" in log
    assert (folder / "izvori").is_dir() and "SWITCH" in (folder / "profil.md").read_text(encoding="utf-8")


def test_style_tags_terciles_and_numbers():
    import numpy as np
    import pandas as pd
    from model.team_dossier import style_line, style_tags
    teams = list("ABCDEF")
    s = pd.DataFrame({"n": 5, "pace": [80, 75, 74, 72, 71, 70], "3par": [45, 40, 38, 36, 34, 30],
                      "oreb_pct": [35, 30, 29, 28, 27, 20], "pf": [24, 22, 21, 20, 19, 18],
                      "o_3par": [45, 40, 38, 36, 34, 30], "o_p2": [58, 55, 53, 52, 50, 45], "o_ftr": [30, 28, 27, 26, 25, 20],
                      "dlabocina": [10, 9, 8, 8, 7, 6], "vadi_zvezdi": [7, 5, 4, 3, 2, np.nan],
                      "udel_G": [60, 45, 45, 45, 45, 40], "udel_F": [30, 35, 35, 35, 35, 40], "udel_C": [10, 20, 20, 20, 20, 20]},
                     index=teams)
    a = style_tags("A", s, {"центри": {"poeni_pct": 12.0}, "бекови": {"poeni_pct": -3.0}})
    assert a["tempo"]["oznaka"] == "брзо" and a["tempo"]["brojki"] == {"poseda": 80.0, "rang": 1, "od": 6}
    assert a["napad_preku"]["oznaka"] == "преку бекови"
    assert a["dozvoluva"]["oznaka"].endswith("пушта над лигата на центри (+12% поени)")
    f = style_tags("F", s, {"центри": {"poeni_pct": -1.0}})
    assert f["tempo"]["oznaka"] == "бавно" and f["rotacija"]["oznaka"] == "кратка"
    assert f["vadenje_zvezdi"]["oznaka"] is None          # no big-margin games → no label, no guess
    assert "силна заштита на кош" in f["dozvoluva"]["oznaka"]
    assert style_tags("C", s, None)["napad_preku"]["oznaka"] == "избалансиран"
    assert "80.0 поседи, 1/6" in style_line(a)
