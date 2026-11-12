import pytest

from research.llm_client import parse_json_text
from research.llm_eval import score_t1, score_t9
from research.testset_build import _statement, surname, text_statuses

T1_EX = {"tekst": "Detroit Pistons — injuries\nDuren will be re-evaluated in two weeks due to a right hamstring strain.",
         "izvor_url": "u", "ocekuvano": [{"igrac": "Jalen Duren", "tim": "Detroit Pistons", "status": "OUT"}]}


def item(status, quote, igrac="Jalen Duren"):
    return {"items": [{"igrac": igrac, "tim": "Detroit Pistons", "status": status, "citat": quote}]}


def test_t1_correct_status_and_real_quote():
    sc = score_t1(item("OUT", "Duren will be re-evaluated in two weeks"), T1_EX)
    assert (sc["tochni"], sc["vkupno"], sc["izmisleni"]) == (1, 1, 0)


def test_t1_invented_quote_is_wrong_even_with_right_status():
    sc = score_t1(item("OUT", "Duren is out for the season"), T1_EX)
    assert (sc["tochni"], sc["izmisleni"]) == (0, 1)


def test_t1_near_and_missed():
    assert score_t1(item("DOUBTFUL", "Duren will be re-evaluated in two weeks"), T1_EX)["blisku"] == 1
    sc = score_t1({"items": []}, T1_EX)
    assert (sc["tochni"], sc["propushteni"]) == (0, 1)


def test_t1_forbidden_field_rejects_answer():
    ans = item("OUT", "Duren will be re-evaluated in two weeks")
    ans["probability"] = 0.9
    assert score_t1(ans, T1_EX)["tochni"] == 0


T9_EX = {"tekst": "Head coach Ibon Navarro praised Nwora. Former coach Dusan Alimpijevic left. Opponent Mike James scored.",
         "ocekuvano": {"trener": "Ibon Navarro", "sostav": ["Jordan Nwora", "Chima Moneke"]}}


def test_t9_coach_and_roster_checked_against_official():
    ans = {"trener": {"ime": "Ibon Navarro", "citat": "Head coach Ibon Navarro praised Nwora"},
           "klucni_igraci": [{"igrac": "Jordan Nwora", "uloga": "scorer", "citat": "praised Nwora"},
                             {"igrac": "Mike James", "uloga": "x", "citat": "Opponent Mike James scored"}]}
    sc = score_t9(ans, T9_EX)
    assert (sc["tochni"], sc["vkupno"], sc["izmisleni"]) == (2, 3, 0)     # Mike James is not on the roster


def test_t9_wrong_coach_and_missed_coach():
    sc = score_t9({"trener": {"ime": "Dusan Alimpijevic", "citat": "Former coach Dusan Alimpijevic left"}}, T9_EX)
    assert (sc["tochni"], sc["vkupno"]) == (0, 1)
    assert score_t9({"trener": None, "klucni_igraci": []}, T9_EX)["propushteni"] == 1


def test_parse_json_text_strips_fences():
    assert parse_json_text('```json\n{"items": []}\n```') == {"items": []}
    assert parse_json_text('Here: {"a": 1} done') == {"a": 1}
    with pytest.raises(ValueError):
        parse_json_text("no json")


def test_el_statement_clauses_and_surname():
    st = _statement("Tomer Ginat is available while Bar Timor is ruled out for Hapoel’s game against Besiktas")
    assert st == [("Tomer Ginat", "AVAILABLE"), ("Bar Timor", "OUT")]
    assert _statement("Dubai has not yet sent an injury report for its game") == []
    assert surname("Porter Jr., Michael") == "porter" and surname("NAVARRO, IBON") == "navarro"
    assert text_statuses("He is listed as questionable") == {"QUESTIONABLE"}


def test_stat_filter_and_quota_breaker(monkeypatch):
    from research import llm_client
    from research.t10_news import STAT_RE
    assert STAT_RE.search("Returned after a torn ACL, scoring 15 points")
    assert STAT_RE.search("Crvena Zvezda won 84-82 over ASVEL")
    assert not STAT_RE.search("was ruled out for Rounds 5 and 6 with an ankle injury")
    assert not STAT_RE.search("exercised the option for the 2027-28 season")
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("LLM_API_KEY", "x")
    monkeypatch.setattr(llm_client, "EXHAUSTED", {"m-spent"})
    c = llm_client.LLMClient("m-spent")
    try:
        c.ask_json_meta("s", "never-cached text 31.10", "test-v0")
        raise AssertionError("expected fail-fast")
    except RuntimeError as exc:
        assert "квотата" in str(exc)


def test_t10_junk_filter_and_effect():
    from research.llm_eval import effect_of, score_t10
    from research.t10_news import relevant
    assert relevant("Mavericks waive Dwight Powell", "", ["Mavericks", "Dallas"], set())
    assert relevant("Kyrie explains offer to pay fine", "", ["Mavericks"], {"kyrie", "irving"})
    assert not relevant("2026 NBA Cup: Schedule, format, groups", "How to watch", ["Mavericks", "Dallas"], {"irving"})
    assert effect_of("Miljenovic is da valutare (ankle)") == "QUESTIONABLE"
    assert effect_of("ruled out for Friday") == "OUT"
    assert effect_of("is available again") == "AVAILABLE"
    assert effect_of("is unavailable") == "OUT"
    ex = {"tekst": "Duren will be re-evaluated in two weeks and will miss Friday.",
          "ocekuvano": [{"igrac": "Jalen Duren", "tipovi": ["ПОВРЕДА", "СОСТАВ"], "vlijanie": "OUT"}]}
    good = {"items": [{"tip": "INJURY", "igrac": "Jalen Duren", "sto": "will miss Friday", "vlijanie": "not stated",
                       "citat": "will miss Friday"}]}
    assert score_t10(good, ex)["tochni"] == 1
    wrong_eff = {"items": [{**good["items"][0], "sto": "is questionable"}]}
    s = score_t10(wrong_eff, ex)
    assert s["tochni"] == 0 and s["blisku"] == 1
    wrong_type = {"items": [{**good["items"][0], "tip": "CONTRACT"}]}
    assert score_t10(wrong_type, ex)["tochni"] == 0
    invented = {"items": [{**good["items"][0], "citat": "out for the season"}]}
    s = score_t10(invented, ex)
    assert s["izmisleni"] == 1 and s["propushteni"] == 1


def test_rotowire_parser(tmp_path):
    from datetime import date
    from research.t10_news import _rotowire

    class F:
        def get(self, url, path, params=None):
            return ('<div class="news-update"><a class="news-update__player-link" href="/x">Stefan Miljenovic</a>'
                    '<div class="news-update__headline">Questionable for Thursday</div></div>'
                    '<div class="news-update__timestamp">October 29, 2026</div>'
                    '<div class="news-update__news">Miljenovic (ankle) is questionable, per the club.</div>'
                    '<div class="news-update"><a class="news-update__player-link" href="/y">Furkan Korkmaz</a>'
                    '<div class="news-update__timestamp">October 30, 2026</div>'
                    '<div class="news-update__news">Korkmaz is out.</div>').encode()
    out = _rotowire(F(), tmp_path, "2026-10-01", date(2026, 10, 31), ["Stefan Miljenovic", "Chris Jones"])
    assert len(out) == 1 and out[0]["objaveno"] == "2026-10-29"
    assert "Miljenovic (ankle) is questionable" in out[0]["tekst"] and "Korkmaz" not in out[0]["tekst"]


def test_excerpt_keeps_title_and_team_sentences_verbatim():
    from research.t10_news import excerpt
    filler = " ".join(f"Other team note number {i} about the league." for i in range(60))
    text = f"Nets news roundup\n{filler} Cam Thomas is out with a hamstring strain. He will be re-evaluated in two weeks. {filler}"
    out = excerpt(text, ["Nets", "Thomas"])
    assert out.startswith("Nets news roundup")
    assert "Cam Thomas is out with a hamstring strain." in out and "He will be re-evaluated in two weeks." in out
    assert len(out) < len(text) / 4
    assert all(line in text for line in out.split("\n"))          # verbatim → quotes still checkable in the source
    assert excerpt("short text", ["X"]) == "short text"
