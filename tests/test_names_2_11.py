"""Tome 2.11: four errors in povredi_2026-11-02 (names with suffixes, Jalen/Jaylin Williams, effect of OLD absences)."""
import pandas as pd

from model import absences as ab
from model.injury_nba import split_name
from model.logs_common import clean_suffix, disambiguate_labels


def test_suffix_names():
    assert clean_suffix("Udeh, Jr.") == "Udeh Jr."
    assert clean_suffix("Hardaway jr") == "Hardaway Jr."
    assert clean_suffix("Murphy III") == "Murphy III"
    assert clean_suffix("Davis") == "Davis"
    assert split_name("Udeh, Jr., Ernest") == ("Udeh Jr.", "Ernest")
    assert split_name("Porter Jr., Michael") == ("Porter Jr.", "Michael")


def test_one_comma_name_does_not_flip_the_whole_nba_roster():
    roster = pd.DataFrame({"team": "CLE", "player_code": ["1", "2", "3"],
                           "player": ["Ernest Udeh, Jr.", "Jaylon Tyson", "Peyton Watson"]})
    assert ab.match_player("Tyson", "Jaylon", "CLE", roster)[0] == "2"
    assert ab.match_player(*split_name("Udeh, Jr., Ernest"), "CLE", roster)[0] == "1"


def test_jalen_and_jaylin_williams():
    roster = pd.DataFrame({"team": "OKC", "player_code": ["1", "2"], "player": ["Jalen Williams", "Jaylin Williams"]})
    assert ab.match_player("Williams", "Jalen", "OKC", roster)[0] == "1"
    assert ab.match_player("Williams", "Jaylin", "OKC", roster)[0] == "2"
    r = roster.assign(label="Williams J. (OKC)")
    logs = pd.DataFrame({"player_code": ["2"], "player": ["Jaylin Williams"], "label": ["Williams J. (OKC)"]})
    disambiguate_labels(r, logs)
    assert r["label"].tolist() == ["Williams Jal. (OKC)", "Williams Jay. (OKC)"]
    assert logs["label"].tolist() == ["Williams Jay. (OKC)"]           # same player → same label everywhere


def test_same_person_two_codes_keeps_label():
    r = pd.DataFrame({"player_code": ["1", "2"], "player": ["MAJAUSKAS, AURIMAS"] * 2, "label": ["Majauskas A. (KLA)"] * 2})
    disambiguate_labels(r)
    assert r["label"].tolist() == ["Majauskas A. (KLA)"] * 2
