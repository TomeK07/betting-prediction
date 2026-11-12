"""`python main.py llm …` — test sets and quality control of the free LLM researchers (LLM_ISTRAZUVACI §6).

  llm testset            build tests/llm/T1.jsonl, T9.jsonl, T10.jsonl from official data (no AI in the answers)
  llm test [--modeli] [--zadaci T1,T9,T10]   run the models on the test sets → archive/llm_log.db + table
  llm rezime             the table again + the two-model agreement rate per model (СЛЕДИ checker)
  llm dnevno             T10 news for teams playing today/tomorrow — САМО РАЧНО кога Томе ќе каже (не автоматски)
  llm site               T10 (7 days) + T9 for all 82 — исто САМО РАЧНО; resumes after the quota
  llm napredok           table: team | news | T9 | last update → reports/llm_napredok.md
  llm izvadok            T10 with excerpts: official test + real sources full vs excerpt (Tome 1.11, decides USE_EXCERPT)
  llm vesti --tim RED    T10 news for one team (any of the 82)
  llm t9 --tim RED       coach / GM proposal → data/timovi/…/predlozi/ → `profil odobri` (on demand, not daily)
  llm pilot --tim RED    vesti + t9 for one team, with the cost report (reports/pilot_<tim>_<date>.md)
"""
from __future__ import annotations

import json
from datetime import date, timedelta

import pandas as pd

from core.config import Settings
from research.llm_eval import LOG_DB, candidate_models, run_eval, summary
from research.testset_build import build_all

SEASON_START = date(2026, 7, 1)          # Tome 30.10: only season 2026-27 (the summer moves belong to it)
LEAGUES = ["euroleague", "eurocup", "nba"]


def cmd_llm(settings: Settings, akcija: str, modeli: str | None, zadaci: str, today: date,
            tim: str | None = None, league: str = "all") -> int:
    models = [m.strip() for m in modeli.split(",")] if modeli else None
    if akcija == "testset":
        tasks = [t.strip().upper() for t in zadaci.split(",")]
        res = build_all(settings, since_t1=today - timedelta(days=10), since_t9=today - timedelta(days=75)) \
            if {"T1", "T9"} & set(tasks) else {}
        if "T10" in tasks:                    # built FROM T1.jsonl → run after it (or alone: --zadaci T10)
            from research.t10_testset import build_t10
            res["T10"] = build_t10(settings, today)
        print(json.dumps(res, ensure_ascii=False, indent=1))
        print("→ tests/llm/T1.jsonl, T9.jsonl, T10.jsonl (точниот одговор: само официјални извори)")
        return 0
    if akcija == "rezime":
        from research.t10_news import agreement_summary
        print(summary(LOG_DB))
        print("\n## Согласување на двата модели (проверувач со статус СЛЕДИ)\n")
        print(agreement_summary())
        return 0
    if akcija == "dnevno":
        return run_daily(settings, today, league, models)
    if akcija == "site":
        return run_all(settings, today, models)
    if akcija == "izvadok":
        from research.llm_client import task_models
        from research.t10_izvadok import cmd_izvadok
        return cmd_izvadok(settings, today, models or task_models("T10"))
    if akcija == "napredok":
        print(progress_table(settings, today))
        return 0
    if akcija in ("vesti", "t9", "pilot"):
        found = find_team(tim, league)
        if not found:
            return 2
        lg, code = found
        if akcija == "vesti":
            return run_team_news(settings, lg, code, today, models)
        return run_pilot(settings, lg, code, today, models, only_t9=akcija == "t9")
    models = models or candidate_models()
    print(f"Модели ({len(models)}): {', '.join(models)}")
    print(run_eval(models, [t.strip().upper() for t in zadaci.split(",")]))
    return 0


def find_team(code: str | None, league: str) -> tuple[str, str] | None:
    from research.t10_news import team_names
    if not code:
        print("--tim: код на тим (пр. RED, DAL)")
        return None
    hits = [(lg, code.upper()) for lg in LEAGUES if (league in ("all", lg)) and code.upper() in team_names(lg)]
    if len(hits) != 1:
        print(f"„{code}“: " + ("нема во config/timovi_iminja.yaml" if not hits else
                                f"има во повеќе лиги ({', '.join(h[0] for h in hits)}) — додај --league"))
        return None
    return hits[0]


def teams_playing(settings: Settings, days: list[date], league: str = "all") -> list[tuple[str, str, str]]:
    """(league, code, game_date) for every team with a game on one of `days` (schedule.csv), today's games first."""
    out = []
    for lg in LEAGUES if league == "all" else [league]:
        path = settings.paths.processed / lg / settings.league(lg).season / "schedule.csv"
        if not path.exists():
            continue
        s = pd.read_csv(path, dtype={"game_code": str})
        s = s[s["game_date"].isin([d.isoformat() for d in days])]
        for r in s.itertuples():
            for code in (r.home, r.away):
                if isinstance(code, str) and (lg, code) not in {(o[0], o[1]) for o in out}:
                    out.append((lg, code, r.game_date))
    return sorted(out, key=lambda x: (x[2], x[0], x[1]))


def _news(settings: Settings, lg: str, code: str, today: date, models: list[str] | None, days: int = 30):
    from research.llm_client import task_models
    from research.t10_news import injuries_to_vesti, run_news, team_names
    from research.t9_pilot import _official
    from research.testset_build import Fetcher
    from datetime import datetime, timezone
    nm = team_names(lg)[code]
    models = models or task_models("T10")
    coach, roster, labels = _official(settings, Fetcher(settings), lg, code)
    news = run_news(settings, lg, code, nm["ime"], nm["prebaruvanje"], nm["zborovi"], models, today, roster, labels, days)
    n_inj = injuries_to_vesti(settings, lg, nm["ime"], news, task_models("T1"), datetime.now(timezone.utc))
    return news, n_inj, coach, models


def run_team_news(settings: Settings, lg: str, code: str, today: date, models: list[str] | None) -> int:
    news, n_inj, _, models = _news(settings, lg, code, today, models)
    s = news["stats"]
    print(f"{lg}/{code}: извори {s['izvori']} · повици {s['povici']} (кеш {s['od_kesh']}) · токени {s['tokeni']} · "
          f"во vesti.md {len(news['zapisi'])} · еден модел {len(news['eden_model'])} · одбиено {len(news['otfrleno'])} · "
          f"без одговор {len(s['bez_odgovor'])} · → data/vesti {n_inj}")
    return 0


DAILY_DAYS = 7                 # Tome 31.10: the daily fill reads only the last 7 days (quota)


def run_daily(settings: Settings, today: date, league: str, models: list[str] | None) -> int:
    """Tome 31.10: every day only the teams playing today or tomorrow; T10 (injuries/lineups/minutes) before all else."""
    import time
    from pipeline.profile_cmd import cmd_profil
    from research.llm_client import EXHAUSTED
    teams = teams_playing(settings, [today, today + timedelta(days=1)], league)
    print(f"Тимови што играат {today} / {today + timedelta(days=1)}: {len(teams)}")
    t0, rows, touched = time.monotonic(), [], set()
    from research.t10_news import team_names
    for lg, code, gd in teams:
        if code not in team_names(lg):
            rows.append(f"| {gd} | {lg} | {code} | — | — | — | — | нема во config/timovi_iminja.yaml |")
            continue
        news, n_inj, _, used = _news(settings, lg, code, today, models, days=DAILY_DAYS)
        s = news["stats"]
        touched.add(lg)
        rows.append(f"| {gd} | {lg} | {code} | {s['izvori']} | {s['povici']} ({s['od_kesh']} кеш) | {len(news['zapisi'])} | "
                    f"{n_inj} | {len(s['bez_odgovor'])} без одговор |")
        if set(used[:2]) & EXHAUSTED:              # a news item needs BOTH models → do not burn the other one's quota
            rows.append(f"| | | | | | | | **квотата на {', '.join(sorted(set(used[:2]) & EXHAUSTED))} е потрошена — "
                        "останатите тимови подоцна / утре** |")
            break
    for lg in touched:
        cmd_profil(settings, lg, today)                 # profil.md shows the new news (7 days)
    rep = [f"# Дневни вести (T10) · {today}", "", f"Вкупно време: {time.monotonic() - t0:.0f} s", "",
           "| Натпревар | Лига | Тим | извори | повици | vesti.md (нови) | → data/vesti | забелешка |",
           "|---|---|---|---|---|---|---|---|", *rows]
    out = settings.paths.reports / f"vesti_dnevno_{today}.md"
    out.write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(rep))
    print(f"\nИзвештај: {out}")
    return 0


PROGRESS = "_napredok_llm.json"      # data/timovi/: last T10 / T9 date per team (resume after the quota ends)
T9_REFRESH_DAYS = 30                 # coach / GM change rarely
LOCK = "_napredok_llm.lock"          # heartbeat of a running `llm site` (14:00 task must not start a second one)
LOCK_STALE_S = 2 * 3600              # no heartbeat for 2 h = the run died → the lock is ignored


def _progress(teams_dir) -> dict:
    """{"T10": {"lg/CODE": date}, "T9": {...}, "zastanato": {...}}; T9 proposals already on disk count as done."""
    p = teams_dir / PROGRESS
    st = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    st.setdefault("T10", {})
    st.setdefault("T9", {})
    for f in teams_dir.glob("*/*/predlozi/*_T9.yaml"):             # t9-v2 since 31.10 (pilot RED, DAL)
        key, day = f"{f.parts[-4]}/{f.parts[-3]}", f.name[:10]
        if day >= "2026-10-31" and day > st["T9"].get(key, ""):
            st["T9"][key] = day
    return st


QUOTA_LEAGUE_ORDER = ["nba", "euroleague", "eurocup"]   # Tome 1.11: EuroCup last (weak news coverage)


def all_teams_order(settings: Settings, today: date) -> tuple[list[tuple[str, str]], set]:
    """Tome 1.11 quota priority: teams playing TODAY first (daily news), then all others weekly, EuroCup last."""
    from research.t10_news import team_names
    rank = {lg: i for i, lg in enumerate(QUOTA_LEAGUE_ORDER)}
    playing = sorted({(lg, c) for lg, c, _ in teams_playing(settings, [today])}, key=lambda x: (rank[x[0]], x[1]))
    rest = [(lg, c) for lg in QUOTA_LEAGUE_ORDER for c in sorted(team_names(lg)) if (lg, c) not in playing]
    return playing + rest, set(playing)


def run_all(settings: Settings, today: date, models: list[str] | None) -> int:
    """Tome 1.11: T10 (7 days) + T9 for all 82 teams; playing teams daily, the rest weekly; T9 every 30 days.
    Order: T10 for all first (quota priority), then T9. Quota spent → stop, progress saved, the 14:00 task resumes."""
    import os
    import time
    from pipeline.profile_cmd import TEAMS_DIR

    lock = TEAMS_DIR / LOCK
    if lock.exists() and time.time() - lock.stat().st_mtime < LOCK_STALE_S:
        print(f"`llm site` веќе работи (последен знак {time.time() - lock.stat().st_mtime:.0f} s) — не пуштам втор.")
        return 0
    lock.write_text(str(os.getpid()), encoding="utf-8")
    try:
        return _run_all(settings, today, models, lock)
    finally:
        lock.unlink(missing_ok=True)


def _run_all(settings: Settings, today: date, models: list[str] | None, lock) -> int:
    import time
    from pipeline.profile_cmd import TEAMS_DIR, cmd_profil
    from research.llm_client import EXHAUSTED, task_models
    from research.t10_news import team_names
    from research.t9_pilot import run_team

    st = _progress(TEAMS_DIR)
    order, playing = all_teams_order(settings, today)

    def save():
        (TEAMS_DIR / PROGRESS).write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
        lock.touch()                                  # heartbeat
    t0, done, touched, stop = time.monotonic(), {"T10": [], "T9": []}, set(), None

    def due(task: str, key: str, is_playing: bool) -> bool:
        last = st[task].get(key)
        if not last:
            return True
        age = (today - date.fromisoformat(last)).days
        return age >= (T9_REFRESH_DAYS if task == "T9" else (1 if is_playing else DAILY_DAYS))

    for task in ("T10", "T9"):
        used = models or task_models(task)
        for lg, code in order:
            key = f"{lg}/{code}"
            if code not in team_names(lg) or not due(task, key, (lg, code) in playing):
                continue
            if set(used[:2]) & EXHAUSTED:
                stop = {"zadaca": task, "tim": key, "modeli": sorted(set(used[:2]) & EXHAUSTED)}
                break
            nm = team_names(lg)[code]
            try:
                if task == "T10":
                    _news(settings, lg, code, today, models, days=DAILY_DAYS)
                else:
                    run_team(settings, lg, code, nm["ime"], nm["prebaruvanje"], nm["zborovi"], used, SEASON_START, today)
            except Exception as exc:                  # one team's failure must not stop the others
                print(f"{task} {key}: ❌ {type(exc).__name__}: {str(exc)[:150]}")
                continue
            if set(used[:2]) & EXHAUSTED:              # quota ran out DURING this team → incomplete, redo next time
                stop = {"zadaca": task, "tim": key, "modeli": sorted(set(used[:2]) & EXHAUSTED)}
                break
            st[task][key] = today.isoformat()
            done[task].append(key)
            touched.add(lg)
            save()
            print(f"{task} {key}: ✅ ({time.monotonic() - t0:.0f} s)", flush=True)
        if stop:
            break
    st["zastanato"] = ({**stop, "koga": f"{today}", "zabeleska": "квотата потрошена — продолжува во 14:00"}
                       if stop else {"koga": f"{today}", "zabeleska": "сè што доспева е готово"})
    save()
    for lg in touched:
        cmd_profil(settings, lg, today)
    print(f"\nГотово: T10 {len(done['T10'])} · T9 {len(done['T9'])} · {time.monotonic() - t0:.0f} s"
          + (f" · ЗАСТАНАТО кај {stop['zadaca']} {stop['tim']} (квота: {', '.join(stop['modeli'])})" if stop else ""))
    print(progress_table(settings, today))
    return 0


def progress_table(settings: Settings, today: date) -> str:
    """Tome 1.11: team | news ✅/❌ | T9 ✅/❌ | last update — written to reports/llm_napredok.md."""
    from pipeline.profile_cmd import TEAMS_DIR
    st = _progress(TEAMS_DIR)
    order, playing = all_teams_order(settings, today)
    rows, miss10, miss9 = [], 0, 0
    for lg, code in order:
        key = f"{lg}/{code}"
        t10, t9 = st["T10"].get(key), st["T9"].get(key)
        ok10 = bool(t10) and (today - date.fromisoformat(t10)).days < (1 if (lg, code) in playing else DAILY_DAYS)
        ok9 = bool(t9) and (today - date.fromisoformat(t9)).days < T9_REFRESH_DAYS
        miss10 += not ok10
        miss9 += not ok9
        rows.append(f"| {lg} | {code}{' 🏀' if (lg, code) in playing else ''} | {'✅' if ok10 else '❌'} | {'✅' if ok9 else '❌'} | "
                    f"{max(filter(None, [t10, t9]), default='—')} |")
    z = st.get("zastanato") or {}
    out = [f"# Напредок LLM (T10 вести 7 дена + T9 тренер/ГМ) · {today}", "",
           f"Фалат: вести {miss10} / {len(order)} · T9 {miss9} / {len(order)} · 🏀 = игра денес/утре (вестите се освежуваат дневно)",
           f"Последно застанување: {z.get('koga', '—')} · {z.get('zadaca', '')} {z.get('tim', '')} · {z.get('zabeleska', '')}", "",
           "| Лига | Тим | вести | T9 | последно ажурирање |", "|---|---|---|---|---|", *rows]
    path = settings.paths.reports / "llm_napredok.md"
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return "\n".join(out[:4]) + f"\n→ {path}"


def run_pilot(settings: Settings, league: str, tim: str, today: date, models: list[str] | None, only_t9: bool = False) -> int:
    """T10 news (30 days) + T9 coach/GM proposal + the Python dossier for one team, with cost numbers."""
    import time
    from pipeline.profile_cmd import TEAMS_DIR, cmd_profil
    from research.llm_client import task_models
    from research.t10_news import team_names
    from research.t9_pilot import run_team

    nm = team_names(league)[tim]
    t0 = time.monotonic()
    empty = {"zapisi": [], "eden_model": [], "otfrleno": [], "izvori": [],
             "stats": {"izvori": 0, "povici": 0, "od_kesh": 0, "tokeni": 0, "sekundi": 0.0, "bez_odgovor": [], "soglasuvanje": {}}}
    if only_t9:
        news, n_inj, coach, models10 = empty, 0, None, []
    else:
        news, n_inj, coach, models10 = _news(settings, league, tim, today, models)
    models9 = models or task_models("T9")
    path, t9 = run_team(settings, league, tim, nm["ime"], nm["prebaruvanje"], nm["zborovi"], models9, SEASON_START, today,
                        extra=news["izvori"])
    cmd_profil(settings, league, today)                      # dossier re-rendered with the new vesti.md
    wall = time.monotonic() - t0
    s = news["stats"]
    agr = lambda d: " · ".join(f"{m}: {v['potvrdeni']}/{v.get('stavki', v.get('fakti', 0))}" for m, v in (d or {}).items()) or "—"
    rep = [f"# Пилот T9/T10 — {tim} ({league}) · {today}", "",
           f"Модели: T10 {' + '.join(models10[:2]) or '—'} · T9 {' + '.join(models9[:2])} · официјален тренер: {coach or '—'}", "",
           "## Трошок", "", "| Дел | извори | повици | од кеш | токени | сек (модели) |", "|---|---|---|---|---|---|",
           f"| T10 вести (30 дена) | {s['izvori']} | {s['povici']} | {s['od_kesh']} | {s['tokeni']} | {s['sekundi']:.0f} |",
           f"| T9 тренер/ГМ | {t9['napisi']} | {t9['povici']} | {t9['od_kesh']} | {t9['tokeni']} | {t9['sekundi']:.0f} |",
           f"| **вкупно** | | **{s['povici'] + t9['povici']}** | | **{s['tokeni'] + t9['tokeni']}** | вкупно време {wall:.0f} s |",
           "", f"Согласување (потврдени од другиот модел / ставки): T10 {agr(s.get('soglasuvanje'))} · T9 {agr(t9.get('soglasuvanje'))}",
           "", f"Повреди/состави од последните 24 ч → data/vesti (T1): {n_inj}", "",
           f"## Вести во vesti.md (двата модели): {len(news['zapisi'])}", "",
           *[f"- {r['objaveno'][:10]} · {r['tip']} · {r['igrac'] or '—'} · {r['sto']}" for r in news["zapisi"][:15]], "",
           f"## Само еден модел (не влегува): {len(news['eden_model'])}", "",
           *[f"- {r['objaveno'][:10]} · {r['tip']} · {r['igrac'] or '—'} · {r['sto'][:90]} ({r['modeli']})" for r in news["eden_model"][:25]], "",
           f"## Отфрлено од Python (T10): {len(news['otfrleno'])}", "", *[f"- {r}" for r in news["otfrleno"][:40]], "",
           f"## Се коси со нашите бројки (T9): {len(t9['kosenje'])}", "", *[f"- {r}" for r in t9["kosenje"]], "",
           f"## Без одговор: {len(s['bez_odgovor'])}", "", *[f"- {r}" for r in s["bez_odgovor"][:20]], "",
           "## Извори", "", *[f"- {x['objaveno'][:10]} [{x['naslov'][:90]}]({x['izvor_url']})" for x in news["izvori"]], "",
           f"T9 предлог: `{path.relative_to(TEAMS_DIR.parent.parent).as_posix()}` (+ .md) → `python main.py profil odobri`"]
    out = settings.paths.reports / f"pilot_{tim}_{today}.md"
    out.write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(rep[:14]))
    print(f"\nИзвештај: {out}")
    return 0
