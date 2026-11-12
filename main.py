"""Betting Prediction CLI (BUILD_PROMPT §7)."""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime
from zoneinfo import ZoneInfo

from core.config import load_settings
from core.logging_setup import setup_logging
from scrapers.http_client import SourceError

log = logging.getLogger("main")

LEAGUE_GROUPS = {"all": ["euroleague", "eurocup", "nba"]}
NOT_YET = {"proveri": 7}


def _parse_date(text: str | None, tz: str) -> date:
    return date.fromisoformat(text) if text else datetime.now(ZoneInfo(tz)).date()


def cmd_fetch(args, settings) -> int:
    from pipeline.fetch import fetch_league

    day = _parse_date(args.date, settings.timezone)
    leagues = LEAGUE_GROUPS.get(args.league, [args.league])
    failed = False
    for key in leagues:
        try:
            s = fetch_league(settings, key, day, refresh=args.refresh, refresh_box=args.refresh_box,
                             season=getattr(args, 'season', None))
        except NotImplementedError as exc:
            log.warning("%s", exc)
            continue
        except SourceError as exc:
            log.error("%s: изворот не одговара — %s", key, exc)
            failed = True
            continue
        name = settings.league(key).name_mk
        print(f"\n=== {name} · {day} ===")
        print(f"Завршени натпревари: {s.games_finished} · box score: {s.boxscores}"
              + (f" · НЕДОСТАСУВААТ: {s.boxscores_missing}" if s.boxscores_missing else ""))
        print(f"Логови: {s.team_log_rows} тимски редови, {s.player_log_rows} играчки редови (сортирани по датум)")
        print(f"Натпревари на {day}: " + ("нема" if not s.games_on_day else ""))
        for g in s.games_on_day:
            print(f"  {g}")
        print(f"Зачувано во: {s.out_dir}")
    return 1 if failed else 0


def cmd_mozzart(settings) -> int:
    """Line snapshots of BOTH bookmakers (cache only; no model reads them before Phase 7)."""
    from scrapers.evrotip import snapshot as evrotip_snapshot
    from scrapers.mozzart import snapshot as mozzart_snapshot
    code = 0
    for name, fn in (("Mozzart", mozzart_snapshot), ("evrotip", evrotip_snapshot)):
        try:
            path, s = fn(settings)
        except (SourceError, RuntimeError) as exc:
            log.error("%s не одговара / блокира: %s", name, exc)
            code = 1
            continue
        groups = s.get("by_competition") or s.get("by_league") or {}
        comps = ", ".join(f"{k} {v}" for k, v in sorted(groups.items()))
        n_odds = s.get("kodds", s.get("odds"))
        print(f"{name} линии зачувани: {s['matches']} натпревари/играчи, {n_odds} квоти ({comps}) -> {path}")
    print("(само кеш — моделот не ги користи до Фаза 7)")
    return code


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="main.py", description="Betting Prediction — NBA · Евролига · Еврокуп")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    leagues = ["euroleague", "eurocup", "nba", "all"]
    p = sub.add_parser("fetch", help="повлечи и кеширај податоци")
    p.add_argument("--league", choices=leagues, default="all")
    p.add_argument("--date", help="YYYY-MM-DD (стандардно: денес)")
    p.add_argument("--refresh", action="store_true", help="повлечи распоред/статистики одново (старите се чуваат)")
    p.add_argument("--refresh-box", action="store_true", help="повлечи ги и кешираните box score одново")
    p.add_argument("--season", help="минатата сезона за backtest: E2025 / U2025 / 2025-26 (стандардно: тековна)")

    p = sub.add_parser("povredi", help="отсуства (НОВО/СТАРО) и ефект врз соиграчите (Фаза 6)")
    p.add_argument("--league", choices=leagues, default="all")
    p.add_argument("--date", help="YYYY-MM-DD (стандардно: денес)")
    p = sub.add_parser("profil", help="профили на тимови: БРОЈКИ (неделно) / `profil odobri` за предлози од агентите")
    p.add_argument("akcija", nargs="?", help="`odobri` (предлози од агентите) или код на тим (преглед), пр. RED")
    p.add_argument("--league", choices=leagues, default="all")
    p.add_argument("--prifati")
    p.add_argument("--odbij")
    p = sub.add_parser("llm", help="бесплатни LLM истражувачи: тест-сет / тест на моделите / резиме (LLM_ISTRAZUVACI §6)")
    p.add_argument("akcija", choices=["testset", "test", "rezime", "dnevno", "site", "napredok", "izvadok", "vesti", "t9", "pilot"])
    p.add_argument("--tim", help="за vesti / t9 / pilot: код на тимот (пр. RED, DAL)")
    p.add_argument("--league", choices=leagues, default="all")
    p.add_argument("--modeli", help="листа одделена со запирка (стандардно: сите достапни во FreeLLMAPI)")
    p.add_argument("--zadaci", default="T1,T9,T10")
    p = sub.add_parser("vesti", help="последни вести за тим од vesti.md (T10, проверено од 2 модела)")
    p.add_argument("tim")
    p.add_argument("--league", choices=leagues, default="all")
    p.add_argument("-n", type=int, default=15)
    p = sub.add_parser("mozzart", help="(исто како linii)")
    p = sub.add_parser("linii", help="зачувај снимка од линиите (Mozzart + evrotip), само кеш")
    p.add_argument("--izvestaj", action="store_true", help="табела на денот: маржа, движење, Mozzart vs evrotip")
    p.add_argument("--date")

    p = sub.add_parser("picks", help="анализа + избори (Фаза 7) — по default НА ХАРТИЈА, ништо не се уплаќа")
    p.add_argument("--league", choices=leagues, default="all")
    p.add_argument("--date")
    p.add_argument("--hartija", action="store_true", default=True, help="(default) избори само на хартија")
    p = sub.add_parser("pregled", help="ПРЕГЛЕД на сите опции во 3 нивоа по пазарен процент (не е препорака, ништо не се запишува)")
    p.add_argument("--league", choices=leagues, default="nba")
    p.add_argument("--date")
    p.add_argument("--natprevar", help="само еден натпревар, пр. PHI-CHI")
    p = sub.add_parser("grade", help="внеси резултати од box score за избори што чекаат")
    p.add_argument("--date", help="само избори од овој датум (стандардно: сите што чекаат)")
    p.add_argument("--dry-run", action="store_true", help="покажи што би се запишало, без запишување")
    p = sub.add_parser("stats", help="статистика на изборите")
    p.add_argument("--league", choices=["euroleague", "eurocup", "nba"])
    p.add_argument("--nedela", action="store_true", help="само последните 7 дена")
    p = sub.add_parser("migrate", help="еднократно: стара CSV архива → archive/izbori.db")
    p.add_argument("--force", action="store_true", help="направи ја базата одново (старата се чува како копија)")
    p = sub.add_parser("izbori", help="листа на избори со ID (за uplata)")
    p.add_argument("--datum")
    p.add_argument("--league", choices=["euroleague", "eurocup", "nba"])
    p.add_argument("--status")
    p = sub.add_parser("uplata", help="запиши дали избор е навистина уплатен и колку (за „ROI уплатено“)")
    p.add_argument("id", type=int)
    p.add_argument("--iznos", type=float, help="износ во денари")
    p.add_argument("--ne", action="store_true", help="не е уплатено")
    p = sub.add_parser("backtest", help="walk-forward backtest на тимскиот модел (Фаза 4)")
    p.add_argument("--league", choices=["euroleague", "eurocup", "nba", "all"], default="all")
    p.add_argument("--pocetok", action="store_true", help="тест на почеток на сезона (пренос меѓу сезони)")
    p.add_argument("--odmor", action="store_true", help="тест на одмор / back-to-back")
    p.add_argument("--igraci", action="store_true", help="backtest на играчкиот модел (Фаза 5)")
    p.add_argument("--kandidati", action="store_true", help="тест на параметри-кандидати (прв стрелец, враќање)")
    p.add_argument("--pristrasnost", action="store_true", help="почеток на сезона: лига=15 / неделна корекција")
    p.add_argument("--niski", action="store_true", help="ниски линии (тројки 0.5/1.5, скокови 2.5/3.5, асистенции 1.5/2.5)")
    p.add_argument("--devig", action="store_true", help="de-vig методи (пропорционален / power / Shin) на снимките")
    p = sub.add_parser("dnevno", help="дневна автоматика (Task Scheduler): 14 / 20 / clv / zakazi — ништо не се уплаќа")
    p.add_argument("--rezim", choices=["14", "20", "clv", "zakazi"], required=True)
    p.add_argument("--bez-llm", action="store_true", help="14: без `llm site` на крај")
    p = sub.add_parser("nitro", help="НИТРО: факти што Claude ги нашол → Python ги симнува и проверува → vesti → povredi → picks")
    p.add_argument("vlez", help="data/raw/vesti/nitro/<датум>/<ознака>_vlez.json")
    p.add_argument("--date")
    p.add_argument("--bez-picks", action="store_true")
    p = sub.add_parser("proveri", help="опционално: нови вести/состави за веќе избраните (Фаза 7)")
    p.add_argument("--date")

    for stream in (sys.stdout, sys.stderr):      # Task Scheduler / cp1252 consoles: Cyrillic output must not crash
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    settings = load_settings()

    if args.command == "fetch":
        code = cmd_fetch(args, settings)
        if not getattr(args, "season", None):
            cmd_mozzart(settings)        # every fetch also stores a line snapshot (history for CLV/backtest)
        return code
    if args.command == "dnevno":
        from pipeline.dnevno_cmd import cmd_dnevno
        return cmd_dnevno(settings, args.rezim, llm=not args.bez_llm)
    if args.command == "nitro":
        from research.nitro import cmd_nitro
        return cmd_nitro(settings, args.vlez, _parse_date(args.date, settings.timezone), picks=not args.bez_picks)
    if args.command == "pregled":
        from pipeline.pregled_cmd import cmd_pregled
        return cmd_pregled(settings, args.league, _parse_date(args.date, settings.timezone), args.natprevar)
    if args.command == "picks":
        from pipeline.picks_cmd import cmd_picks
        return cmd_picks(settings, args.league, _parse_date(args.date, settings.timezone))
    if args.command == "povredi":
        from pipeline.injuries_cmd import cmd_povredi
        return cmd_povredi(settings, args.league, _parse_date(args.date, settings.timezone))
    if args.command == "profil":
        from pipeline.profile_cmd import cmd_profil, cmd_profil_odobri
        if args.akcija == "odobri":
            return cmd_profil_odobri(args.prifati, args.odbij)
        if args.akcija:
            from pipeline.profile_cmd import cmd_profil_show
            return cmd_profil_show(args.akcija, args.league)
        return cmd_profil(settings, args.league, _parse_date(None, settings.timezone))
    if args.command == "vesti":
        from pipeline.profile_cmd import cmd_vesti
        return cmd_vesti(args.tim, args.league, args.n)
    if args.command == "llm":
        from pipeline.llm_cmd import cmd_llm
        return cmd_llm(settings, args.akcija, args.modeli, args.zadaci, _parse_date(None, settings.timezone), args.tim,
                       args.league)
    if args.command == "linii" and args.izvestaj:
        from pipeline.lines_cmd import lines_report
        print(lines_report(settings, _parse_date(args.date, settings.timezone)))
        return 0
    if args.command in ("mozzart", "linii"):
        return cmd_mozzart(settings)
    if args.command == "migrate":
        from pipeline.archive_cmds import cmd_migrate
        return cmd_migrate(args.force)
    if args.command == "stats":
        from pipeline.archive_cmds import cmd_stats
        return cmd_stats(args.league, args.nedela)
    if args.command == "izbori":
        from pipeline.archive_cmds import cmd_izbori
        return cmd_izbori(args.datum, args.league, args.status)
    if args.command == "uplata":
        from pipeline.archive_cmds import cmd_uplata
        return cmd_uplata(args.id, args.iznos, args.ne)
    if args.command == "backtest":
        from pipeline.backtest_cmd import cmd_backtest
        kind = ("pocetok" if args.pocetok else "odmor" if args.odmor else "igraci" if args.igraci
                else "kandidati" if args.kandidati else "pristrasnost" if args.pristrasnost else "niski" if args.niski
                else "devig" if args.devig else "tim")
        return cmd_backtest(settings, args.league, kind)
    if args.command == "grade":
        from pipeline.grade_cmd import cmd_grade
        return cmd_grade(settings, args.date, args.dry_run)
    print(f"Командата '{args.command}' уште не е изградена (Фаза {NOT_YET[args.command]}). Види docs/PLAN.md.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
