"""`python main.py dnevno --rezim 14|20|clv|zakazi` — daily automation (Tome 31.10; times 14/20 since 1.11), run by
Windows Task Scheduler.

14:00  grade → stats → povredi → fetch → picks (all leagues) → reports/dnevno/YYYY-MM-DD.md → CLV tasks → llm site
       (T10 + T9, all 82) until 23:00
20:00  NBA focus: povredi → fetch → picks → section „Ажурирање 20:00“ in the same report
clv    fetch (= new data + line snapshot of both books) 30 min before each league's first game today
zakazi registers the two daily tasks (14:00, 20:00): wake the computer, run if missed, one run at a time

Every step is its own process; a failed step goes to reports/dnevno/GRESKI.log and the next one runs.
Nothing is staked: picks stays in paper mode. The report's first line is „⚠️ ИТНО“ when something needs Tome.
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from core.config import Settings

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "dnevno"
ERRORS = OUT / "GRESKI.log"
LEAGUES = ["euroleague", "eurocup", "nba"]
LEAGUE_MK = {"euroleague": "ЕЛ", "eurocup": "ЕК", "nba": "NBA"}
MAIN, UPDATE = "14", "20"             # Tome 1.11: main run 14:00, update 20:00 (was 12/16)
STEPS = {MAIN: [["grade"], ["stats"], ["povredi"], ["fetch"], ["picks"], ["pregled", "--league", "all"]],
         UPDATE: [["povredi", "--league", "nba"], ["fetch", "--league", "nba"], ["picks", "--league", "nba"],
                  ["pregled", "--league", "nba"]]}       # Tome 2.11: overview after picks
LLM_UNTIL = (23, 0)                    # `llm site` stops at 23:00 (progress saved after each team; resumes tomorrow)
TIMEOUT_S: dict[str, int] = {}
CLV_BEFORE = timedelta(minutes=30)
TASK_PREFIX = "Betting_"
URGENT = "⚠️ ИТНО"
MAX_CANDIDATES = 10
MAX_ABSENCES = 12
PAPER, DEPENDS = "САМО_ХАРТИЈА", "ЗАВИСИ_ОД_СОСТАВ"
PLAYS = {"AVAILABLE", "PROBABLE"}             # on the report but plays → not an absence
# 1.11: the PC slept until 18:27 → 12:00, 16:00 and CLV started together and fought over the same cache files
# (WinError 32). One run at a time: a later run waits for the earlier one.
RUN_LOCK = OUT / ".dnevno.lock"
RUN_LOCK_STALE_S = 3 * 3600
RUN_LOCK_WAIT_S = 2 * 3600


def _now(settings: Settings) -> datetime:
    return datetime.now(ZoneInfo(settings.timezone)).replace(tzinfo=None)


def run_step(args: list[str], label: str, now: datetime) -> bool:
    """One `main.py` command in its own process; output → logs/, failure → GRESKI.log. True = ok."""
    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f"{now:%Y-%m-%d_%H%M}_{label.replace(':', '')}_{'_'.join(args[:2])}.log"   # ':' = NTFS stream
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    try:
        r = subprocess.run([sys.executable, "main.py", *args], cwd=ROOT, env=env, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=TIMEOUT_S.get(args[0], 3600),
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))   # no console to close by hand
        log.write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
        if r.returncode == 0:
            return True
        tail = (r.stderr.strip().splitlines() or r.stdout.strip().splitlines() or ["—"])[-1]
        why = f"exit {r.returncode}: {tail[:200]}"
    except Exception as exc:                              # timeout, missing python … → logged, next step runs
        why = f"{type(exc).__name__}: {exc}"[:250]
    with ERRORS.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M} | {label} | {' '.join(args)} | {why} | лог: {log.name}\n")
    return False


# ---------------- report pieces (numbers only from the archive / schedule / injury code) ----------------

def _archive() -> sqlite3.Connection:
    return sqlite3.connect(ROOT / "archive" / "izbori.db")


def yesterday_block(day: date) -> list[str]:
    y = (day - timedelta(days=1)).isoformat()
    a = pd.read_sql("SELECT liga, status, ishod, p_kalibriran FROM izbori WHERE datum=?", _archive(), params=[y])
    if a.empty:
        return [f"- нема избори за {y}"]
    out = []
    for lg, g in a.groupby("liga"):
        ch = g[g["status"].isin(["DA", "NE"])]
        allr = g[g["ishod"].isin(["DA", "NE"])]
        wait = int((g["status"] == "CEKA").sum())
        txt = (f"избрани {int((ch.status == 'DA').sum())}/{len(ch)} = {(ch.status == 'DA').mean() * 100:.0f}% "
               f"(очек. {ch.p_kalibriran.mean() * 100:.0f}%)" if len(ch) else "избрани: нема оценети")
        if len(allr):
            txt += f" · сите кандидати (и одбиени) {(allr.ishod == 'DA').mean() * 100:.0f}% (n={len(allr)})"
        if wait:
            txt += f" · **{wait} уште не оценети**"
        out.append(f"- {LEAGUE_MK.get(lg, lg)}: {txt}")
    return out


def games_today(settings: Settings, day: date) -> pd.DataFrame:
    rows = []
    for lg in LEAGUES:
        path = settings.paths.processed / lg / settings.league(lg).season / "schedule.csv"
        if not path.exists():
            continue
        s = pd.read_csv(path, dtype={"game_code": str})
        s = s[s["game_date"] == day.isoformat()]
        if len(s):
            start = pd.to_datetime(s["date_utc"], utc=True).dt.tz_convert(settings.timezone).dt.tz_localize(None)
            rows.append(s.assign(liga=lg, start=start)[["liga", "start", "home", "away"]])
    return pd.concat(rows).sort_values("start") if rows else pd.DataFrame(columns=["liga", "start", "home", "away"])


def games_block(games: pd.DataFrame) -> list[str]:
    if games.empty:
        return ["- нема натпревари"]
    return [f"- {LEAGUE_MK[lg]} ({len(g)}): " + ", ".join(f"{r.start:%H:%M} {r.home}–{r.away}" for r in g.itertuples())
            for lg, g in games.groupby("liga", sort=False)]


def candidates_block(day: date) -> tuple[list[str], int]:
    """Today's candidates still open (CEKA): model % vs market %. Returns (lines, picks waiting for Tome)."""
    a = pd.read_sql("SELECT liga, natprevar, tim, igrac, pazar, strana, linija, kvota, p_kalibriran, p_pazar, ev, "
                    "pravila_primeneti, status FROM izbori WHERE datum=?", _archive(), params=[day.isoformat()])
    if a.empty:
        return ["- нема кандидати (нема линии или натпревари)"], 0
    rules = a["pravila_primeneti"].fillna("")
    a["odluka"] = (a["status"] == "CEKA") & ~rules.str.contains(PAPER) & ~rules.str.contains(DEPENDS)
    open_ = a[a["status"] == "CEKA"].sort_values(["odluka", "ev"], ascending=False)
    lines = [f"Вкупно {len(a)} запишани · отворени {len(open_)} · одбиени {int((a.status == 'ODBIEN').sum())} · "
             f"**за одлука {int(a.odluka.sum())}**", "",
             "| | Натпревар | Избор | Модел % | Пазар % | EV | Квота |", "|---|---|---|---|---|---|---|"]
    for r in open_.head(MAX_CANDIDATES).itertuples():
        who = next((x for x in (r.igrac, r.tim) if isinstance(x, str)), "")
        tag = "🟢 одлука" if r.odluka else ("хартија" if PAPER in (r.pravila_primeneti or "") else "состав?")
        pp = f"{r.p_pazar * 100:.0f}" if pd.notna(r.p_pazar) else "нема податок"
        lines.append(f"| {tag} | {r.natprevar} | {who} {r.pazar} {r.strana} {r.linija:g} | {r.p_kalibriran * 100:.0f} | "
                     f"{pp} | {r.ev * 100:+.1f}% | {r.kvota:.2f} |")
    if open_.empty:
        lines = lines[:1] + ["", "- ниеден отворен кандидат"]
    return lines, int(a["odluka"].sum())


def absences_block(settings: Settings, day: date, games: pd.DataFrame) -> list[str]:
    from pipeline.injuries_cmd import league_absences
    out = []
    for lg in games["liga"].unique():
        try:
            df, notes = league_absences(settings, lg, day)
        except Exception as exc:
            out.append(f"- {LEAGUE_MK[lg]}: отсуствата не се прочитани ({type(exc).__name__}) → види GRESKI.log")
            continue
        pending = next((n.split(":", 1)[1] for n in notes if n.startswith("__pending__:")), "")
        if pending:
            out.append(f"- {LEAGUE_MK[lg]}: уште НЕ предале извештај: {pending.replace(',', ', ')}")
        if df.empty:
            continue
        plays = df["status"].fillna("").str.upper().isin(PLAYS)
        new = df[(df["novo"] == True) & (df["minuti_sezona"].fillna(0) >= 15) & ~plays]  # noqa: E712
        new = new.sort_values("minuti_sezona", ascending=False)
        for r in new.head(MAX_ABSENCES).itertuples():
            out.append(f"- {LEAGUE_MK[lg]}: {r.label} — {r.status}, НОВО, {r.minuti_sezona:.0f} мин")
        if len(new) > MAX_ABSENCES:
            out.append(f"- … уште {len(new) - MAX_ABSENCES} (reports/povredi_{day}_{lg}.md)")
    return out or ["- нема нови отсуства на играчи од ротацијата (≥ 15 мин)"]


# ---------------- report file ----------------

def _flags_path(day: date) -> Path:
    return OUT / f".flags_{day}.json"


def write_report(day: date, section: str, flags: list[str], first: bool) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path, fp = OUT / f"{day}.md", _flags_path(day)
    old_flags = json.loads(fp.read_text(encoding="utf-8")) if fp.exists() and not first else []
    all_flags = list(dict.fromkeys(old_flags + flags))
    fp.write_text(json.dumps(all_flags, ensure_ascii=False), encoding="utf-8")
    body = ""
    if path.exists() and not first:
        body = path.read_text(encoding="utf-8")
        if body.startswith(URGENT):
            body = body.split("\n", 1)[1]
    if not body:
        body = f"# Дневен извештај · {day}\n\n*Автоматски (Task Scheduler) · ништо не е уплатено · бројки само од кодот.*\n"
    head = f"{URGENT}: " + " · ".join(all_flags) + "\n" if all_flags else ""
    path.write_text(head + body.rstrip() + "\n\n" + section.strip() + "\n", encoding="utf-8")
    return path


def notify(title: str, text: str) -> None:
    """Windows toast (best effort; the report file is the real output)."""
    ps = ("[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null;"
          "$x = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02);"
          "$t = $x.GetElementsByTagName('text'); $t.Item(0).InnerText = $env:BT_TITLE; $t.Item(1).InnerText = $env:BT_TEXT;"
          "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('Betting Prediction').Show("
          "[Windows.UI.Notifications.ToastNotification]::new($x))")
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], env=dict(os.environ, BT_TITLE=title, BT_TEXT=text[:200]),
                       capture_output=True, timeout=30)
    except Exception:
        pass


# ---------------- Task Scheduler ----------------

def _register(name: str, trigger: str, args: str) -> bool:
    """PowerShell Register-ScheduledTask (locale-independent dates; schtasks /SD failed on 31.10)."""
    # pythonw = no console window: 1.11 and 2.11 the visible window was closed mid-run (0xC000013A = console closed)
    py = ROOT / "venv" / "Scripts" / "pythonw.exe"
    ps = (f"$a = New-ScheduledTaskAction -Execute '{py}' -Argument '{args}' -WorkingDirectory '{ROOT}';"
          f"$t = {trigger};"
          "$s = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries "
          "-WakeToRun -MultipleInstances Queue -ExecutionTimeLimit (New-TimeSpan -Hours 10);"
          f"Register-ScheduledTask -TaskName '{name}' -Action $a -Trigger $t -Settings $s -Force | Out-Null")
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True)
    return r.returncode == 0


def schedule_clv(games: pd.DataFrame, now: datetime) -> list[str]:
    """One-off task 30 min before the first game of each league today (closing prices for CLV)."""
    out = []
    for lg, g in games.groupby("liga", sort=False):
        at = g["start"].min() - CLV_BEFORE
        if at <= now + timedelta(minutes=2):
            out.append(f"- {LEAGUE_MK[lg]}: прв натпревар {g['start'].min():%H:%M} — премногу блиску, без CLV задача")
            continue
        ok = _register(f"{TASK_PREFIX}CLV_{lg}", f"New-ScheduledTaskTrigger -Once -At '{at:%Y-%m-%dT%H:%M}'", "main.py dnevno --rezim clv")
        out.append(f"- {LEAGUE_MK[lg]}: fetch + линии во **{at:%H:%M}** (прв натпревар {g['start'].min():%H:%M})"
                   + ("" if ok else " — ❌ задачата не е регистрирана"))
    return out


def cmd_zakazi() -> int:
    for old in ("Dnevno_1200", "Dnevno_1600"):           # 1.11: replaced by 14:00 / 20:00
        subprocess.run(["powershell", "-NoProfile", "-Command",
                        f"Unregister-ScheduledTask -TaskName '{TASK_PREFIX}{old}' -Confirm:$false -ErrorAction SilentlyContinue"],
                       capture_output=True)
    ok = all([_register(f"{TASK_PREFIX}Dnevno_1400", "New-ScheduledTaskTrigger -Daily -At '14:00'", f"main.py dnevno --rezim {MAIN}"),
              _register(f"{TASK_PREFIX}Dnevno_2000", "New-ScheduledTaskTrigger -Daily -At '20:00'", f"main.py dnevno --rezim {UPDATE}")])
    print("Закажано: 14:00 и 20:00 дневно (буди го компјутерот, пропуштено → штом се вклучи, една по друга)"
          if ok else "❌ регистрацијата не успеа")
    return 0 if ok else 1


# ---------------- entry ----------------

def acquire_run_lock(wait_s: float = RUN_LOCK_WAIT_S, poll_s: float = 20.0) -> bool:
    import time
    OUT.mkdir(parents=True, exist_ok=True)
    end = time.monotonic() + wait_s
    while True:
        try:
            os.close(os.open(RUN_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            return True
        except FileExistsError:
            try:
                if time.time() - RUN_LOCK.stat().st_mtime > RUN_LOCK_STALE_S:   # a killed run left it behind
                    RUN_LOCK.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() >= end:
                return False
            time.sleep(poll_s)


def cmd_dnevno(settings: Settings, rezim: str, llm: bool = True) -> int:
    if rezim == "zakazi":
        return cmd_zakazi()
    if not acquire_run_lock():
        with ERRORS.open("a", encoding="utf-8") as f:
            f.write(f"{datetime.now():%Y-%m-%d %H:%M} | {rezim} | чекање | друга dnevno задача уште работи (>2 ч) — "
                    f"овој циклус е прескокнат\n")
        return 1
    try:
        return _cmd_dnevno(settings, rezim, llm)
    finally:
        RUN_LOCK.unlink(missing_ok=True)


def _cmd_dnevno(settings: Settings, rezim: str, llm: bool = True) -> int:
    now = _now(settings)
    day = now.date()
    if rezim == "clv":
        return 0 if run_step(["fetch"], "clv", now) else 1
    label = f"{rezim}:00"
    main = rezim == MAIN
    failed = [" ".join(a) for a in STEPS[rezim] if not run_step(a, label, now)]
    games = games_today(settings, day)
    cands, decide = candidates_block(day)
    flags = []
    if failed:
        flags.append(f"{label} падна: {', '.join(failed)} (GRESKI.log)")
    if decide:
        flags.append(f"{decide} избор(и) чекаат твоја одлука пред уплата")
    sec = [f"## {'Главна 14:00' if main else 'Ажурирање 20:00 (NBA)'} ({now:%H:%M})", ""]
    if main:
        sec += ["### Вчера", *yesterday_block(day), "", "### Денес — натпревари", *games_block(games), ""]
    sec += ["### Кандидати (модел % vs пазар %)", *cands, "", "### Отсуства (нови, ротација)",
            *absences_block(settings, day, games), ""]
    if main and len(games):
        sec += ["### CLV снимки (автоматски)", *schedule_clv(games, now), ""]
    sec.append("Чекори: " + ("сите ✅" if not failed else "❌ " + ", ".join(failed) + " → reports/dnevno/GRESKI.log"))
    path = write_report(day, "\n".join(sec), flags, first=main)
    notify(f"Betting · {label} · {day}", (f"{URGENT}: " + " · ".join(flags)) if flags else f"Извештај: {path.name}")
    print(f"Извештај: {path}")
    RUN_LOCK.unlink(missing_ok=True)                     # `llm site` runs for hours and has its own lock
    until = now.replace(hour=LLM_UNTIL[0], minute=LLM_UNTIL[1], second=0, microsecond=0)
    if main and llm and until > now:
        TIMEOUT_S["llm"] = int((until - now).total_seconds())
        ok = run_step(["llm", "site"], "llm", now)        # Tome 1.11: all 82 teams, resumes where the quota stopped
        prog = ROOT / "data" / "timovi" / "_napredok_llm.json"
        stop = (json.loads(prog.read_text(encoding="utf-8")).get("zastanato") or {}) if prog.exists() else {}
        status = ((f"⚠️ квотата потрошена кај {stop['zadaca']} {stop['tim']} — продолжува утре во 14:00"
                   if stop.get("tim") else "✅ сè што доспева е готово") if ok else "❌ види GRESKI.log")
        with path.open("a", encoding="utf-8") as f:
            f.write(f"\nLLM вести + T9 (`llm site`): {status} · табела: reports/llm_napredok.md\n")
    return 1 if failed else 0
