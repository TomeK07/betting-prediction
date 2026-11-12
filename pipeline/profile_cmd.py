"""`python main.py profil` — weekly team profiles (NUMBERS) + `profil odobri` for agent proposals (DESCRIPTION)."""
from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from core.config import ROOT, Settings
from model.team_model import params_from_config
from model.team_profile import build_league_profiles

TEAMS_DIR = ROOT / "data" / "timovi"


def cmd_profil(settings: Settings, league: str, today: date) -> int:
    leagues = ["euroleague", "eurocup", "nba"] if league == "all" else [league]
    for lg in leagues:
        cfg = settings.league(lg)
        folder = settings.paths.processed / lg / cfg.season
        if not (folder / "team_logs.csv").exists():
            print(f"{cfg.name_mk}: нема податоци — прво `python main.py fetch --league {lg}`")
            continue
        read = lambda n: pd.read_csv(folder / n, dtype={"game_code": str, "player_code": str})
        changes = build_league_profiles(read("team_logs.csv"), read("player_logs.csv"), read("roster.csv"),
                                        lg, cfg.name_mk, TEAMS_DIR, today,
                                        params_from_config(cfg.team_model) if cfg.team_model else None)
        n_changes = sum(len(v) for v in changes.values())
        print(f"{cfg.name_mk}: {len(changes)} профили во data/timovi/{lg}/ · промени од box score: {n_changes}")
        for team, ch in changes.items():
            for c in ch:
                print(f"  {team}: {c}")
    return 0


def pending_proposals(root: Path = TEAMS_DIR) -> list[Path]:
    return sorted(root.glob("*/*/predlozi/*.yaml"))


def cmd_profil_odobri(accept: Optional[str] = None, reject: Optional[str] = None, root: Path = TEAMS_DIR) -> int:
    """Agents only PROPOSE (T9/T10). Tome approves: the proposal is merged into profil.yaml (`opis`) / promeni.md."""
    if not accept and not reject:
        items = pending_proposals(root)
        if not items:
            print("Нема предлози од агентите за одобрение.")
            return 0
        for p in items:
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            print(f"- {p.relative_to(root)}: {data.get('tip', 'ОПИС')} · извор {data.get('izvor_url')} · „{data.get('citat', '')[:80]}“")
        print("Прифати: python main.py profil odobri --prifati <патека>   · Одбиј: --odbij <патека>")
        return 0
    path = root / (accept or reject)
    if not path.exists():
        print(f"Нема предлог {path}")
        return 2
    team_dir = path.parent.parent
    done = team_dir / "predlozi" / ("prifateni" if accept else "odbieni")
    done.mkdir(parents=True, exist_ok=True)
    if accept:
        prop = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        yml = team_dir / "profil.yaml"
        prof = yaml.safe_load(yml.read_text(encoding="utf-8")) if yml.exists() else {}
        if prop.get("tip") in ("TRENER", "TREJD", "POTPISUVANJE", "DOLGA_POVREDA", "STARTNA_PETORKA"):
            with open(team_dir / "promeni.md", "a", encoding="utf-8") as f:
                f.write(f"| {prop.get('objaveno', date.today().isoformat())} | {prop['tip']}: {prop.get('detali', '')} "
                        f"| {prop.get('izvor_url')} („{prop.get('citat', '')[:60]}“) — одобрено од Томе |\n")
        else:
            opis = prof.get("opis") or {}
            # the checker's working lists stay in the archived proposal, not in the profile
            opis.update({k: v for k, v in prop.items() if k not in ("tip", "otfrleno_od_python", "bez_odgovor")})
            opis["odobreno"] = date.today().isoformat()
            prof["opis"] = opis
            yml.write_text(yaml.safe_dump(prof, allow_unicode=True, sort_keys=False), encoding="utf-8")
    shutil.move(str(path), done / path.name)
    print(("Прифатено" if accept else "Одбиено") + f": {path.name}")
    return 0


def _team_folder(code: str, league: str) -> Optional[Path]:
    hits = [p for p in TEAMS_DIR.glob(f"*/{code.upper()}") if p.is_dir() and (league == "all" or p.parent.name == league)]
    if len(hits) != 1:
        print(f"„{code}“: " + ("нема таков тим" if not hits else
                                f"има во повеќе лиги ({', '.join(h.parent.name for h in hits)}) — додај --league"))
        return None
    return hits[0]


def cmd_profil_show(code: str, league: str) -> int:
    folder = _team_folder(code, league)
    if folder is None:
        return 2
    print((folder / "profil.md").read_text(encoding="utf-8"))
    cards = sorted((folder / "igraci").glob("*.md")) if (folder / "igraci").exists() else []
    print(f"Картички ({len(cards)}): " + ", ".join(c.stem for c in cards))
    print(f"Предлози што чекаат: {len(list((folder / 'predlozi').glob('*.yaml'))) if (folder / 'predlozi').exists() else 0}")
    return 0


def cmd_vesti(code: str, league: str, n: int = 15) -> int:
    from model.team_dossier import read_news
    folder = _team_folder(code, league)
    if folder is None:
        return 2
    rows = read_news(folder)
    print(f"Вести — {folder.parent.name}/{folder.name}: {len(rows)} записи (последни {min(n, len(rows))})")
    for r in rows[:n]:
        print(f"- {r['datum']} · {r['tip']} · {r['igrac']} · {r['sto']}\n    влијание: {r['vlijanie']} · {r['izvor']} · "
              f"модели: {r['modeli']}")
    return 0
