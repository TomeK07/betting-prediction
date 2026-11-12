"""Tome 1.11: may T10 read only an excerpt (`t10_news.excerpt`) instead of the whole text? Two checks, then Tome decides.

1. Official T10 test set with excerpts (only the examples the excerpt shortens cost new calls; the rest reuse the cache).
2. Real sources the excerpt shortens: the model's validated items on the FULL text (cached answer, no new call) vs on
   the excerpt (new call) — same model, same Python checks (`_items`). Lost ROTATION/INJURY items weigh most.
→ reports/t10_izvadok_<date>.md. USE_EXCERPT stays False until Tome switches it on.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

from core.config import Settings

MINUTES_TYPES = {"ПОВРЕДА", "СОСТАВ", "РОТАЦИЈА"}


def compare_real(settings: Settings, model: str, limit: int) -> tuple[list[str], dict]:
    from research import t10_news as tn
    from research.llm_client import CACHE, LLMClient
    from research.t9_pilot import _official
    from research.tasks import t10_vesti
    from research.testset_build import Fetcher

    cl = LLMClient(model, timeout_s=90, retries=5)
    key = lambda v, u: hashlib.sha256(f"{v}|{model}|{t10_vesti.SYSTEM_PROMPT}|{u}".encode()).hexdigest()
    sig = lambda items: {(x["tip"], (x["igrac"] or "").lower()) for x in items}
    lines, tot = [], {"izvori": 0, "isti": 0, "izgubeni": 0, "izgubeni_minuti": 0, "novi": 0, "bez_odgovor": 0}
    for team_dir in sorted((tn.TEAMS_DIR).glob("*/*")):
        lg, code = team_dir.parts[-2], team_dir.parts[-1]
        nm = tn.team_names(lg).get(code)
        if not nm or not (team_dir / "izvori").exists() or tot["izvori"] >= limit:
            continue
        _, roster, labels = _official(settings, Fetcher(settings), lg, code)
        keys = [*nm["zborovi"], nm["ime"], *roster]
        for f in sorted((team_dir / "izvori").glob("*.json")):
            if tot["izvori"] >= limit:
                break
            src = json.loads(f.read_text(encoding="utf-8"))
            full = t10_vesti.user_text(nm["ime"], lg, src["izvor_url"], src["objaveno"], src["tekst"])
            cached = CACHE / f"{key(t10_vesti.PROMPT_VERSION, full)}.json"
            ex = tn.excerpt(src["tekst"], keys)
            if ex == src["tekst"] or not cached.exists():
                continue
            a_full = json.loads(cached.read_text(encoding="utf-8"))
            a_full = a_full.get("answer", a_full)
            try:
                a_ex = cl.ask_json(t10_vesti.SYSTEM_PROMPT,
                                   t10_vesti.user_text(nm["ime"], lg, src["izvor_url"], src["objaveno"], ex),
                                   t10_vesti.EXCERPT_VERSION)
            except Exception:
                tot["bez_odgovor"] += 1
                continue
            i_full = sig(tn._items(a_full, src, roster, labels, model)[0])
            i_ex = sig(tn._items(a_ex, src, roster, labels, model)[0])
            lost = i_full - i_ex
            tot["izvori"] += 1
            tot["isti"] += i_full == i_ex
            tot["izgubeni"] += len(lost)
            tot["izgubeni_minuti"] += sum(t in MINUTES_TYPES for t, _ in lost)
            tot["novi"] += len(i_ex - i_full)
            if i_full != i_ex:
                lines.append(f"- {lg}/{code} `{f.name}` {len(src['tekst'])}→{len(ex)} знаци · изгубено {sorted(lost)} · "
                             f"ново {sorted(i_ex - i_full)}")
    return lines, tot


def cmd_izvadok(settings: Settings, today: date, models: list[str], limit: int = 12) -> int:   # 13:30 → should end before the 14:00 run
    from research import t10_news as tn
    from research.llm_eval import run_eval
    tn.USE_EXCERPT = True                              # only inside this test run
    rep = [f"# T10 со извадок — тест · {today}", "",
           f"Извадок = наслов + реченици со тимот/играч од составот (1 пред, 2 по); заштеда на реалните извори ≈ 37% текст.", "",
           "## 1. Официјален T10 тест-сет (само скратените примери се нови повици)", "", run_eval(models, ["T10"]), ""]
    for m in models:
        lines, tot = compare_real(settings, m, limit)
        rep += [f"## 2. Реални извори — {m}", "",
                f"Споредени {tot['izvori']} (без одговор {tot['bez_odgovor']}) · исти ставки {tot['isti']}/{tot['izvori']} · "
                f"изгубени ставки {tot['izgubeni']} (од нив повреда/состав/ротација **{tot['izgubeni_minuti']}**) · нови {tot['novi']}",
                "", *lines, ""]
    rep.append("Одлука (Томе): вклучи `USE_EXCERPT` во `research/t10_news.py` само ако тестот поминува и нема изгубени "
               "повреди/состави/ротации.")
    out = settings.paths.reports / f"t10_izvadok_{today}.md"
    out.write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(rep))
    print(f"\nИзвештај: {out}")
    return 0
