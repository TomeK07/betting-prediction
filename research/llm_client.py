"""OpenAI-compatible client for FreeLLMAPI (LLM_ISTRAZUVACI §7). Disabled unless configured.

LLM = само extract. Анализата е 100% во Python.
FreeLLMAPI се користи ИСКЛУЧИВО за структуриран JSON extract од симнат текст
(T1 повреди, T2 превјуа, T3 улога/минути, T10 вести) — со извор + цитат.
Никогаш: веројатности, проекции, препораки, EV, picks, мислење.

Keys only from .env (never in code). temperature=0, JSON output, timeout, 3 retries, cache by
hash(prompt version + model + prompt + text) so the same text is never sent twice.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from typing import Optional

from core.config import ROOT

log = logging.getLogger(__name__)
CACHE = ROOT / "data" / "raw" / "vesti" / "llm_cache"
# models that stayed rate-limited through every retry: the free quota is spent (often for the day) →
# fail fast for the rest of this run instead of ~7 min of waiting per call (DAL pilot 31.10: 5 h)
EXHAUSTED: set[str] = set()


class LLMDisabled(RuntimeError):
    pass


class LLMClient:
    def __init__(self, model: str, base_url: Optional[str] = None, api_key: Optional[str] = None,
                 timeout_s: float = 60.0, retries: int = 3):
        try:
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env")
        except ImportError:
            pass
        self.model = model
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        if not self.base_url or not self.api_key:
            raise LLMDisabled("LLM_BASE_URL / LLM_API_KEY не се поставени во .env")
        self.timeout_s, self.retries = timeout_s, retries

    def ask_json(self, system_prompt: str, user_text: str, prompt_version: str) -> dict:
        return self.ask_json_meta(system_prompt, user_text, prompt_version)[0]

    def ask_json_meta(self, system_prompt: str, user_text: str, prompt_version: str) -> tuple[dict, dict]:
        """(answer, meta) — meta: provider that served it (FreeLLMAPI `X-Routed-Via`), seconds, from cache."""
        key = hashlib.sha256(f"{prompt_version}|{self.model}|{system_prompt}|{user_text}".encode()).hexdigest()
        path = CACHE / f"{key}.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(saved, dict) and "__meta" in saved:
                return saved["answer"], {**saved["__meta"], "od_kesh": True}
            return saved, {"provajder": None, "sekundi": None, "od_kesh": True}
        if self.model in EXHAUSTED:
            raise RuntimeError(f"LLM {self.model} не одговара: квотата е потрошена (429 на сите обиди претходно)")
        from openai import OpenAI
        client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout_s, max_retries=0)
        last = None
        for attempt in range(1, self.retries + 1):
            t0 = time.monotonic()
            try:
                raw = client.chat.completions.with_raw_response.create(
                    model=self.model, temperature=0, response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}])
                resp = raw.parse()
                data = parse_json_text(resp.choices[0].message.content or "")
                meta = {"provajder": raw.headers.get("x-routed-via"), "sekundi": round(time.monotonic() - t0, 2),
                        "tokeni": getattr(resp.usage, "total_tokens", None) if resp.usage else None}
                CACHE.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"answer": data, "__meta": meta}, ensure_ascii=False), encoding="utf-8")
                return data, {**meta, "od_kesh": False}
            except Exception as exc:                      # network, invalid JSON, provider errors
                last = exc
                log.warning("LLM %s обид %d/%d: %s", self.model, attempt, self.retries, exc)
                if attempt < self.retries:            # 429 = free-tier limit per minute → wait it out
                    time.sleep((30 if "429" in str(exc) else 5) * attempt)
        if "429" in str(last):
            EXHAUSTED.add(self.model)
        raise RuntimeError(f"LLM {self.model} не одговара: {last}")


def parse_json_text(text: str) -> dict:
    """Strict JSON, tolerating a ```json fence or prose around one object (some Gemini models add them)."""
    text = text.strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("одговорот не содржи JSON објект")
    return json.loads(text[start:end + 1])


def task_models(task: str) -> list[str]:
    """settings.yaml → llm_istrazuvaci: models that passed the test for `task`, then the declared checker."""
    import yaml
    cfg = (yaml.safe_load((ROOT / "config" / "settings.yaml").read_text(encoding="utf-8")) or {}).get("llm_istrazuvaci") or {}
    if not cfg.get("enabled"):
        raise LLMDisabled("llm_istrazuvaci.enabled: false")
    models = list((cfg.get("models") or {}).get(task) or [])
    extra = (cfg.get("kontrolor") or {}).get(task)
    return models + ([extra] if extra and extra not in models else [])
