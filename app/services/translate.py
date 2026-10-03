"""Marathi -> English translation via GROQ (OpenAI-compatible chat API).

- No key (NLP_GROQ_API_KEY empty) -> TranslationResult(available=False);
  analysis and all other endpoints keep working, frontend shows an honest
  empty state instead of failing.
- Long documents are chunked by paragraph (blank-line split) so no single
  request exceeds model limits; chunk translations are joined with "\\n\\n".
- Any network/API error -> available=False with a short reason (never 500
  from /translate; the route returns 200 + unavailable, 422 only for blank
  input, 503 never -- translation is enhancement, not gating).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.core.config import settings

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
# Small chunks keep each request fast and inside free-tier per-minute token
# budgets (Devanagari tokenizes to ~2-4 chars/token, so 2500 chars is roughly
# 600-1200 input tokens). Long PDFs that previously went out as ONE giant
# request now go out as several small ones with 429 backoff between them.
MAX_CHARS_PER_CALL = 2500
MAX_429_RETRIES = 3


@dataclass
class TranslationResult:
    available: bool
    text: str = ""
    provider: str = "groq"
    model: str = ""
    reason: str = ""
    chunks: int = 0


def _chunk_paragraphs(text: str, limit: int = MAX_CHARS_PER_CALL) -> list[str]:
    paras = [p.strip() for p in text.split("\n\n")]
    paras = [p for p in paras if p]
    if not paras:
        return [text] if text.strip() else []
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for para in paras:
        extra = len(para) + (2 if current else 0)
        if current and current_len + extra > limit:
            chunks.append("\n\n".join(current))
            current, current_len = [para], len(para)
        else:
            current.append(para)
            current_len += extra
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _output_budget(chunk: str) -> int:
    """Output token budget scaled to chunk size (avoids cut-off long docs)."""
    return max(512, min(4096, len(chunk) // 2))


SYSTEM_PROMPT = (
    "You are a Marathi to English translator for legal and financial "
    "documents. Translate the ENTIRE user text from start to finish -- "
    "every paragraph, every line. Never stop after the first lines, never "
    "summarize, never refuse long texts. Preserve numbers, dates, amounts "
    "and names exactly. Return ONLY the English translation, no commentary."
)


def _post_chat(payload: dict, headers: dict, timeout_s: float):
    """POST with 429 backoff. Returns the raw response (200) or raises."""
    import time

    import httpx

    attempts = 0
    while True:
        attempts += 1
        try:
            resp = httpx.post(GROQ_URL, json=payload, headers=headers,
                              timeout=timeout_s)
        except Exception as exc:
            raise RuntimeError(f"translation request failed: {exc}") from exc
        if resp.status_code in (401, 403):
            raise RuntimeError("translation rejected (invalid API key).")
        if resp.status_code == 429 and attempts <= MAX_429_RETRIES:
            retry_after = 0.0
            try:
                retry_after = float(
                    resp.headers.get("retry-after", 2 * attempts))
            except Exception:
                retry_after = float(2 * attempts)
            time.sleep(min(max(retry_after, 0.0), 30.0))
            continue
        if resp.status_code == 429:
            raise RuntimeError(
                "translation rate-limited (free-tier quota busy) -- "
                "press Retry in a few seconds.")
        if resp.status_code >= 400:
            raise RuntimeError(
                f"translation failed (HTTP {resp.status_code}).")
        return resp


def _parse_content(resp) -> tuple[str, str]:
    try:
        data = resp.json()
        choice = data["choices"][0]
        content = choice["message"]["content"]
        finish = str(choice.get("finish_reason") or "")
    except Exception as exc:
        raise RuntimeError(f"unexpected translation response: {exc}") from exc
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("empty translation response.")
    return content.strip(), finish


def _split_half(piece: str) -> tuple[str, str]:
    """Split a stubborn piece near its middle (paragraph/line/word edge)."""
    mid = len(piece) // 2
    cut = -1
    for sep, keep in (("\n\n", 2), ("\n", 1)):
        idx = piece.rfind(sep, 0, mid + 500)
        if idx > 100:
            cut = idx + keep
            break
    if cut < 0:
        idx = piece.rfind(" ", 0, mid + 200)
        cut = idx + 1 if idx > 100 else mid
    head, tail = piece[:cut].strip(), piece[cut:].strip()
    return (head, tail) if head and tail else (piece, "")


def _translate_piece(piece: str, api_key: str, model: str, timeout_s: float,
                     headers: dict, depth: int = 0) -> str:
    budget = _output_budget(piece)
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": budget,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": piece},
        ],
    }
    content, finish = _parse_content(
        _post_chat(payload, headers, timeout_s))
    if finish == "length":
        # Output budget cut the translation off: retry once with 2x budget.
        payload["max_tokens"] = min(8192, budget * 2)
        content, finish = _parse_content(
            _post_chat(payload, headers, timeout_s))
    if len(piece) > 700 and len(content) < 0.25 * len(piece) and depth < 2:
        # Model stopped early (e.g. title only). Never argue with it via a
        # "continue" follow-up -- that elicits refusal sentences that pollute
        # the output. Instead split and translate smaller halves, which the
        # model handles reliably, then stitch.
        head, tail = _split_half(piece)
        if tail:
            return (_translate_piece(head, api_key, model, timeout_s,
                                     headers, depth + 1)
                    + "\n\n"
                    + _translate_piece(tail, api_key, model, timeout_s,
                                       headers, depth + 1))
    return content


def _translate_chunk(chunk: str, api_key: str, model: str,
                     timeout_s: float) -> str:
    headers = {"Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json"}
    return _translate_piece(chunk, api_key, model, timeout_s, headers, 0)


def translate_to_english(text: str,
                         api_key: str | None = None,
                         model: str | None = None,
                         timeout_s: float | None = None) -> TranslationResult:
    """Translate Marathi text to English (graceful unavailable statuses)."""
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}.")
    if not text.strip():
        raise ValueError("Input text must not be empty.")
    key = api_key if api_key is not None else settings.groq_api_key
    chosen = model or settings.groq_model
    timeout = timeout_s if timeout_s is not None else settings.groq_timeout_s
    if not key or not key.strip():
        return TranslationResult(
            available=False, provider="groq", model=chosen,
            reason=("No GROQ API key configured (NLP_GROQ_API_KEY). "
                    "Add it to .env to enable English translation."))
    chunks = _chunk_paragraphs(text.strip())
    if not chunks:
        raise ValueError("Input text must not be empty.")
    translated: list[str] = []
    for pos, chunk in enumerate(chunks):
        if pos > 0:
            import time as _time
            _time.sleep(1.0)  # be gentle with free-tier requests/minute
        try:
            translated.append(_translate_chunk(chunk, key.strip(), chosen, timeout))
        except RuntimeError as exc:
            return TranslationResult(
                available=False, provider="groq", model=chosen,
                reason=str(exc), chunks=len(translated))
    return TranslationResult(
        available=True, text="\n\n".join(translated).strip(),
        provider="groq", model=chosen, chunks=len(translated))
