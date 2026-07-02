"""Cost estimation + budget caps for OpenAI image generation.

Image-generation pricing changes over time and (for ``gpt-image-1``) is actually
token-based, so the numbers here are **deliberately approximate** per-image USD
figures. They exist to power a *budget cap* and a dry-run cost preview, NOT to be
an exact invoice. Every estimate carries ``approximate: True`` and (when the model
is unknown) ``known_pricing: False`` so callers can warn loudly.

Verify current prices at https://openai.com/api/pricing/ before a large run.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

__all__ = [
    "PRICES", "DEFAULT_MODEL", "DEFAULT_QUALITY", "FALLBACK_PER_IMAGE_USD",
    "parse_size", "per_image_usd", "estimate_cost", "within_budget", "PRICING_NOTE",
]

DEFAULT_MODEL = "gpt-image-1"
DEFAULT_QUALITY = "medium"
# Used when a model is not in the table — intentionally a bit pessimistic so a cap
# is conservative rather than surprising.
FALLBACK_PER_IMAGE_USD = 0.08

PRICING_NOTE = ("APPROXIMATE per-image USD — verify at https://openai.com/api/pricing/ "
                "(gpt-image-1 is token-based; these are rough averages).")

# model -> quality -> size_bucket -> approx USD/image.
# Sizes are bucketed by the longer side: "1024" (square-ish) vs "1536"/"1792" (large).
PRICES: Dict[str, Dict[str, Dict[str, float]]] = {
    "gpt-image-1": {
        "low":    {"1024": 0.011, "1536": 0.016},
        "medium": {"1024": 0.042, "1536": 0.063},
        "high":   {"1024": 0.167, "1536": 0.25},
    },
    "dall-e-3": {
        "standard": {"1024": 0.04, "1792": 0.08},
        "hd":       {"1024": 0.08, "1792": 0.12},
    },
    "dall-e-2": {
        "standard": {"256": 0.016, "512": 0.018, "1024": 0.02},
    },
}

# Quality aliases so callers can pass dall-e or gpt-image vocabularies loosely.
_QUALITY_ALIASES = {
    "auto": "medium", "standard": "medium", "hd": "high",
    "low": "low", "medium": "medium", "high": "high",
}


def parse_size(size: Any) -> Tuple[int, int]:
    """Parse ``1024`` / ``"1024"`` / ``"1024x1536"`` / ``"auto"`` -> ``(w, h)``.

    ``auto`` (or empty) -> ``(1024, 1024)``.
    """
    if size is None:
        return (1024, 1024)
    if isinstance(size, int):
        return (size, size)
    s = str(size).strip().lower()
    if s in ("", "auto"):
        return (1024, 1024)
    if "x" in s:
        a, b = s.split("x", 1)
        return (int(a), int(b))
    n = int(s)
    return (n, n)


def size_string(size: Any) -> str:
    w, h = parse_size(size)
    return f"{w}x{h}"


def _size_bucket(size: Any, table_for_quality: Dict[str, float]) -> str:
    """Pick the closest available size key (by longer side) from a price row."""
    w, h = parse_size(size)
    longest = max(w, h)
    keys = sorted(int(k) for k in table_for_quality)
    # choose the smallest bucket >= longest, else the largest available
    chosen = next((k for k in keys if k >= longest), keys[-1])
    return str(chosen)


def per_image_usd(model: str, size: Any, quality: str = DEFAULT_QUALITY) -> Tuple[float, bool]:
    """Return ``(usd_per_image, known_pricing)`` for one image."""
    model = (model or DEFAULT_MODEL).strip()
    q = _QUALITY_ALIASES.get((quality or DEFAULT_QUALITY).strip().lower(), DEFAULT_QUALITY)
    table = PRICES.get(model)
    if table is None:
        return FALLBACK_PER_IMAGE_USD, False
    qrow = table.get(q) or next(iter(table.values()))
    bucket = _size_bucket(size, qrow)
    return float(qrow[bucket]), True


def estimate_cost(
    model: str, size: Any, n: int, quality: str = DEFAULT_QUALITY,
) -> Dict[str, Any]:
    """Estimate the total cost of generating ``n`` images. Always approximate."""
    n = max(0, int(n))
    unit, known = per_image_usd(model, size, quality)
    total = round(unit * n, 4)
    return {
        "model": model,
        "size": size_string(size),
        "quality": quality,
        "n": n,
        "per_image_usd": round(unit, 4),
        "total_usd": total,
        "approximate": True,
        "known_pricing": known,
        "note": PRICING_NOTE,
    }


def within_budget(estimate: Dict[str, Any], max_cost_usd: Optional[float]) -> bool:
    """True if the estimate is within ``max_cost_usd`` (None ⇒ no cap ⇒ always True)."""
    if max_cost_usd is None:
        return True
    return float(estimate.get("total_usd", 0.0)) <= float(max_cost_usd)
