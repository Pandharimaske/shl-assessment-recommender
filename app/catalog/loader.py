"""
Catalog loader — reads shl_product_catalog.json and maps keys → test_type codes.
"""
import json
from pathlib import Path
from typing import Optional

CATALOG_PATH = Path(__file__).parent / "shl_product_catalog.json"


KEYS_TO_TYPE: dict[str, str] = {
    "Ability & Aptitude":           "A",
    "Biodata & Situational Judgment": "B",
    "Competencies":                 "C",
    "Development & 360":            "D",
    "Assessment Exercises":         "E",
    "Knowledge & Skills":           "K",
    "Personality & Behavior":       "P",
    "Simulations":                  "S",
}


def _compute_test_type(keys: list[str]) -> str:
    """Convert keys list to comma-separated test_type string. E.g. ['Knowledge & Skills','Simulations'] → 'K,S'"""
    codes = [KEYS_TO_TYPE[k] for k in keys if k in KEYS_TO_TYPE]
    return ",".join(sorted(set(codes))) if codes else "K"


def load_catalog() -> list[dict]:
    """
    Load and normalise the SHL product catalog.

    Returns list of dicts with fields:
        entity_id, name, url, test_type, description,
        job_levels, languages, duration, remote, adaptive
    """
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"catalog not found at {CATALOG_PATH}. "
            "Download https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json "
            "and save it as app/catalog/shl_product_catalog.json"
        )

    with open(CATALOG_PATH, encoding="utf-8") as f:
        raw: list[dict] = json.load(f)

    catalog = []
    for item in raw:
        if item.get("status") != "ok":
            continue
        catalog.append({
            "entity_id":   item["entity_id"],
            "name":        item["name"],
            "url":         item["link"],
            "test_type":   _compute_test_type(item.get("keys", [])),
            "description": item.get("description", ""),
            "job_levels":  item.get("job_levels", []),
            "languages":   item.get("languages", []),
            "duration":    item.get("duration", ""),
            "remote":      item.get("remote", "yes"),
            "adaptive":    item.get("adaptive", "no"),
            "keys":        item.get("keys", []),
        })

    return catalog



_CATALOG: Optional[list[dict]] = None


def get_catalog() -> list[dict]:
    global _CATALOG
    if _CATALOG is None:
        _CATALOG = load_catalog()
    return _CATALOG
