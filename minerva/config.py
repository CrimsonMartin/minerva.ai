"""Configuration loading.

Config lives in minerva.config.json at the repo/working root so it is
editable by hand like everything else. `python -m minerva init` writes
the default file.
"""

import json
from pathlib import Path

DEFAULT_CONFIG = {
    "llm": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "local",
        "chat_model": "qwen3:14b",
        "embed_model": "nomic-embed-text",
        "temperature": 0.2,
        "max_tokens": 3000,
        "timeout_seconds": 300,
    },
    "vault": "vault",
    "pubmed": {
        "email": "",
        "seed_results": 20,
        "expand_results": 8,
        "full_text": True,
    },
    "scoring": {
        "depth": {"alpha": 1.0, "beta": 0.15},
        "breadth": {"alpha": 0.55, "beta": 1.0},
    },
    "merge_threshold": 0.80,
    "reflect_every": 8,
    "ingest": {
        "min_chars_per_page": 200,
    },
    "tree": {
        "leaf_chars": 1200,
        "group_chars": 3500,
        "max_paragraphs": 500,
    },
}

CONFIG_FILENAME = "minerva.config.json"


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(root: Path | None = None) -> dict:
    root = root or Path.cwd()
    path = root / CONFIG_FILENAME
    config = DEFAULT_CONFIG
    if path.exists():
        config = _merge(DEFAULT_CONFIG, json.loads(path.read_text()))
    config["_root"] = str(root)
    return config


def write_default_config(root: Path | None = None) -> Path:
    root = root or Path.cwd()
    path = root / CONFIG_FILENAME
    if not path.exists():
        path.write_text(json.dumps(DEFAULT_CONFIG, indent=2) + "\n")
    return path


def vault_path(config: dict) -> Path:
    return Path(config["_root"]) / config["vault"]
