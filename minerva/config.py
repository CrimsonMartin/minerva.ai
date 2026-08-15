"""Configuration loading.

Config lives in minerva.config.json at the repo/working root so it is
editable by hand like everything else. `python -m minerva init` writes
the default file. LLM connection settings can also come from the
environment (or a `.env` file at the root) via the MINERVA_LLM_*
variables — those win over the JSON file, so a `.env` can point one
checkout at a different LM Studio host/model without touching config.
"""

import json
import os
from pathlib import Path

DEFAULT_CONFIG = {
    "llm": {
        "base_url": "http://localhost:1234/v1",
        "api_key": "lm-studio",
        "chat_model": "qwen/qwen3-14b",
        "embed_model": "text-embedding-nomic-embed-text-v1.5",
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
    "link_threshold": 0.62,
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

# Environment variable → path into the config dict.
ENV_OVERRIDES = {
    "MINERVA_LLM_BASE_URL": ("llm", "base_url"),
    "MINERVA_LLM_API_KEY": ("llm", "api_key"),
    "MINERVA_LLM_CHAT_MODEL": ("llm", "chat_model"),
    "MINERVA_LLM_EMBED_MODEL": ("llm", "embed_model"),
}


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_dotenv(root: Path) -> dict[str, str]:
    """Parse KEY=VALUE lines from `.env` (no export, quotes optional)."""
    path = root / ".env"
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values


def load_config(root: Path | None = None) -> dict:
    root = root or Path.cwd()
    path = root / CONFIG_FILENAME
    config = DEFAULT_CONFIG
    if path.exists():
        config = _merge(DEFAULT_CONFIG, json.loads(path.read_text()))
    dotenv = _read_dotenv(root)
    for env_key, (section, key) in ENV_OVERRIDES.items():
        value = os.environ.get(env_key, dotenv.get(env_key))
        if value:
            config = _merge(config, {section: {key: value}})
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
