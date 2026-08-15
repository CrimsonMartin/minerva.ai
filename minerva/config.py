"""Configuration.

Settings live in DEFAULT_CONFIG below — edit them there. The LLM
connection can be overridden without touching code via the MINERVA_LLM_*
variables, from the environment or a `.env` file at the root
(environment wins over `.env`).
"""

import json
import os
from pathlib import Path

DEFAULT_CONFIG = {
    "llm": {
        "base_url": "http://localhost:1234/v1",
        "api_key": "lm-studio",
        "chat_model": "qwen/qwen3-14b",
        # Optional separate model for the final report synthesis (e.g. a
        # thinking variant, while chat_model runs fast for the many small
        # structured calls). Empty = use chat_model.
        "report_model": "",
        # Extra request-body fields merged into chat calls, by role. Lets a
        # deployment pass server-specific knobs without touching code — e.g.
        # disable a thinking model's reasoning for the many small structured
        # calls while the one big synthesis call keeps it:
        #   chat_extra_body: {"chat_template_kwargs": {"enable_thinking": false}}
        "chat_extra_body": {},
        "report_extra_body": {},
        "embed_base_url": "",  # optional separate endpoint for embeddings; falls back to base_url
        # "local:<hf-model-id>" embeds in-process via sentence-transformers
        # (downloads on first use); a plain name uses the HTTP endpoint.
        "embed_model": "local:Qwen/Qwen3-Embedding-0.6B",
        "temperature": 0.2,
        # Reasoning/thinking models spend tokens on reasoning before the
        # answer, and that spend counts against max_tokens — leave headroom
        # or long structured replies get truncated mid-JSON.
        "max_tokens": 8192,
        "timeout_seconds": 300,
    },
    "vault": "vault",
    "pubmed": {
        "email": "REDACTED@example.com",
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

# Environment variable → path into the config dict.
ENV_OVERRIDES = {
    "MINERVA_LLM_BASE_URL": ("llm", "base_url"),
    "MINERVA_LLM_API_KEY": ("llm", "api_key"),
    "MINERVA_LLM_CHAT_MODEL": ("llm", "chat_model"),
    "MINERVA_LLM_REPORT_MODEL": ("llm", "report_model"),
    "MINERVA_LLM_CHAT_EXTRA_BODY": ("llm", "chat_extra_body"),
    "MINERVA_LLM_REPORT_EXTRA_BODY": ("llm", "report_extra_body"),
    "MINERVA_LLM_EMBED_BASE_URL": ("llm", "embed_base_url"),
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
    config = dict(DEFAULT_CONFIG)  # never hand back the module-level dict itself
    dotenv = _read_dotenv(root)
    for env_key, (section, key) in ENV_OVERRIDES.items():
        value = os.environ.get(env_key, dotenv.get(env_key))
        if value:
            if isinstance(DEFAULT_CONFIG[section][key], dict):
                value = json.loads(value)  # dict-valued settings are JSON in env
            config = _merge(config, {section: {key: value}})
    config["_root"] = str(root)
    return config


def vault_path(config: dict) -> Path:
    return Path(config["_root"]) / config["vault"]
