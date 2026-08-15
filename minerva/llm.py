"""Thin client for any OpenAI-compatible endpoint (LM Studio, vLLM, llama.cpp, Ollama).

Every call is a single-purpose request. Local models do far better with
many small, schema-constrained calls than one long agentic context, so
this module offers exactly two things: chat() for free text and
chat_json() for structured output with lenient parsing plus one retry.
"""

import json

import requests


class LLMError(RuntimeError):
    pass


class LLM:
    def __init__(self, config: dict):
        llm = config["llm"]
        self.base_url = llm["base_url"].rstrip("/")
        self.chat_model = llm["chat_model"]
        self.embed_model = llm["embed_model"]
        self.temperature = llm["temperature"]
        self.max_tokens = llm["max_tokens"]
        self.timeout = llm["timeout_seconds"]
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {llm['api_key']}"

    # ------------------------------------------------------------- chat

    def chat(self, system: str, user: str, json_mode: bool = False) -> str:
        body = {
            "model": self.chat_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        response = self.session.post(
            f"{self.base_url}/chat/completions", json=body, timeout=self.timeout
        )
        if response.status_code == 400 and json_mode:
            # Some servers reject response_format; fall back to prompt-only JSON.
            body.pop("response_format")
            response = self.session.post(
                f"{self.base_url}/chat/completions", json=body, timeout=self.timeout
            )
        if response.status_code != 200:
            raise LLMError(f"chat failed ({response.status_code}): {response.text[:500]}")
        return response.json()["choices"][0]["message"]["content"]

    def chat_json(self, system: str, user: str) -> dict:
        """Chat expecting a JSON object back, with one repair retry."""
        text = self.chat(system, user, json_mode=True)
        try:
            return _extract_json(text)
        except ValueError:
            retry = (
                f"{user}\n\nYour previous reply was not valid JSON. "
                "Reply again with ONLY a valid JSON object, no prose."
            )
            text = self.chat(system, retry, json_mode=True)
            return _extract_json(text)

    # -------------------------------------------------------- embeddings

    def embed(self, text: str) -> list[float]:
        response = self.session.post(
            f"{self.base_url}/embeddings",
            json={"model": self.embed_model, "input": text},
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise LLMError(f"embed failed ({response.status_code}): {response.text[:500]}")
        return response.json()["data"][0]["embedding"]


def _extract_json(text: str) -> dict:
    """Pull the first balanced JSON object out of a model reply."""
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in reply")
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
        elif ch == "\\":
            escape = True
        elif ch == '"':
            in_string = not in_string
        elif not in_string:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start : i + 1])
    raise ValueError("unbalanced JSON object in reply")
