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
        self.embed_base_url = (llm.get("embed_base_url") or llm["base_url"]).rstrip("/")
        self.chat_model = llm["chat_model"]
        self.report_model = llm.get("report_model") or llm["chat_model"]
        self.chat_extra_body = llm.get("chat_extra_body") or {}
        self.report_extra_body = llm.get("report_extra_body") or {}
        self.embed_model = llm["embed_model"]
        self.temperature = llm["temperature"]
        self.max_tokens = llm["max_tokens"]
        self.timeout = llm["timeout_seconds"]
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {llm['api_key']}"
        self._local_embedder = None

    # ------------------------------------------------------------- chat

    def chat(self, system: str, user: str, response_format: dict | None = None,
             model: str | None = None, extra_body: dict | None = None) -> str:
        body = {
            "model": model or self.chat_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        # Deployment-specific fields (e.g. chat_template_kwargs); the small
        # structured calls default to chat_extra_body, callers may override.
        body.update(self.chat_extra_body if extra_body is None else extra_body)
        if response_format:
            body["response_format"] = response_format
        response = self.session.post(
            f"{self.base_url}/chat/completions", json=body, timeout=self.timeout
        )
        if response.status_code == 400 and response_format:
            # Server rejected the format constraint — degrade one rung:
            # json_schema -> json_object -> prompt-only JSON.
            if response_format.get("type") == "json_schema":
                return self.chat(system, user, {"type": "json_object"},
                                 model=model, extra_body=extra_body)
            return self.chat(system, user, model=model, extra_body=extra_body)
        if response.status_code != 200:
            raise LLMError(f"chat failed ({response.status_code}): {response.text[:500]}")
        return response.json()["choices"][0]["message"]["content"]

    def chat_json(self, system: str, user: str, schema: dict | None = None) -> dict:
        """Chat expecting a JSON object back, with one repair retry.

        With `schema`, servers that support structured output (LM Studio,
        llama.cpp, vLLM) constrain decoding so the reply can't be malformed
        or missing required keys; others degrade via the ladder in chat().
        """
        response_format: dict = {"type": "json_object"}
        if schema:
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": schema.get("title", "reply"),
                                "strict": True, "schema": schema},
            }
        text = self.chat(system, user, response_format)
        try:
            return _extract_json(text)
        except ValueError:
            retry = (
                f"{user}\n\nYour previous reply was not valid JSON. "
                "Reply again with ONLY a valid JSON object, no prose."
            )
            text = self.chat(system, retry, response_format)
            try:
                return _extract_json(text)
            except ValueError as exc:
                # Surface as LLMError so callers treat it like any other
                # failed call (skip the step), not a crash.
                raise LLMError(f"unparseable JSON reply after retry: {exc}") from exc

    # -------------------------------------------------------- embeddings

    def embed(self, text: str) -> list[float]:
        # "local:<hf-model-id>" runs the model in-process (no server needed);
        # anything else goes to the OpenAI-compatible endpoint.
        if self.embed_model.startswith("local:"):
            if self._local_embedder is None:
                from sentence_transformers import SentenceTransformer
                self._local_embedder = SentenceTransformer(self.embed_model[len("local:"):])
            return self._local_embedder.encode(text, normalize_embeddings=True).tolist()
        response = self.session.post(
            f"{self.embed_base_url}/embeddings",
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
