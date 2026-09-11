import os
import json
import logging
from typing import Any, Dict, Optional

from app.config import settings

logger = logging.getLogger(__name__)


def _is_provider_error(exc: Exception) -> bool:
    """Return true for remote/API failures that should trigger Groq failover."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code is not None:
        return True
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "resource_exhausted",
            "quota",
            "rate limit",
            "unavailable",
            "timeout",
            "connection",
            "http 4",
            "http 5",
        )
    )


class LLMClient:
    def __init__(
        self, 
        provider: Optional[str] = None, 
        model_name: Optional[str] = None, 
        api_key: Optional[str] = None, 
        **kwargs
    ):
        # Resolve provider and model from settings if not explicitly passed
        self.provider = (provider or getattr(settings, "AGENT_LLM_PROVIDER", "hybrid")).strip().lower()
        self.gemini_client = None
        self.groq_client = None
        self.provider_clients = []
        self.last_provider = None

        if self.provider == "hybrid":
            errors = []
            for provider_name in ("gemini", "groq"):
                try:
                    self.provider_clients.append(LLMClient(provider=provider_name))
                except Exception as exc:
                    errors.append(f"{provider_name}: {exc}")
                    logger.warning("Hybrid provider %s is unavailable: %s", provider_name, exc)
            if not self.provider_clients:
                raise RuntimeError("No hybrid LLM provider is available: " + "; ".join(errors))
            self.model_name = " -> ".join(
                f"{client.provider}:{client.model_name}" for client in self.provider_clients
            )

        elif self.provider == "gemini":
            try:
                from google import genai
            except ImportError as exc:
                raise RuntimeError(
                    "Gemini support is not installed. Install requirements-gemini.txt."
                ) from exc
            self.model_name = model_name or getattr(settings, "GEMINI_MODEL", "gemini-3.6-flash")
            key = (
                api_key 
                or getattr(settings, "GEMINI_API_KEY", None) 
                or os.environ.get("GEMINI_API_KEY") 
                or os.environ.get("GOOGLE_API_KEY")
            )
            if not key:
                raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY is not set in settings or environment.")
            self.gemini_client = genai.Client(api_key=key)

        elif self.provider == "groq":
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "Groq support is not installed. Install requirements-groq.txt."
                ) from exc
            self.model_name = model_name or getattr(settings, "GROQ_MODEL", "openai/gpt-oss-120b")
            key = (
                api_key 
                or getattr(settings, "GROQ_API_KEY", None) 
                or os.environ.get("GROQ_API_KEY")
            )
            if not key:
                raise ValueError("GROQ_API_KEY is not set in settings or environment.")
            self.groq_client = AsyncOpenAI(
                api_key=key,
                base_url="https://api.groq.com/openai/v1"
            )

        elif self.provider == "mock":
            self.model_name = model_name or "mock-model"

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    async def chat_json(
        self, 
        messages: Optional[list] = None, 
        prompt: Optional[str] = None, 
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        response_schema: Optional[Any] = None,
        schema: Optional[Any] = None,
        **kwargs
    ) -> dict:
        """Executes a chat completion and returns a parsed JSON dictionary."""
        
        # Resolve parameter aliases
        response_schema = response_schema or schema or kwargs.get("response_schema") or kwargs.get("schema")

        if prompt is not None and user_prompt is None:
            user_prompt = prompt

        if messages:
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "system" and not system_prompt:
                    system_prompt = content
                elif role == "user" and not user_prompt:
                    user_prompt = content

        user_prompt = user_prompt or kwargs.get("user_prompt", "")
        system_prompt = system_prompt or kwargs.get("system_prompt", "You are a helpful AI assistant that outputs structured JSON.")

        if not user_prompt and not messages:
            raise ValueError("Either 'messages', 'prompt', or 'user_prompt' must be provided to chat_json().")

        # Hybrid mode tries Gemini first and fails over to Groq on any provider
        # or parsing failure. It never runs both providers after a success.
        if self.provider == "hybrid":
            last_error = None
            for client in self.provider_clients:
                try:
                    result = await client.chat_json(
                        user_prompt=user_prompt,
                        system_prompt=system_prompt,
                        response_schema=response_schema,
                    )
                    self.last_provider = client.provider
                    return result
                except Exception as exc:
                    last_error = exc
                    logger.warning(
                        "LLM provider %s failed; trying next hybrid provider: %s",
                        client.provider,
                        exc,
                    )
            raise RuntimeError(f"All hybrid LLM providers failed: {last_error}")

        # 1. Mock Provider
        if self.provider == "mock":
            schema_name = getattr(response_schema, "__name__", "")
            if schema_name == "ReasonResponse":
                if "(no evidence)" in user_prompt:
                    return {"reason": "Mock mode found no evidence."}
                return {"value": "314,258 (thousand)"}
            return {
                "query_type": "direct",
                "search_query": user_prompt,
                "reasoning": "Mock mode active - returning synthetic payload.",
                "status": "mock_success",
            }

        # 2. Gemini Provider
        if self.provider == "gemini":
            from google.genai import types

            if not self.gemini_client:
                raise RuntimeError("gemini_client is not initialized. Ensure GEMINI_API_KEY is set.")

            config_kwargs = {
                "system_instruction": system_prompt,
                "response_mime_type": "application/json",
                "temperature": 0.0,
            }

            if response_schema:
                try:
                    config = types.GenerateContentConfig(
                        **config_kwargs,
                        response_schema=response_schema,
                    )
                    response = await self.gemini_client.aio.models.generate_content(
                        model=self.model_name,
                        contents=user_prompt,
                        config=config,
                    )
                    return json.loads(response.text)
                except Exception as schema_err:
                    # Quota, availability, and transport failures must escape
                    # to hybrid mode instead of spending a second Gemini call.
                    if _is_provider_error(schema_err):
                        raise
                    logger.warning(
                        f"Gemini native schema enforcement failed ({schema_err}). "
                        "Falling back to prompt-guided JSON generation."
                    )

            # Fallback without strict response_schema if schema validation failed or wasn't provided
            fallback_prompt = (
                f"{system_prompt}\n\n"
                f"Respond strictly in valid JSON matching the required schema."
            )
            config = types.GenerateContentConfig(
                system_instruction=fallback_prompt,
                response_mime_type="application/json",
                temperature=0.0,
            )
            response = await self.gemini_client.aio.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config=config,
            )
            return json.loads(response.text)

        # 3. Groq Provider
        if self.provider == "groq":
            if not self.groq_client:
                raise RuntimeError("groq_client is not initialized. Ensure GROQ_API_KEY is set.")

            groq_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            response = await self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=groq_messages,
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            raw_content = response.choices[0].message.content
            return json.loads(raw_content)

        raise RuntimeError(f"Unhandled provider: {self.provider}")
