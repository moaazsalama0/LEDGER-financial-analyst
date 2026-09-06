import os
import json
import logging
from typing import Any, Dict, Optional
from google import genai
from google.genai import types
from openai import AsyncOpenAI

from app.config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(
        self, 
        provider: Optional[str] = None, 
        model_name: Optional[str] = None, 
        api_key: Optional[str] = None, 
        **kwargs
    ):
        # Resolve provider and model from settings if not explicitly passed
        self.provider = provider or getattr(settings, "AGENT_LLM_PROVIDER", "gemini")
        
        if self.provider == "gemini":
            self.model_name = model_name or getattr(settings, "GEMINI_MODEL", "gemini-2.0-flash")
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
            self.model_name = model_name or getattr(settings, "GROQ_MODEL", "llama-3.3-70b-versatile")
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

        # 1. Mock Provider
        if self.provider == "mock":
            return {
                "query_type": "direct",
                "search_query": user_prompt,
                "reasoning": "Mock mode active - returning synthetic payload.",
                "status": "mock_success",
            }

        # 2. Gemini Provider
        if self.provider == "gemini":
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