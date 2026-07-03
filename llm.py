"""LLM-agnostic client. Default: Anthropic. Swap by passing a different LLMClient."""
from __future__ import annotations
import os
from abc import ABC, abstractmethod


class LLMClient(ABC):
    @abstractmethod
    def complete(self, messages: list[dict], system: str = "") -> str: ...


class AnthropicClient(LLMClient):
    def __init__(self, model: str = "claude-sonnet-4-6"):
        import anthropic
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        self.client = anthropic.Anthropic()
        self.model = model

    def complete(self, messages: list[dict], system: str = "") -> str:
        resp = self.client.messages.create(
            model=self.model, max_tokens=4096, system=system, messages=messages
        )
        return resp.content[0].text


class OpenAIClient(LLMClient):
    def __init__(self, model: str = "gpt-4o"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model

    def complete(self, messages: list[dict], system: str = "") -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + messages
        resp = self.client.chat.completions.create(model=self.model, messages=msgs)
        return resp.choices[0].message.content


class GeminiClient(LLMClient):
    def __init__(self, model: str = "gemini-2.0-flash"):
        self._model_name = model

    def complete(self, messages: list[dict], system: str = "") -> str:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel(
            self._model_name,
            system_instruction=system or None,
        )
        contents = [
            {"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]}
            for m in messages
        ]
        return model.generate_content(contents).text


class GroqClient(LLMClient):
    def __init__(self, model: str | None = None):
        from groq import Groq
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self.client = Groq()

    def complete(self, messages: list[dict], system: str = "") -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + messages
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=msgs,
            max_tokens=4096,
        )
        return resp.choices[0].message.content


class OllamaClient(LLMClient):
    def __init__(self, model: str | None = None):
        from openai import OpenAI
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        self.client = OpenAI(base_url=base_url, api_key="ollama")

    def complete(self, messages: list[dict], system: str = "") -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + messages
        resp = self.client.chat.completions.create(model=self.model, messages=msgs)
        return resp.choices[0].message.content


class BedrockClient(LLMClient):
    def __init__(self, model: str | None = None, region_name: str | None = None):
        import boto3
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        self.model = model or os.getenv("AWS_BEDROCK_MODEL") or os.getenv("BEDROCK_MODEL", "amazon.nova-lite-v1:0")
        self.region_name = region_name or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        self.client = boto3.client("bedrock-runtime", region_name=self.region_name)

    def complete(self, messages: list[dict], system: str = "") -> str:
        bedrock_msgs = []
        for m in messages:
            role = m["role"]
            if role == "model":
                role = "assistant"
            bedrock_msgs.append({
                "role": role,
                "content": [{"text": m["content"]}]
            })

        system_config = [{"text": system}] if system else []

        resp = self.client.converse(
            modelId=self.model,
            messages=bedrock_msgs,
            system=system_config,
            inferenceConfig={"maxTokens": 4096}
        )
        return resp["output"]["message"]["content"][0]["text"]


def default_client() -> LLMClient:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    provider = os.getenv("LLM_PROVIDER", "").lower()
    if provider == "ollama" or (not provider and os.getenv("OLLAMA_MODEL")):
        return OllamaClient()
    if provider == "gemini" or (not provider and os.getenv("GEMINI_API_KEY")):
        return GeminiClient()
    if provider == "openai" or (not provider and os.getenv("OPENAI_API_KEY")):
        return OpenAIClient()
    if provider == "groq" or (not provider and os.getenv("GROQ_API_KEY")):
        return GroqClient()
    if provider == "bedrock" or provider == "aws" or (not provider and os.getenv("AWS_ACCESS_KEY_ID")):
        return BedrockClient()
    return AnthropicClient()
