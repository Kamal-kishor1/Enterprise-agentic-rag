import os
import logfire
from typing import Any, List, Optional
from litellm import Router

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration

from app.config import settings

# 1. Configure the Router with Primary (Groq) and Fallback (Gemini)
model_list = [
    # Primary: Fast inference on Groq
    {
        "model_name": "enterprise-llm",
        "litellm_params": {
            "model": f"groq/{settings.GROQ_MODEL or 'openai/gpt-oss-120b'}",
            "api_key": settings.GROQ_API_KEY,
            "max_tokens": 550,
            "temperature": 0.1,
        },
    },
    # Secondary Fallback: Google Gemini 1.5 Flash (High quota, zero-downtime)
    {
        "model_name": "enterprise-llm",
        "litellm_params": {
            "model": "gemini/gemini-1.5-flash",
            "api_key": settings.GEMINI_API_KEY,
            "max_tokens": 550,
            "temperature": 0.1,
        },
    },
]

# 2. Initialize LiteLLM Router with Fallback Policy
router = Router(
    model_list=model_list,
    routing_strategy="simple-shuffle",
    num_retries=1,
    timeout=10.0,
    fallbacks=[{"enterprise-llm": ["gemini/gemini-1.5-flash"]}],
)


# 3. LangChain BaseChatModel wrapper for native LangGraph compatibility
class ResilientLLMGateway(BaseChatModel):
    """
    Enterprise LLM Gateway providing automatic multi-model failover (Groq -> Gemini).
    Fully compatible with LangGraph async invocation and streaming.
    """

    model_name: str = "enterprise-llm"

    @property
    def _llm_type(self) -> str:
        return "resilient-llm-gateway"

    def _format_messages(self, messages: List[BaseMessage]) -> List[dict]:
        formatted = []
        for m in messages:
            role = "user" if m.type in ("human", "user") else "assistant"
            if m.type == "system":
                role = "system"
            formatted.append({"role": role, "content": m.content})
        return formatted

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        formatted = self._format_messages(messages)
        resp = router.completion(model=self.model_name, messages=formatted, **kwargs)
        content = resp.choices[0].message.content
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content))]
        )

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        with logfire.span("LLM Gateway Execution", model=self.model_name):
            formatted = self._format_messages(messages)
            try:
                resp = await router.acompletion(
                    model=self.model_name, messages=formatted, **kwargs
                )
                used_model = getattr(resp, "model", "unknown")
                logfire.info(f"Gateway routed to: {used_model}", model_used=used_model)
                content = resp.choices[0].message.content
                return ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content=content))]
                )
            except Exception as e:
                logfire.error(f"All LLM Gateway routes failed: {e}")
                raise e


# Export the singleton gateway instance
gateway_llm = ResilientLLMGateway()
