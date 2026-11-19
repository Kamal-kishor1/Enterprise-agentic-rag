import os

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = (
    "True"  # Prevents slow network requests for model price maps
)
import logfire
from typing import Any, List, Optional
from litellm import Router

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk
from langchain_core.outputs import ChatResult, ChatGeneration, ChatGenerationChunk

from app.config import settings

# 1. Configure the Router with Primary (Groq) and Fallback (Groq)
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
    # Secondary Fallback: Groq 20B with Fallback API Key (High quota, zero-downtime Fallover)
    {
        "model_name": "enterprise-llm",
        "litellm_params": {
            "model": "groq/openai/gpt-oss-20b",
            "api_key": settings.GROQ_FALLBACK_API_KEY or settings.GROQ_API_KEY,
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
    fallbacks=[{"enterprise-llm": ["enterprise-llm-fallback"]}],
)


# 3. LangChain BaseChatModel wrapper for native LangGraph compatibility
class ResilientLLMGateway(BaseChatModel):
    """
    Enterprise LLM Gateway providing automatic multi-model failover (Groq -> Gemini).
    Fully compatible with LangGraph async invocation and SSE streaming.
    """

    model_name: str = "enterprise-llm"
    streaming: bool = False

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
                if self.streaming:
                    # Stream tokens in real-time and dispatch to LangChain's callback manager
                    response = await router.acompletion(
                        model=self.model_name,
                        messages=formatted,
                        stream=True,
                        **kwargs,
                    )
                    collected_tokens = []
                    async for chunk in response:
                        delta = chunk.choices[0].delta.content or ""
                        if delta:
                            collected_tokens.append(delta)
                            if run_manager:
                                # This triggers the 'on_chat_model_stream' event in LangGraph!
                                await run_manager.on_llm_new_token(
                                    delta,
                                    chunk=ChatGenerationChunk(
                                        message=AIMessageChunk(content=delta)
                                    ),
                                )
                    content = "".join(collected_tokens)
                else:
                    resp = await router.acompletion(
                        model=self.model_name,
                        messages=formatted,
                        stream=False,
                        **kwargs,
                    )
                    content = resp.choices[0].message.content

                used_model = (
                    getattr(resp, "model", "unknown")
                    if not self.streaming
                    else "enterprise-llm-stream"
                )
                logfire.info(f"Gateway routed to: {used_model}", model_used=used_model)

                return ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content=content))]
                )
            except Exception as e:
                logfire.error(f"All LLM Gateway routes failed: {e}")
                raise e


# Export dedicated instances:
# 1. Non-streaming for planner (fast internal decisions)
gateway_llm = ResilientLLMGateway(streaming=False)

# 2. Streaming for responder (real-time SSE token emission)
gateway_llm_streaming = ResilientLLMGateway(streaming=True)
