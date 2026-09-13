"""v0.7 LLM 客户端包。

- :class:`LLMClient` 协议
- :class:`HTTPLLMClient`（OpenAI-compatible /chat/completions）
- :class:`MockLLMClient`（无 LLM 时的兜底）
- :func:`get_default_client` 工厂
- :class:`LLMError` 业务异常
"""

from app.llm.llm_client import (
    HTTPLLMClient,
    LLMClient,
    LLMError,
    MockLLMClient,
    get_default_client,
    reset_default_client_for_test,
)
from app.llm.llm_config import LLMConfig, load_llm_config

__all__ = [
    "HTTPLLMClient",
    "LLMClient",
    "LLMConfig",
    "LLMError",
    "MockLLMClient",
    "get_default_client",
    "load_llm_config",
    "reset_default_client_for_test",
]
