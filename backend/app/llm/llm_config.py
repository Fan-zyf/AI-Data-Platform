"""v0.7 LLM 配置。

约定：
- 通过环境变量启用真实 LLM：``LLM_API_KEY`` 必须非空、``LLM_BASE_URL`` 必须非空、``LLM_MODEL`` 必须非空。
- 任何一项缺失 → 自动降级为 Mock 模式，**绝不抛错阻塞测试**。
- 真实 key 仅允许来自环境变量；源码中绝无硬编码。
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    timeout_seconds: float
    temperature: float
    max_tokens: int

    @property
    def is_mock(self) -> bool:
        return not (self.api_key and self.base_url and self.model)


def load_llm_config() -> LLMConfig:
    base_url = (os.getenv("LLM_BASE_URL") or "").rstrip("/")
    api_key = (os.getenv("LLM_API_KEY") or "").strip()
    model = (os.getenv("LLM_MODEL") or "").strip()
    try:
        timeout_seconds = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
    except ValueError:
        timeout_seconds = 30.0
    try:
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    except ValueError:
        temperature = 0.3
    try:
        max_tokens = int(os.getenv("LLM_MAX_TOKENS", "1024"))
    except ValueError:
        max_tokens = 1024
    return LLMConfig(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout_seconds=timeout_seconds,
        temperature=temperature,
        max_tokens=max_tokens,
    )
