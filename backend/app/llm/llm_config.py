"""v0.7 LLM 配置。

约定：
- 通过环境变量启用真实 LLM：``LLM_API_KEY`` 必须非空、``LLM_BASE_URL`` 必须非空、``LLM_MODEL`` 必须非空。
- 任何一项缺失 → 自动降级为 Mock 模式，**绝不抛错阻塞测试**。
- 真实 key 仅允许来自环境变量；源码中绝无硬编码。
- 自动从仓库根目录 ``.env`` 加载（若已通过 shell 显式设置，则 shell 优先，不被覆盖）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_repo_root_dotenv() -> None:
    """从仓库根目录 ``.env`` 注入环境变量（仅补缺失，不覆盖）。"""
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    # backend/app/llm/llm_config.py → 4 级 parents 是仓库根
    root = Path(__file__).resolve().parents[3]
    env_path = root / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=False)


_load_repo_root_dotenv()


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
