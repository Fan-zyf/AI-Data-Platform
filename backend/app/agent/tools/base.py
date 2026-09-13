"""v0.7 Tool 基类。

每个 Tool 是「对已有 v0.4~v0.6 服务的轻量包装」：

- ``name``：英文短名，路由与 LLM prompt 引用；
- ``description``：中文描述，前端 trace 与 prompt 共用；
- ``execute(dataset_id, experiment_id=None) -> dict``：同步执行；
- ``required_for``：标识本工具的最小入参集合（用于校验）。

工具内部严禁 import 任何 LLM / OpenAI SDK；仅作为「对已有 service 的读取层」存在。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar

logger = logging.getLogger(__name__)


class ToolError(Exception):
    """工具执行异常。"""

    def __init__(self, code: str, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


@dataclass
class ToolContext:
    """工具执行的输入上下文。"""

    dataset_id: str
    experiment_id: str | None = None
    source_version_id: str = "original"
    max_summary_rows: int | None = None
    regenerate_shap: bool = False


@dataclass
class ToolResult:
    """工具执行结果。"""

    name: str
    status: str  # "ok" | "skipped" | "error"
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    summary: str = ""  # 给前端 trace / LLM 用的简短文本

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "data": self.data,
            "error": self.error,
            "summary": self.summary,
        }


class BaseTool:
    """所有 Tool 的基类。"""

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    requires_experiment: ClassVar[bool] = False
    # 关键词路由权重
    keywords: ClassVar[tuple[str, ...]] = ()

    def execute(self, context: ToolContext) -> ToolResult:
        raise NotImplementedError

    def safe_execute(self, context: ToolContext) -> ToolResult:
        try:
            result = self.execute(context)
        except ToolError as exc:
            logger.warning("tool %s error [%s] %s", self.name, exc.code, exc.message)
            return ToolResult(name=self.name, status="error", error=f"[{exc.code}] {exc.message}")
        except Exception as exc:  # noqa: BLE001 - 工具异常统一收敛
            logger.exception("tool %s unexpected error", self.name)
            return ToolResult(name=self.name, status="error", error=f"工具执行失败：{exc}")
        if not isinstance(result, ToolResult):
            return ToolResult(name=self.name, status="ok", data={"value": result})
        return result

    def matches(self, question: str) -> bool:
        if not self.keywords:
            return False
        lowered = (question or "").lower()
        return any(kw.lower() in lowered for kw in self.keywords)
