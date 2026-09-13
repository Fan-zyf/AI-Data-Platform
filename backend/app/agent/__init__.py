"""v0.7 AI Data Analyst Agent 包。

- :func:`app.agent.agent_service.analyze` 是 API 入口；
- :mod:`app.agent.tools` 提供 5 个 Tool；
- :mod:`app.agent.tool_registry` 提供关键词路由；
- :class:`app.agent.analyst_agent.AgentError` 是统一业务异常。
"""

from app.agent.agent_service import analyze
from app.agent.analyst_agent import AgentError, run_analyst
from app.agent.tool_registry import ToolPlan, ToolRegistry
from app.agent.tools import (
    ALL_TOOLS,
    BaseTool,
    DatasetTool,
    EDATool,
    MLTool,
    ReportTool,
    SHAPTool,
    ToolContext,
    ToolError,
    ToolResult,
)

__all__ = [
    "ALL_TOOLS",
    "AgentError",
    "BaseTool",
    "DatasetTool",
    "EDATool",
    "MLTool",
    "ReportTool",
    "SHAPTool",
    "ToolContext",
    "ToolError",
    "ToolPlan",
    "ToolRegistry",
    "ToolResult",
    "analyze",
    "run_analyst",
]
