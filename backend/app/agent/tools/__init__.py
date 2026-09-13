"""v0.7 Tools 包：统一对外暴露 5 个工具类。"""

from app.agent.tools.base import BaseTool, ToolContext, ToolError, ToolResult
from app.agent.tools.dataset_tool import DatasetTool
from app.agent.tools.eda_tool import EDATool
from app.agent.tools.ml_tool import MLTool
from app.agent.tools.report_tool import ReportTool
from app.agent.tools.shap_tool import SHAPTool

ALL_TOOLS: tuple[type[BaseTool], ...] = (
    DatasetTool,
    EDATool,
    MLTool,
    SHAPTool,
    ReportTool,
)

__all__ = [
    "ALL_TOOLS",
    "BaseTool",
    "DatasetTool",
    "EDATool",
    "MLTool",
    "ReportTool",
    "SHAPTool",
    "ToolContext",
    "ToolError",
    "ToolResult",
]
