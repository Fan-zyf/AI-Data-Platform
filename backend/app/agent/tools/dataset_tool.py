"""v0.7 Dataset Tool：读取 Dataset Session 元信息与版本列表。"""

from __future__ import annotations

import logging
from typing import Any

from app.agent.tools.base import BaseTool, ToolContext, ToolError, ToolResult
from app.services import dataset_manager, version_manager

logger = logging.getLogger(__name__)


class DatasetTool(BaseTool):
    name = "dataset"
    description = "数据集元信息：行数、列数、原始文件名、版本列表、是否过期。"
    keywords = (
        "数据", "数据集", "元信息", "基本信息", "概况", "大小", "版本", "样本",
        "dataset", "overview", "info",
    )

    def execute(self, context: ToolContext) -> ToolResult:
        try:
            session = dataset_manager.load_session(context.dataset_id)
        except Exception as exc:  # noqa: BLE001
            raise ToolError("dataset_not_found", f"数据集不存在或已过期：{exc}", 404) from exc
        try:
            versions = version_manager.list_versions(context.dataset_id)
        except Exception:  # noqa: BLE001 - 列表失败时退化为空数组
            versions = []

        data: dict[str, Any] = {
            "dataset_id": context.dataset_id,
            "name": getattr(session, "name", "") or "",
            "rows": getattr(session, "rows", None),
            "n_columns": getattr(session, "columns", None),
            "columns": getattr(session, "column_names", []) or [],
            "source_filename": getattr(session, "original_filename", None),
            "source_format": getattr(session, "file_type", None),
            "created_at": _iso(getattr(session, "created_at", None)),
            "expires_at": _iso(getattr(session, "expires_at", None)),
            "version_count": len(versions),
            "versions": [
                _version_to_dict(v) for v in versions
            ],
            "quality": {},
        }
        rows = data["rows"]
        cols = data["n_columns"]
        fmt = data["source_format"]
        summary = (
            f"数据集 {data['name'] or context.dataset_id[:8]}：{rows} 行 × {cols} 列 "
            f"（{fmt or 'csv'}），共 {data['version_count']} 个版本"
        )
        return ToolResult(name=self.name, status="ok", data=data, summary=summary)


def _iso(value):
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _version_to_dict(v) -> dict:
    if isinstance(v, dict):
        return {
            "version_id": v.get("version_id"),
            "parent": v.get("parent_version_id", "original"),
            "operations": v.get("operations") or [],
            "created_at": _iso(v.get("created_at")),
        }
    return {
        "version_id": getattr(v, "version_id", None),
        "parent": getattr(v, "parent_version_id", "original"),
        "operations": getattr(v, "operations", None) or [],
        "created_at": _iso(getattr(v, "created_at", None)),
    }
