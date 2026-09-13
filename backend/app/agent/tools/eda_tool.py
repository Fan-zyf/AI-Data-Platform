"""v0.7 EDA Tool：包装 v0.2/v0.4 EDA 服务。

读取 dataset_id 关联的 Dataset Session（默认 source 版本），
调用 :func:`eda_service.run_eda` 并把结果裁剪为 agent 友好的摘要结构。
"""

from __future__ import annotations

import logging
from typing import Any

from app.agent.tools.base import BaseTool, ToolContext, ToolError, ToolResult
from app.services import eda_service, version_manager

logger = logging.getLogger(__name__)


class EDATool(BaseTool):
    name = "eda"
    description = "数据集自动 EDA：字段类型、缺失率、数值统计、分类分布、异常值、相关性与规则型洞察。"
    keywords = (
        "数据", "特征", "字段", "列", "类型", "缺失", "异常", "分布", "相关",
        "eda", "missing", "data", "column", "field", "feature",
    )

    def execute(self, context: ToolContext) -> ToolResult:
        try:
            eda_response = eda_service.run_eda(
                context.dataset_id, version_id=context.source_version_id
            )
        except Exception as exc:  # noqa: BLE001
            raise ToolError("eda_failed", f"EDA 执行失败：{exc}", 500) from exc
        eda_payload = eda_response.model_dump() if hasattr(eda_response, "model_dump") else dict(eda_response)

        dataset_info = eda_payload.get("dataset") or {}
        missing_block = eda_payload.get("missing_analysis") or {}
        missing_top_all = missing_block.get("by_column") or []
        numeric_stats = eda_payload.get("numeric_summaries") or []
        categorical = eda_payload.get("categorical_summaries") or []
        correlation = eda_payload.get("correlation") or {}
        summary_block = eda_payload.get("summary") or {}
        warnings = summary_block.get("rule_based_insights") or []

        top_missing = []
        for col in missing_top_all:
            if not isinstance(col, dict):
                continue
            try:
                mp = float(col.get("missing_percentage") or 0.0)
            except (TypeError, ValueError):
                mp = 0.0
            if mp > 0:
                top_missing.append({
                    "column": col.get("column"),
                    "missing_rate": mp / 100.0 if mp > 1 else mp,
                    "missing_count": col.get("missing_count"),
                })
        top_missing.sort(key=lambda x: x["missing_rate"], reverse=True)

        data: dict[str, Any] = {
            "rows": dataset_info.get("rows") or eda_payload.get("n_rows"),
            "columns_count": dataset_info.get("columns") or eda_payload.get("n_columns"),
            "missing": {
                "overall_missing_percentage": missing_block.get("overall_missing_percentage"),
                "total_missing": missing_block.get("total_missing"),
                "high_missing_columns": summary_block.get("high_missing_columns"),
            },
            "missing_top": top_missing[:5],
            "numeric_columns": summary_block.get("numeric_columns") or len(numeric_stats),
            "categorical_columns": summary_block.get("categorical_columns") or len(categorical),
            "outlier_columns": summary_block.get("columns_with_outliers"),
            "strong_correlations": summary_block.get("strong_correlations"),
            "correlation": _trim_correlation(correlation),
            "warnings": warnings,
        }
        summary_text = _build_summary(data, dataset_info)
        return ToolResult(name=self.name, status="ok", data=data, summary=summary_text)


def _trim_correlation(corr: dict) -> dict:
    """对 Pearson 相关矩阵做对称剪裁，仅保留绝对值 top-5。"""
    if not isinstance(corr, dict):
        return {}
    pairs: list[dict] = []
    seen = set()
    for a, row in corr.items():
        if not isinstance(row, dict):
            continue
        for b, v in row.items():
            key = tuple(sorted((a, b)))
            if a == b or key in seen:
                continue
            seen.add(key)
            try:
                num = float(v)
            except (TypeError, ValueError):
                continue
            if num != num:  # NaN
                continue
            pairs.append({"a": a, "b": b, "value": num})
    pairs.sort(key=lambda p: abs(p["value"]), reverse=True)
    return {"top_pairs": pairs[:5]}


def _build_summary(data: dict, dataset_info: dict) -> str:
    bits: list[str] = []
    rows = data.get("rows")
    cc = data.get("columns_count")
    if rows is not None and cc is not None:
        bits.append(f"数据集 {rows} 行 × {cc} 列")
    warnings = data.get("warnings") or []
    if warnings:
        bits.append(f"质量警告 {len(warnings)} 条")
    top_missing = data.get("missing_top") or []
    if top_missing:
        first = top_missing[0]
        bits.append(f"最高缺失 {first['column']}({first['missing_rate']:.1%})")
    return "；".join(bits) if bits else "EDA 摘要为空"


# 解决 unused import（version_manager 用于校验存在性，但由 eda_service 内部处理）
_ = version_manager
