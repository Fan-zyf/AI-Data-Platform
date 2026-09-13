"""v0.7 SHAP Tool：复用 v0.6 SHAP 服务，把解释结果裁剪为 agent 友好结构。

支持两种模式：
- ``cache``：experiment 已有 ``shap_result.json`` → 直接读取；
- ``compute``：未生成 SHAP → 现场调用 :func:`shap_service.run_explain` 计算。
"""

from __future__ import annotations

import logging
from typing import Any

from app.agent.tools.base import BaseTool, ToolContext, ToolError, ToolResult
from app.services import experiment_manager as em
from app.services import shap_service

logger = logging.getLogger(__name__)


class SHAPTool(BaseTool):
    name = "shap"
    description = "SHAP 模型解释：全局特征重要性、Summary 贡献分布、若干单点预测解释。"
    keywords = (
        "解释", "贡献", "重要性", "shap", "explain", "特征影响", "为什么",
        "important", "contribution", "feature importance",
    )
    requires_experiment = True

    def execute(self, context: ToolContext) -> ToolResult:
        if not context.experiment_id:
            return ToolResult(
                name=self.name,
                status="skipped",
                summary="未指定 experiment_id，跳过 SHAP 解释读取。",
            )
        try:
            location = em.find_experiment_location(context.experiment_id)
        except em.ExperimentError as exc:
            raise ToolError(exc.code, exc.message, exc.status_code) from exc
        if not location:
            raise ToolError("experiment_not_found", "实验不存在或已被删除。", 404)
        _, exp_dir = location

        payload: dict | None = None
        is_cache_hit = False
        if not context.regenerate_shap:
            payload = em.read_shap_result(exp_dir)
            is_cache_hit = payload is not None
        if payload is None:
            try:
                from app.models.explainability import ExplainRequest

                request = ExplainRequest(
                    max_summary_rows=context.max_summary_rows or 120,
                    regenerate=context.regenerate_shap,
                )
                payload = shap_service.run_explain(context.experiment_id, request)
            except shap_service.ExplainabilityError as exc:
                # 缓存命中 / 数据缺失 等情况允许 tool 降级为 skipped
                if exc.status_code in (404, 422):
                    return ToolResult(
                        name=self.name,
                        status="skipped",
                        error=f"[{exc.code}] {exc.message}",
                        summary="SHAP 暂不可用：源数据缺失。",
                    )
                raise ToolError(exc.code, exc.message, exc.status_code) from exc
            except Exception as exc:  # noqa: BLE001
                raise ToolError("shap_failed", f"SHAP 计算失败：{exc}", 500) from exc

        # 缓存命中：把 payload.cached 强制置为 True，便于 Agent 上下文传达
        if is_cache_hit:
            payload = dict(payload)
            payload["cached"] = True

        data = _compress(payload)
        summary = _build_summary(data)
        return ToolResult(name=self.name, status="ok", data=data, summary=summary)


def _compress(payload: dict) -> dict[str, Any]:
    global_block = payload.get("global") or {}
    importance = list(global_block.get("importance") or [])
    feature_names = list(global_block.get("feature_names") or [])
    pairs = sorted(
        [
            {"feature": n, "importance": float(imp or 0.0)}
            for n, imp in zip(feature_names, importance)
        ],
        key=lambda x: x["importance"],
        reverse=True,
    )
    samples = payload.get("samples") or []
    sample_pick = samples[0] if samples else {}
    sample_explanation = {
        "index": sample_pick.get("index"),
        "prediction": sample_pick.get("prediction"),
        "contributions_top": (sample_pick.get("top_contributions") or [])[:5],
    }
    return {
        "explainer_type": payload.get("explainer_type"),
        "model_class": payload.get("model_class"),
        "task_type": payload.get("task_type"),
        "class_label_used": payload.get("class_label_used"),
        "base_value": payload.get("base_value"),
        "n_features_post_preprocessing": payload.get("n_features_post_preprocessing"),
        "n_rows_used": global_block.get("n_rows_used"),
        "top_features": [p["feature"] for p in pairs[:5]],
        "top_pairs": pairs[:5],
        "sample_explanation": sample_explanation,
        "warnings": payload.get("warnings") or [],
        "cached": payload.get("cached", False),
        "created_at": payload.get("created_at"),
    }


def _build_summary(data: dict) -> str:
    bits: list[str] = []
    if data.get("explainer_type"):
        label = {
            "tree": "Tree",
            "linear": "Linear",
            "kernel": "Kernel",
        }.get(data["explainer_type"], data["explainer_type"])
        bits.append(f"解释器 {label}")
    top = data.get("top_features") or []
    if top:
        bits.append("Top 特征：" + "、".join(map(str, top[:3])))
    if data.get("cached"):
        bits.append("（缓存）")
    return "；".join(bits) if bits else "SHAP 摘要为空"
