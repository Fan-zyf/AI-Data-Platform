"""v0.7 ML Tool：读取 experiment 详情、CV 表现与测试集指标。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.agent.tools.base import BaseTool, ToolContext, ToolError, ToolResult
from app.services import experiment_manager as em

logger = logging.getLogger(__name__)


class MLTool(BaseTool):
    name = "ml"
    description = "机器学习实验：最佳模型、候选 CV 对比、测试集指标（准确率 / AUC / 回归指标等）。"
    keywords = (
        "模型", "评估", "指标", "训练", "auc", "准确率", "f1", "召回", "精确",
        "ml", "model", "metric", "train", "accuracy", "precision", "recall",
    )
    requires_experiment = True

    def execute(self, context: ToolContext) -> ToolResult:
        if not context.experiment_id:
            return ToolResult(
                name=self.name,
                status="skipped",
                summary="未指定 experiment_id，跳过模型评估读取。",
            )
        try:
            location = em.find_experiment_location(context.experiment_id)
        except em.ExperimentError as exc:
            raise ToolError(exc.code, exc.message, exc.status_code) from exc
        if not location:
            raise ToolError(
                "experiment_not_found",
                "实验不存在或已被删除。",
                404,
            )
        dataset_id, exp_dir = location
        try:
            metadata = em.get_experiment(dataset_id, context.experiment_id)
        except Exception as exc:  # noqa: BLE001
            raise ToolError("experiment_invalid", f"读取实验元信息失败：{exc}", 500) from exc

        evaluation: dict = {}
        eval_path = Path(exp_dir) / em.EVALUATION_FILE
        if eval_path.is_file():
            try:
                evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ToolError("experiment_invalid", f"evaluation.json 损坏：{exc}", 500) from exc

        # v0.5 真实 evaluation.json 结构：{ cv: { results: { model_name: { folds, metrics, ... } } }, test: {...} }
        cv_block = (evaluation or {}).get("cv") or {}
        test_block = (evaluation or {}).get("test") or {}
        test_metrics = (test_block or {}).get("metrics") or {}
        cv_results = (cv_block or {}).get("results") or {}
        primary_metric = (cv_block or {}).get("primary_metric") or ""

        # 选 CV 表的 top-5 候选供 LLM 参考（cv_results 是 dict[model_name, metrics]）
        cv_top = []
        if isinstance(cv_results, dict):
            for model_name, payload in cv_results.items():
                if not isinstance(payload, dict):
                    continue
                metrics_block = payload.get("metrics") or {}
                # 从 primary_metric 或第一项指标中取出 mean / std
                primary_value = None
                primary_std = None
                if primary_metric and primary_metric in metrics_block:
                    primary_value = metrics_block[primary_metric].get("mean")
                    primary_std = metrics_block[primary_metric].get("std")
                elif metrics_block:
                    first_key = next(iter(metrics_block.keys()))
                    primary_value = metrics_block[first_key].get("mean")
                    primary_std = metrics_block[first_key].get("std")
                cv_top.append({
                    "model": model_name,
                    "score": primary_value,
                    "std": primary_std,
                    "metric": primary_metric,
                    "folds": payload.get("folds"),
                })
        elif isinstance(cv_results, list):
            # 兼容 list 形态
            for row in cv_results[:5]:
                if not isinstance(row, dict):
                    continue
                cv_top.append({
                    "model": row.get("model") or row.get("name"),
                    "score": row.get("primary_score") or row.get("score"),
                    "std": row.get("primary_std") or row.get("std"),
                    "metric": primary_metric,
                })

        data: dict[str, Any] = {
            "task_type": metadata.get("task_type"),
            "best_model": metadata.get("best_model"),
            "features": metadata.get("features") or [],
            "target_column": metadata.get("target_column"),
            "random_state": metadata.get("random_state"),
            "test_size": metadata.get("test_size"),
            "cv_folds": metadata.get("cv_folds"),
            "class_labels": metadata.get("class_labels") or [],
            "metrics": test_metrics,
            "test_metrics": test_metrics,
            "cv_table_top": cv_top,
            "explainability": metadata.get("explainability"),
            "experiment_id": context.experiment_id,
            "dataset_id": dataset_id,
        }
        summary = _build_ml_summary(data)
        return ToolResult(name=self.name, status="ok", data=data, summary=summary)


def _build_ml_summary(data: dict) -> str:
    bits: list[str] = []
    if data.get("task_type") == "classification":
        bits.append("分类")
    elif data.get("task_type") == "regression":
        bits.append("回归")
    bits.append(f"模型 {data.get('best_model') or '?'}")
    metrics = data.get("metrics") or {}
    bullets = []
    for k in ("accuracy", "f1", "auc", "rmse", "mae", "r2"):
        if k in metrics and metrics[k] is not None:
            try:
                bullets.append(f"{k}={float(metrics[k]):.4f}")
            except (TypeError, ValueError):
                pass
    if bullets:
        bits.append("测试集 " + " ".join(bullets[:4]))
    explainability = data.get("explainability") or {}
    if explainability.get("status") == "computed":
        bits.append("已生成 SHAP")
    return "；".join(bits) if bits else "ML 摘要为空"
