"""v0.7 Agent 业务服务层。

负责：
- 校验 dataset_id / experiment_id 存在性（在交给 agent 之前）；
- 捕获 ``AgentError`` 转换为结构化响应；
- 把 agent 结果二次封装为 API 期望的字段。
"""

from __future__ import annotations

import logging
from typing import Any

from app.agent.analyst_agent import AgentError, run_analyst
from app.services import dataset_manager, experiment_manager as em

logger = logging.getLogger(__name__)


def analyze(
    question: str,
    dataset_id: str,
    experiment_id: str | None = None,
    source_version_id: str = "original",
    max_summary_rows: int | None = None,
    regenerate_shap: bool = False,
) -> dict[str, Any]:
    # 校验 dataset 存在（防止 agent 在无效 dataset 上浪费工具调用）
    try:
        dataset_manager.load_session(dataset_id)
    except Exception as exc:  # noqa: BLE001
        raise AgentError("dataset_not_found", f"数据集不存在或已过期：{exc}", 404) from exc
    # 校验 experiment（如有）
    if experiment_id:
        try:
            location = em.find_experiment_location(experiment_id)
        except em.ExperimentError as exc:
            raise AgentError(exc.code, exc.message, exc.status_code) from exc
        if not location:
            raise AgentError("experiment_not_found", "实验不存在或已被删除。", 404)
    return run_analyst(
        question=question,
        dataset_id=dataset_id,
        experiment_id=experiment_id,
        source_version_id=source_version_id,
        max_summary_rows=max_summary_rows,
        regenerate_shap=regenerate_shap,
    )
