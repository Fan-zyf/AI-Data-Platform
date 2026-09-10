"""v0.6 模型可解释性 API 路由。

- POST /api/ml/experiments/{experiment_id}/explain
  （按用户指定路径；服务端通过扫描 runtime/datasets 定位实验所属 dataset）

- POST /api/datasets/{dataset_id}/ml/experiments/{experiment_id}/explain
  （与 v0.5 路由风格保持一致的 dataset-scoped 入口）

两个入口共享同一 service，结构化错误统一返回
``{success:false, error:{code, message}}``。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.models.explainability import ExplainRequest
from app.services import shap_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["explainability"])


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"success": False, "error": {"code": code, "message": message}},
    )


def _handle(exc: Exception) -> JSONResponse:
    if isinstance(exc, shap_service.ExplainabilityError):
        logger.warning(
            "explainability error: [%s] %s", exc.code, exc.message
        )
        return _error_response(exc.status_code, exc.code, exc.message)
    logger.exception("unexpected explainability error")  # pragma: no cover
    return _error_response(  # pragma: no cover
        500, "internal_error", "模型可解释性任务执行失败，请稍后重试。"
    )


@router.post(
    "/ml/experiments/{experiment_id}/explain",
    summary="基于已训练实验生成 SHAP 模型解释",
    description=(
        "读取实验持久化的完整 Pipeline（preprocessing + estimator），"
        "根据模型类型自动选择 TreeExplainer / LinearExplainer / KernelExplainer，"
        "在源版本数据（采样）上计算全局特征重要性、SHAP summary 与单点解释。"
        "结果同时落盘到 ``runtime/datasets/<id>/ml/experiments/<exp_id>/shap_result.json``。"
    ),
)
def explain_experiment(experiment_id: str, request: ExplainRequest | None = None):
    try:
        return shap_service.run_explain(
            experiment_id, request or ExplainRequest()
        )
    except Exception as exc:  # noqa: BLE001
        return _handle(exc)


@router.post(
    "/datasets/{dataset_id}/ml/experiments/{experiment_id}/explain",
    summary="（dataset-scoped 同义入口）",
    description="与 ``/api/ml/experiments/{experiment_id}/explain`` 等价；通过 dataset_id 提前校验归属。",
)
def explain_experiment_in_dataset(
    dataset_id: str, experiment_id: str, request: ExplainRequest | None = None
):
    try:
        from app.services import experiment_manager as em

        location = em.find_experiment_location(experiment_id)
        if not location:
            raise shap_service.ExplainabilityError(
                "experiment_not_found",
                "实验不存在或已被删除，请重新训练后再生成解释。",
                404,
            )
        found_dataset_id, _ = location
        if found_dataset_id != dataset_id:
            raise shap_service.ExplainabilityError(
                "experiment_dataset_mismatch",
                f"实验不属于数据集 {dataset_id}，请使用对应的 dataset_id 访问。",
                400,
            )
        return shap_service.run_explain(
            experiment_id, request or ExplainRequest()
        )
    except Exception as exc:  # noqa: BLE001
        return _handle(exc)
