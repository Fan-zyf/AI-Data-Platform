"""v0.5 机器学习 API 路由。

- POST   /datasets/{dataset_id}/ml/train                          （训练 + CV + 评估）
- GET    /datasets/{dataset_id}/ml/experiments                    （实验列表）
- GET    /datasets/{dataset_id}/ml/experiments/{experiment_id}    （实验详情）
- DELETE /datasets/{dataset_id}/ml/experiments/{experiment_id}    （删除实验）
- POST   /datasets/{dataset_id}/ml/experiments/{experiment_id}/predict（批量预测）

防泄漏铁律由 ml_service 保证：统计型预处理只在训练集 Pipeline 内 fit，
Test Set 只参与最后一次评估；错误统一返回 {success:false, error:{code,message}}。
"""

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.models.ml import MLExperimentListItem, MLPredictRequest, MLTrainRequest
from app.services import ml_service
from app.services.data_service import DataServiceError
from app.services.dataset_manager import DatasetSessionError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ml"])


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"success": False, "error": {"code": code, "message": message}},
    )


def _handle(exc: Exception, dataset_id: str) -> JSONResponse:
    if isinstance(exc, (DatasetSessionError, DataServiceError)):
        logger.warning(
            "ML request failed for %s: [%s] %s", dataset_id, exc.code, exc.message
        )
        return _error_response(exc.status_code, exc.code, exc.message)
    logger.exception("unexpected ML error for dataset %s", dataset_id)  # pragma: no cover
    return _error_response(  # pragma: no cover
        500, "internal_error", "机器学习任务执行失败，请稍后重试。"
    )


@router.post(
    "/datasets/{dataset_id}/ml/train",
    summary="训练机器学习模型（train/test split + 训练集内 CV）",
    description=(
        "选择源版本（缺省 original）与目标字段后执行一次完整训练："
        "先在训练集内做预处理 Pipeline + 交叉验证比较候选模型，按主指标选最佳模型，"
        "再在完整训练集 refit 并在保留的测试集上做唯一一次评估。"
        "统计型预处理绝不提前在全量数据上 fit，以防数据泄漏。"
    ),
)
def train_model(dataset_id: str, request: MLTrainRequest):
    try:
        return ml_service.run_training(dataset_id, request)
    except Exception as exc:  # noqa: BLE001 - 统一转结构化错误
        return _handle(exc, dataset_id)


@router.get(
    "/datasets/{dataset_id}/ml/experiments",
    summary="机器学习实验列表",
    description="按创建时间倒序返回该数据集下的全部训练实验摘要。",
)
def list_ml_experiments(dataset_id: str):
    try:
        items = ml_service.list_experiments(dataset_id)
        return {
            "success": True,
            "dataset_id": dataset_id,
            "count": len(items),
            "experiments": [
                item if isinstance(item, dict) else MLExperimentListItem(**item).model_dump()
                for item in items
            ],
        }
    except Exception as exc:  # noqa: BLE001
        return _handle(exc, dataset_id)


@router.get(
    "/datasets/{dataset_id}/ml/experiments/{experiment_id}",
    summary="机器学习实验详情",
    description="返回实验完整元数据、训练集内 CV 对比与最终测试评估结果。",
)
def get_ml_experiment_detail(dataset_id: str, experiment_id: str):
    try:
        detail = ml_service.get_experiment_detail(dataset_id, experiment_id)
        detail["success"] = True
        return detail
    except Exception as exc:  # noqa: BLE001
        return _handle(exc, dataset_id)


@router.delete(
    "/datasets/{dataset_id}/ml/experiments/{experiment_id}",
    summary="删除机器学习实验",
    description="删除实验目录（含模型文件、元数据与评估结果）。删除实验后方可删除其引用的数据版本。",
)
def delete_ml_experiment(dataset_id: str, experiment_id: str) -> JSONResponse:
    try:
        ml_service.delete_experiment(dataset_id, experiment_id)
        return JSONResponse(
            content={"success": True, "message": f"实验 {experiment_id} 已删除。"}
        )
    except Exception as exc:  # noqa: BLE001
        return _handle(exc, dataset_id)


@router.post(
    "/datasets/{dataset_id}/ml/experiments/{experiment_id}/predict",
    summary="基于已训练实验批量预测",
    description=(
        "提交最多 1000 条 JSON 记录；每条记录需包含训练时的全部特征字段"
        "（顺序不限），多余字段会被忽略并给出 warning。模型不会再被重新训练或 fit。"
    ),
)
def predict_records(dataset_id: str, experiment_id: str, request: MLPredictRequest):
    try:
        return ml_service.run_prediction(dataset_id, experiment_id, request)
    except Exception as exc:  # noqa: BLE001
        return _handle(exc, dataset_id)
