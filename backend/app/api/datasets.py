"""Dataset Session 相关 API 路由：自动 EDA 与会话删除。"""

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.models.data import DatasetDeleteResponse
from app.models.eda import EDAResponse
from app.services import eda_service
from app.services.data_service import DataServiceError
from app.services.dataset_manager import DatasetSessionError, delete_session

logger = logging.getLogger(__name__)

router = APIRouter(tags=["datasets"])


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    """与上传接口保持一致的统一错误响应结构。"""
    return JSONResponse(
        status_code=status_code,
        content={"success": False, "error": {"code": code, "message": message}},
    )


@router.get(
    "/datasets/{dataset_id}/eda",
    response_model=EDAResponse,
    summary="数据集自动 EDA",
    description="基于 dataset_id 加载临时会话数据，执行自动 EDA 并返回结构化统计 JSON。",
)
def get_dataset_eda(dataset_id: str):
    """GET /api/datasets/{dataset_id}/eda"""
    try:
        return eda_service.run_eda(dataset_id)
    except (DatasetSessionError, DataServiceError) as exc:
        logger.warning("EDA failed for %s: [%s] %s", dataset_id, exc.code, exc.message)
        return _error_response(exc.status_code, exc.code, exc.message)
    except Exception:  # pragma: no cover - 兜底，避免向用户泄露 traceback
        logger.exception("unexpected EDA error for dataset %s", dataset_id)
        return _error_response(500, "internal_error", "EDA 计算失败，请稍后重试。")


@router.delete(
    "/datasets/{dataset_id}",
    response_model=DatasetDeleteResponse,
    summary="删除数据集会话",
    description="删除临时 Dataset Session（含 source 文件与 metadata）。",
)
def delete_dataset(dataset_id: str):
    """DELETE /api/datasets/{dataset_id}"""
    try:
        delete_session(dataset_id)
        return DatasetDeleteResponse(message="Dataset session deleted")
    except DatasetSessionError as exc:
        return _error_response(exc.status_code, exc.code, exc.message)
    except Exception:  # pragma: no cover - 兜底
        logger.exception("unexpected delete error for dataset %s", dataset_id)
        return _error_response(500, "internal_error", "数据集会话清理失败，请稍后重试。")
