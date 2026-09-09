"""数据处理与数据版本管理 API 路由。

- POST   /datasets/{dataset_id}/processing/preview   （内存 Preview，不落盘）
- POST   /datasets/{dataset_id}/processing/apply     （全部成功才生成新版本）
- GET    /datasets/{dataset_id}/versions             （版本列表）
- GET    /datasets/{dataset_id}/versions/{version_id}（版本详情）
- POST   /datasets/{dataset_id}/versions/compare     （版本对比）
- DELETE /datasets/{dataset_id}/versions/{version_id}（删除派生版本）

所有错误统一为 {success:false, error:{code,message}}，不返回 traceback。
"""

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.models.processing import (
    ORIGINAL_VERSION,
    ProcessingApplyResponse,
    ProcessingPreviewResponse,
    TransformationPlan,
    VersionCompareRequest,
    VersionCompareResponse,
    VersionDetailResponse,
    VersionListItem,
    VersionListResponse,
)
from app.services import processing_service as service
from app.services import version_manager as vm
from app.services.data_service import DataServiceError
from app.services.dataset_manager import DatasetSessionError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["processing"])


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"success": False, "error": {"code": code, "message": message}},
    )


def _handle(exc: Exception, dataset_id: str) -> JSONResponse:
    if isinstance(exc, (DatasetSessionError, DataServiceError)):
        logger.warning("processing failed for %s: [%s] %s", dataset_id, exc.code, exc.message)
        return _error_response(exc.status_code, exc.code, exc.message)
    logger.exception("unexpected processing error for dataset %s", dataset_id)  # pragma: no cover
    return _error_response(500, "internal_error", "数据处理失败，请稍后重试。")  # pragma: no cover


@router.post(
    "/datasets/{dataset_id}/processing/preview",
    response_model=ProcessingPreviewResponse,
    summary="Preview 数据处理（内存执行，不产生版本）",
    description=(
        "提交 source_version_id（缺省 original）与按顺序执行的 operations，"
        "仅在内存执行一次并返回 Before / After 指标与预览，不写入任何新版本。"
    ),
)
def preview_processing(dataset_id: str, plan: TransformationPlan) -> ProcessingPreviewResponse:
    try:
        return service.preview_plan(dataset_id, plan)
    except Exception as exc:  # noqa: BLE001 - 统一转结构化错误
        return _handle(exc, dataset_id)


@router.post(
    "/datasets/{dataset_id}/processing/apply",
    response_model=ProcessingApplyResponse,
    summary="Apply 数据处理并创建新版本",
    description=(
        "按顺序执行 Transformation Plan；只有全部步骤成功后才生成新的派生版本"
        "（UUID），任何一步失败都不会保存半成品版本。原始数据永不修改。"
    ),
)
def apply_processing(dataset_id: str, plan: TransformationPlan) -> ProcessingApplyResponse:
    try:
        return service.apply_plan(dataset_id, plan)
    except Exception as exc:  # noqa: BLE001 - 统一转结构化错误
        return _handle(exc, dataset_id)


@router.get(
    "/datasets/{dataset_id}/versions",
    response_model=VersionListResponse,
    summary="数据版本列表",
    description="返回 original 与全部派生版本（按创建时间升序），每项含行/列/缺失/重复数与操作摘要。",
)
def list_dataset_versions(dataset_id: str) -> VersionListResponse:
    try:
        original, items = vm.list_versions(dataset_id)
        return VersionListResponse(
            success=True,
            dataset_id=dataset_id,
            original=VersionListItem(**original),
            versions=[VersionListItem(**item) for item in items],
        )
    except Exception as exc:  # noqa: BLE001
        return _handle(exc, dataset_id)


@router.post(
    "/datasets/{dataset_id}/versions/compare",
    response_model=VersionCompareResponse,
    summary="版本对比",
    description="对比 Version A 与 Version B 的基础指标与列差异（A→B 方向）。",
)
def compare_versions(dataset_id: str, request: VersionCompareRequest) -> VersionCompareResponse:
    try:
        result = vm.compare_versions(dataset_id, request.version_a, request.version_b)
        return VersionCompareResponse(**result)
    except Exception as exc:  # noqa: BLE001
        return _handle(exc, dataset_id)


@router.get(
    "/datasets/{dataset_id}/versions/{version_id}",
    response_model=VersionDetailResponse,
    summary="数据版本详情",
    description=(
        "返回版本的元信息、质量概览、列画像、前 20 行预览与操作历史；"
        f"version_id 可为 {ORIGINAL_VERSION} 或派生版本 UUID。"
    ),
)
def get_version_detail(dataset_id: str, version_id: str) -> VersionDetailResponse:
    try:
        return VersionDetailResponse(**vm.version_detail(dataset_id, version_id))
    except Exception as exc:  # noqa: BLE001
        return _handle(exc, dataset_id)


@router.delete(
    "/datasets/{dataset_id}/versions/{version_id}",
    summary="删除派生数据版本",
    description=(
        "只允许删除派生版本；original 永远不能删除；"
        "若该版本存在子版本，返回 409 冲突，不会删除。"
    ),
)
def delete_dataset_version(dataset_id: str, version_id: str) -> JSONResponse:
    try:
        vm.delete_derived_version(dataset_id, version_id)
        return JSONResponse(
            content={"success": True, "message": f"数据版本 {version_id} 已删除。"}
        )
    except Exception as exc:  # noqa: BLE001
        return _handle(exc, dataset_id)
