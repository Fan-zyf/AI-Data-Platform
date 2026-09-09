"""数据上传与分析相关路由。

路由只负责 HTTP 传输与错误格式转换，真正的解析/画像逻辑在 services.data_service 中。
"""

import logging
import os

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.models.data import DataUploadResponse
from app.services.data_service import DataServiceError, analyze_upload_file

logger = logging.getLogger(__name__)

router = APIRouter(tags=["data"])

_CHUNK_SIZE = 1024 * 1024  # 1MB，配合上限控制读取内存


async def _read_with_limit(file: UploadFile, max_bytes: int) -> bytes:
    """分块读取上传内容，超过上限立即中止并抛出友好错误。"""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(_CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise DataServiceError(
                "file_too_large",
                f"文件大小超过限制（最大 {settings.data_max_upload_mb}MB），"
                f"请压缩或抽样后上传。",
                status_code=413,
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    """构造统一的结构化错误响应，绝不向客户端暴露 traceback。"""
    return JSONResponse(
        status_code=status_code,
        content={"success": False, "error": {"code": code, "message": message}},
    )


@router.post(
    "/data/upload",
    response_model=DataUploadResponse,
    summary="上传 CSV/Excel 并返回基础数据画像",
)
async def upload_data(file: UploadFile = File(...)):
    """接收一个 .csv / .xlsx / .xls 文件，返回数据画像、质量概览与前 20 行预览。"""
    raw_name = file.filename or ""
    # 兼容不同浏览器对路径的处理，只保留文件名部分
    file_name = os.path.basename(raw_name.replace("\\", "/"))

    try:
        content = await _read_with_limit(file, settings.data_max_upload_bytes)
        result = analyze_upload_file(file_name, content)
        return result
    except DataServiceError as exc:
        logger.warning("upload rejected: code=%s, file=%s", exc.code, file_name)
        return _error_response(exc.status_code, exc.code, exc.message)
    except Exception:
        logger.exception("unexpected error while analyzing upload: %s", file_name)
        return _error_response(
            500,
            "internal_error",
            "服务器处理文件时发生内部错误，请稍后重试。",
        )
    finally:
        await file.close()
