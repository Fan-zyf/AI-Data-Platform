"""数据模块的 Pydantic 模型：数据上传与画像响应结构。

所有接口返回结构都通过这些模型定义，保证前端拿到稳定的 JSON 结构。
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

# ---- 字段推断类型常量 ----
NUMERIC = "numeric"
CATEGORICAL = "categorical"
DATETIME = "datetime"
BOOLEAN = "boolean"
TEXT = "text"

INFERRED_TYPES = (NUMERIC, CATEGORICAL, DATETIME, BOOLEAN, TEXT)


class DatasetInfo(BaseModel):
    """数据集基础信息。"""

    file_name: str = Field(..., description="上传的原始文件名")
    file_type: str = Field(..., description="文件类型，如 csv / xlsx / xls")
    file_size: int = Field(..., description="文件大小（字节）")
    rows: int = Field(..., description="数据总行数")
    columns: int = Field(..., description="数据总列数")


class QualityOverview(BaseModel):
    """数据质量概览。"""

    total_missing: int = Field(0, description="缺失值总数")
    columns_with_missing: int = Field(0, description="包含缺失值的列数")
    duplicate_rows: int = Field(0, description="重复数据行数")
    empty_columns: int = Field(0, description="完全为空的列数")
    constant_columns: int = Field(0, description="取值单一的常量列数")


class ColumnProfile(BaseModel):
    """单列的画像信息。"""

    column_name: str = Field(..., description="字段名")
    dtype: str = Field(..., description="pandas 原始 dtype 名称")
    inferred_type: str = Field(..., description="业务推断类型：numeric/categorical/datetime/boolean/text")
    non_null_count: int = Field(..., description="非空值数量")
    missing_count: int = Field(..., description="缺失值数量")
    missing_percentage: float = Field(..., description="缺失比例（0-100）")
    unique_count: int = Field(..., description="去重后的不同值数量（不含缺失）")


class UploadError(BaseModel):
    """错误信息结构。"""

    code: str = Field(..., description="机器可读错误码")
    message: str = Field(..., description="面向用户的友好错误提示")


class ErrorResponse(BaseModel):
    """统一错误响应结构。"""

    success: bool = False
    error: UploadError


class DatasetDeleteResponse(BaseModel):
    """DELETE /api/datasets/{dataset_id} 的成功响应结构。"""

    success: bool = True
    message: str = Field(..., description="删除结果提示")


class DataUploadResponse(BaseModel):
    """POST /api/data/upload 的成功响应结构。

    成功上传后会创建临时 Dataset Session，因此额外携带 dataset_id 与会话生命周期。
    """

    success: bool = True
    dataset_id: str | None = Field(
        None, description="临时数据集会话 ID（UUID）；创建失败时为 null"
    )
    created_at: datetime | None = Field(None, description="会话创建时间（UTC ISO8601）")
    expires_at: datetime | None = Field(None, description="会话过期时间（UTC ISO8601）")
    dataset: DatasetInfo
    quality: QualityOverview
    column_profiles: list[ColumnProfile] = Field(default_factory=list)
    preview: list[dict[str, Any]] = Field(default_factory=list, description="前 N 行数据预览")
    warnings: list[str] = Field(default_factory=list, description="数据质量提示信息")
