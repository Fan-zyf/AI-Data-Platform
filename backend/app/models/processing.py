"""数据清洗 / 特征工程 / 数据版本管理的 Pydantic 模型。

Transformation Plan 必须是结构化 JSON（operation 数组按顺序执行），
后端只允许执行预定义好的 operation，绝不接受 Python 代码 / eval / exec。

与 v0.3 一致：所有接口返回结构由 Pydantic 显式定义，保证前端拿到稳定的 JSON 结构。
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from app.models.data import ColumnProfile, QualityOverview

# 版本常量：原始版本使用固定标识
ORIGINAL_VERSION = "original"

# ---- 允许的 operation type ----
DROP_DUPLICATES = "drop_duplicates"
FILL_MISSING = "fill_missing"
DROP_COLUMNS = "drop_columns"
CONVERT_TYPE = "convert_type"
REMOVE_OUTLIERS = "remove_outliers"
TEXT_TRANSFORM = "text_transform"
DATE_FEATURES = "date_features"
ONE_HOT_ENCODE = "one_hot_encode"
SCALE_NUMERIC = "scale_numeric"

OPERATION_TYPES = (
    DROP_DUPLICATES,
    FILL_MISSING,
    DROP_COLUMNS,
    CONVERT_TYPE,
    REMOVE_OUTLIERS,
    TEXT_TRANSFORM,
    DATE_FEATURES,
    ONE_HOT_ENCODE,
    SCALE_NUMERIC,
)

# 允许的缺失值填充策略
FILL_STRATEGIES = ("mean", "median", "mode", "constant", "drop_rows", "drop_columns")
# 类型转换目标类型
CONVERT_TARGETS = ("string", "numeric", "datetime", "boolean")
# 异常值处理动作
OUTLIER_ACTIONS = ("clip", "remove_rows")
# 文本处理动作
TEXT_ACTIONS = ("strip", "lowercase", "uppercase")
# 日期特征
DATE_FEATURES_OPTIONS = ("year", "month", "day", "day_of_week", "quarter")
# 数值缩放方法
SCALE_METHODS = ("standardization", "min_max")


class TransformationOperation(BaseModel):
    """单步转换操作（type 决定语义，其余参数按需使用）。

    所有参数都在后端以普通数据处理执行；column 名从不拼装为 Python 表达式。
    """

    type: Literal[
        "drop_duplicates",
        "fill_missing",
        "drop_columns",
        "convert_type",
        "remove_outliers",
        "text_transform",
        "date_features",
        "one_hot_encode",
        "scale_numeric",
    ]
    columns: list[str] = Field(default_factory=list, description="需要处理的字段列表")
    column: str | None = Field(None, description="单个目标字段")
    strategy: Literal["mean", "median", "mode", "constant", "drop_rows", "drop_columns"] | None = Field(
        None, description="缺失值处理策略"
    )
    fill_value: Any = Field(None, description="constant 策略使用的填充值")
    target_type: Literal["string", "numeric", "datetime", "boolean"] | None = Field(
        None, description="类型转换目标类型"
    )
    method: str | None = Field(None, description="异常值检测方法（当前仅支持 IQR）")
    action: Literal["clip", "remove_rows", "strip", "lowercase", "uppercase"] | None = Field(
        None, description="异常值/文本处理动作"
    )
    features: list[Literal["year", "month", "day", "day_of_week", "quarter"]] = Field(
        default_factory=list, description="日期派生特征列表"
    )
    scale_method: Literal["standardization", "min_max"] | None = Field(
        None, description="数值缩放方法"
    )
    allow_high_cardinality: bool = Field(
        False, description="允许高基数分类字段执行 One-Hot（需用户显式确认）"
    )

    @model_validator(mode="after")
    def _validate_required_fields(self) -> "TransformationOperation":
        """按 operation 类型校验必需的参数字段，快速失败并返回清晰 422 提示。"""
        op_type = self.type
        missing: list[str] = []

        def check(cond: bool, name: str) -> None:
            if not cond:
                missing.append(name)

        if op_type == DROP_DUPLICATES:
            pass
        elif op_type in (FILL_MISSING, DROP_COLUMNS, ONE_HOT_ENCODE, SCALE_NUMERIC):
            check(bool(self.columns), "columns")
            if op_type == FILL_MISSING:
                check(self.strategy in FILL_STRATEGIES, "strategy(mean/median/mode/constant/drop_rows/drop_columns)")
            if op_type == SCALE_NUMERIC:
                check(self.scale_method in SCALE_METHODS, "scale_method(standardization/min_max)")
        elif op_type == CONVERT_TYPE:
            check(bool(self.column), "column")
            check(self.target_type in CONVERT_TARGETS, "target_type(string/numeric/datetime/boolean)")
        elif op_type == REMOVE_OUTLIERS:
            check(bool(self.column), "column")
            check(bool(self.method) and str(self.method).casefold() == "iqr", "method(IQR)")
            check(self.action in ("clip", "remove_rows"), "action(clip/remove_rows)")
        elif op_type == TEXT_TRANSFORM:
            check(bool(self.columns), "columns")
            check(self.action in TEXT_ACTIONS, "action(strip/lowercase/uppercase)")
        elif op_type == DATE_FEATURES:
            check(bool(self.column), "column")
            check(
                bool(self.features) and all(f in DATE_FEATURES_OPTIONS for f in self.features),
                "features(year/month/day/day_of_week/quarter)",
            )
        else:  # pragma: no cover - Literal 已限制
            raise ValueError(f"不支持的 operation 类型：{op_type}")

        if missing:
            raise ValueError(f"operation[{op_type}] 缺少必需参数：{'、'.join(missing)}")
        if self.strategy == "constant" and self.fill_value is None:
            raise ValueError("operation[fill_missing] 使用 constant 策略时必须提供 fill_value")
        return self


class TransformationPlan(BaseModel):
    """一次提交的 Transformation Plan（operation 严格按数组顺序执行）。"""

    source_version_id: str | None = Field(
        None, description="源数据版本；缺省时默认 original"
    )
    operations: list[TransformationOperation] = Field(
        min_length=1, description="按顺序执行的转换步骤"
    )


# ============================================================
# Preview / Apply 响应
# ============================================================


class ProcessingPreviewResponse(BaseModel):
    """POST /processing/preview 的响应：只在内存执行，绝不写入新版本。"""

    success: bool = True
    dataset_id: str
    source_version_id: str
    rows_before: int
    rows_after: int
    columns_before: int
    columns_after: int
    missing_before: int
    missing_after: int
    duplicates_before: int
    duplicates_after: int
    generated_columns: list[str] = Field(default_factory=list)
    removed_columns: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    preview_before: list[dict[str, Any]] = Field(default_factory=list)
    preview_after: list[dict[str, Any]] = Field(default_factory=list)


class ProcessingApplyResponse(BaseModel):
    """POST /processing/apply 的响应：全部步骤成功后才会持久化新版本。"""

    success: bool = True
    dataset_id: str
    version_id: str
    parent_version_id: str
    source_version_id: str
    created_at: datetime
    rows: int
    columns: int
    operations_applied: list[dict[str, Any]] = Field(default_factory=list)
    quality: QualityOverview
    warnings: list[str] = Field(default_factory=list)
    preview: list[dict[str, Any]] = Field(default_factory=list)


# ============================================================
# 版本管理
# ============================================================


class VersionListItem(BaseModel):
    """版本列表中的一项。"""

    version_id: str = Field(..., description="版本 ID；original 或合法 UUID")
    parent_version_id: str | None = Field(None, description="父版本 ID（original 为 None）")
    is_original: bool = False
    created_at: datetime | None = None
    rows: int = 0
    columns: int = 0
    missing: int = 0
    duplicates: int = 0
    operation_count: int = 0
    operations_summary: list[str] = Field(default_factory=list, description="简短的人类可读操作摘要")


class VersionListResponse(BaseModel):
    """GET /datasets/{dataset_id}/versions 响应。"""

    success: bool = True
    dataset_id: str
    original: VersionListItem
    versions: list[VersionListItem] = Field(default_factory=list, description="派生版本（按创建时间升序）")


class VersionDetailResponse(BaseModel):
    """GET /datasets/{dataset_id}/versions/{version_id} 响应。"""

    success: bool = True
    dataset_id: str
    version_id: str
    parent_version_id: str | None = None
    is_original: bool = False
    created_at: datetime | None = None
    rows: int
    columns: int
    file_name: str
    operations: list[dict[str, Any]] = Field(default_factory=list, description="该版本的完整操作历史")
    quality: QualityOverview
    column_profiles: list[ColumnProfile] = Field(default_factory=list)
    preview: list[dict[str, Any]] = Field(default_factory=list)


class VersionCompareRequest(BaseModel):
    """版本对比请求。"""

    version_a: str = Field(..., description="版本 A（original 或 UUID）")
    version_b: str = Field(..., description="版本 B（original 或 UUID）")


class VersionCompareResponse(BaseModel):
    """版本对比响应：基础指标 + 列差异，不包含复杂统计差异。"""

    success: bool = True
    dataset_id: str
    version_a: str
    version_b: str
    rows_a: int
    columns_a: int
    missing_a: int
    duplicates_a: int
    rows_b: int
    columns_b: int
    missing_b: int
    duplicates_b: int
    # 以 A -> B 的方向表达：B 中有而 A 没有的列 = generated；A 有而 B 没有的列 = removed
    generated_columns: list[str] = Field(default_factory=list)
    removed_columns: list[str] = Field(default_factory=list)
