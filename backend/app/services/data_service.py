"""数据上传分析服务：负责文件解析、基础数据画像、数据质量概览与预览生成。

设计约定：
- 本模块只做"读取与分析"，绝不修改、清洗或写回用户的原始数据；
- 不在本模块引入 HTTP 概念，业务异常统一抛出 :class:`DataServiceError`，
  由 API 层负责转换为对应 HTTP 响应；
- 尽量保持函数职责单一，方便后续 EDA / 特征工程模块复用其中解析能力。
"""

import io
import logging
import re
from datetime import date, datetime

import numpy as np
import pandas as pd
import pandas.api.types as pdt

from app.core.config import settings
from app.models.data import (
    BOOLEAN,
    CATEGORICAL,
    DATETIME,
    NUMERIC,
    TEXT,
    ColumnProfile,
    DataUploadResponse,
    DatasetInfo,
    QualityOverview,
)

logger = logging.getLogger(__name__)

# CSV 编码尝试顺序：BOM-UTF8 -> UTF-8 -> GB18030（GBK 超集，兼容大部分中文文件）
_CSV_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030")

# 布尔值的严格候选集合（整列为这些值时才判定为布尔型）
_STRICT_BOOLEAN_TOKENS = {"true", "false"}

# 疑似日期字符串的粗略正则（用于先筛掉明显非日期的文本，减少误判）
_DATE_HINT_RE = re.compile(
    r"(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2})"
    r"|(?:\d{1,2}[-/.]\d{1,2}[-/.]\d{4})"
    r"|(?:\d{1,2}:\d{2}(?::\d{2})?)"
)

# 单列字符串类型推断时最多检查的样本量
_SAMPLE_SIZE = 2000

# 判定为"分类"的最大唯一值占比（超过则按自由文本处理）
_CATEGORICAL_MAX_UNIQUE_RATIO = 0.5


class DataServiceError(Exception):
    """业务层自定义异常，携带友好错误信息与建议的 HTTP 状态码。"""

    def __init__(self, code: str, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def _short_error_message(raw: str, limit: int = 160) -> str:
    """将底层异常信息压缩为单行短文本（仅用于辅助提示，不返回 traceback）。"""
    return " ".join(raw.split())[:limit]


def extension_allowed(file_name: str) -> bool:
    """判断文件名扩展名是否在允许列表中。"""
    ext = file_name.lower().rsplit(".", 1)[-1]
    return ext in settings.data_allowed_extensions


def _read_csv(content: bytes) -> pd.DataFrame:
    """尝试以多种编码解析 CSV，给出友好的编码/格式错误提示。"""
    last_parse_error: Exception | None = None

    for encoding in _CSV_ENCODINGS:
        try:
            return pd.read_csv(io.BytesIO(content), encoding=encoding)
        except UnicodeDecodeError:
            # 当前编码无法解码，尝试下一种编码
            continue
        except pd.errors.EmptyDataError:
            raise DataServiceError(
                "csv_empty",
                "CSV 文件为空或未包含任何可解析的数据行。",
                status_code=422,
            ) from None
        except pd.errors.ParserError as exc:  # pragma: no cover - 由 pandas 触发
            last_parse_error = exc
            continue

    if last_parse_error is not None:
        raise DataServiceError(
            "csv_parse_error",
            f"CSV 解析失败，请检查文件格式（分隔符、引号、各行列数是否一致）。"
            f"原因：{_short_error_message(str(last_parse_error))}",
            status_code=422,
        ) from last_parse_error

    raise DataServiceError(
        "csv_encoding_error",
        "无法识别 CSV 文件编码。请将文件另存为 UTF-8 或 GBK/GB18030 编码后重新上传。",
        status_code=422,
    )


def _read_excel(content: bytes, file_type: str) -> pd.DataFrame:
    """解析 .xlsx / .xls 文件（仅读取首个工作表）。"""
    engine = "xlrd" if file_type == "xls" else "openpyxl"
    try:
        return pd.read_excel(io.BytesIO(content), engine=engine)
    except Exception as exc:  # 文件损坏/类型伪装等底层异常，统一转友好提示
        raise DataServiceError(
            "excel_parse_error",
            f"Excel 文件解析失败，请确认文件未损坏且格式正确。原因："
            f"{_short_error_message(str(exc))}",
            status_code=422,
        ) from exc


def _column_missing_mask(series: pd.Series) -> pd.Series:
    """计算列缺失掩码：pandas 缺失值 + 空字符串 / 纯空白字符串。"""
    mask = series.isna()
    if pdt.is_object_dtype(series.dtype) or pdt.is_string_dtype(series.dtype):
        text = series.astype("string").fillna("")
        mask = mask | text.str.strip().eq("")
    return mask


def _infer_string_type(series: pd.Series, missing_mask: pd.Series) -> str:
    """对字符串类型列做业务推断：boolean / datetime / numeric / categorical / text。"""
    values = series.loc[~missing_mask]
    n = len(values)
    if n == 0:
        # 整列为空时无从推断，统一标记为 text（由 empty_columns 指标单独提示）
        return TEXT

    sample = values.head(_SAMPLE_SIZE)
    tokens = sample.astype(str).str.strip().str.lower()

    # 1) 布尔值：全部取值是 true/false 之类的双状态词
    token_set = set(tokens)
    if token_set and token_set <= _STRICT_BOOLEAN_TOKENS:
        return BOOLEAN

    # 2) 日期时间：多数取值满足日期/时间形态且能被 pandas 正确解析
    date_like = tokens.str.contains(_DATE_HINT_RE, regex=True, na=False)
    if date_like.mean() >= 0.8:
        try:
            parsed = pd.to_datetime(tokens, errors="coerce", format="mixed")
        except (TypeError, ValueError):  # pragma: no cover - 兼容旧版本 pandas
            parsed = pd.to_datetime(tokens, errors="coerce")
        if parsed.notna().mean() >= 0.9:
            return DATETIME

    # 3) 数值：绝大多数取值可转为数字
    numeric_ratio = pd.to_numeric(tokens, errors="coerce").notna().mean()
    if numeric_ratio >= 0.9:
        return NUMERIC

    # 4) 分类 vs 文本：按不同取值占比粗略区分
    unique_count = series.loc[~missing_mask].nunique(dropna=True)
    if unique_count / n <= _CATEGORICAL_MAX_UNIQUE_RATIO:
        return CATEGORICAL
    return TEXT


def _infer_column_type(series: pd.Series, missing_mask: pd.Series) -> str:
    """依据 pandas dtype 与数据实际内容推断业务类型。"""
    dtype = series.dtype

    if pdt.is_bool_dtype(dtype):
        return BOOLEAN
    if pdt.is_datetime64_any_dtype(dtype):
        return DATETIME
    if isinstance(dtype, pd.CategoricalDtype):
        return CATEGORICAL
    if pdt.is_numeric_dtype(dtype):
        return NUMERIC
    if pdt.is_object_dtype(dtype) or pdt.is_string_dtype(dtype):
        return _infer_string_type(series, missing_mask)
    if pdt.is_timedelta64_dtype(dtype):
        return NUMERIC
    return TEXT


def _coerce_preview_value(value: Any) -> Any:
    """将单元格值转换为可 JSON 序列化的 Python 原生类型。

    NaN / NaT / pd.NA / None 统一转为 None；datetime 类转为 ISO 字符串。
    """
    if value is None:
        return None

    # 缺失值统一处理（pd.NA 的 isinstance 比较可能抛异常，故包一层）
    try:
        is_missing = bool(pd.isna(value))
    except (TypeError, ValueError):
        is_missing = False
    if is_missing:
        return None

    # 常见的 pandas / numpy 标量
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, np.datetime64, datetime, date)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (pd.Timedelta, np.timedelta64)):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (str, int, float, bool)):
        return value

    # 兜底：尝试使用 numpy 的 item() / Python 原生转换
    try:
        converted = value.item() if hasattr(value, "item") else value
        if isinstance(converted, (str, int, float, bool)) or converted is None:
            return converted
    except (ValueError, AttributeError):
        pass

    return str(value)


def _build_preview(frame: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    """取前 limit 行并转换为可安全 JSON 化的记录列表。"""
    records: list[dict[str, Any]] = []
    head = frame.head(limit)
    columns = list(head.columns)
    for _, row in head.iterrows():
        records.append({str(col): _coerce_preview_value(row[col]) for col in columns})
    return records


def _build_column_profiles(df: pd.DataFrame) -> tuple[list[ColumnProfile], list[pd.Series]]:
    """逐列计算画像，同时返回各列的缺失掩码供质量概览复用。"""
    profiles: list[ColumnProfile] = []
    masks: list[pd.Series] = []
    row_count = len(df)

    for col in df.columns:
        series = df[col]
        missing_mask = _column_missing_mask(series)
        masks.append(missing_mask)

        missing_count = int(missing_mask.sum())
        non_null_count = row_count - missing_count
        missing_percentage = round(missing_count / row_count * 100, 2) if row_count else 0.0

        valid = series.loc[~missing_mask]
        unique_count = int(valid.nunique(dropna=True)) if len(valid) else 0

        profiles.append(
            ColumnProfile(
                column_name=str(col),
                dtype=str(series.dtype),
                inferred_type=_infer_column_type(series, missing_mask),
                non_null_count=non_null_count,
                missing_count=missing_count,
                missing_percentage=missing_percentage,
                unique_count=unique_count,
            )
        )
    return profiles, masks


def _build_quality(
    df: pd.DataFrame,
    profiles: list[ColumnProfile],
    masks: list[pd.Series],
) -> tuple[QualityOverview, list[str]]:
    """根据列画像汇总数据质量指标并生成用户可读的 warnings。"""
    total_missing = 0
    columns_with_missing = 0
    empty_columns = 0
    constant_columns = 0
    empty_names: list[str] = []
    constant_names: list[str] = []

    for profile, mask in zip(profiles, masks):
        missing = profile.missing_count
        total_missing += missing
        if missing > 0:
            columns_with_missing += 1
        if mask.all():
            empty_columns += 1
            empty_names.append(profile.column_name)
            continue  # 空列不再判定常量
        valid_count = profile.non_null_count
        if valid_count > 0 and profile.unique_count == 1:
            constant_columns += 1
            constant_names.append(profile.column_name)

    duplicate_rows = int(df.duplicated().sum())

    overview = QualityOverview(
        total_missing=total_missing,
        columns_with_missing=columns_with_missing,
        duplicate_rows=duplicate_rows,
        empty_columns=empty_columns,
        constant_columns=constant_columns,
    )

    warnings: list[str] = []
    if columns_with_missing > 0:
        warnings.append(
            f"发现 {columns_with_missing} 个包含缺失值的字段（共 {total_missing} 个缺失单元格）"
        )
    if duplicate_rows > 0:
        warnings.append(f"发现 {duplicate_rows} 行重复数据")
    if empty_columns > 0:
        warnings.append(f"存在 {empty_columns} 个完全为空的字段：{', '.join(empty_names[:8])}")
    if constant_columns > 0:
        warnings.append(
            f"存在 {constant_columns} 个取值单一的常量字段（信息量较低）："
            f"{', '.join(constant_names[:8])}"
        )

    # 高基数标识符提示（如 ID 列），供用户后续评估是否用于建模
    for profile in profiles:
        if profile.inferred_type == NUMERIC and profile.non_null_count >= 10:
            unique_ratio = profile.unique_count / profile.non_null_count
            if unique_ratio >= 0.95:
                warnings.append(
                    f"字段 {profile.column_name} 可能属于高基数标识符（如 ID），"
                    f"若用于建模请评估其有效性"
                )

    if not warnings:
        warnings.append("未发现明显的数据质量问题")
    return overview, warnings


def _load_validated_frame(file_name: str, content: bytes) -> tuple[pd.DataFrame, str]:
    """校验并解析上传内容，返回 (DataFrame, 扩展名)。"""
    # --- 基础校验 ---
    if not file_name.strip():
        raise DataServiceError("invalid_file_name", "文件名不能为空，请重新选择文件。")

    ext = file_name.lower().rsplit(".", 1)[-1]
    if ext not in settings.data_allowed_extensions:
        supported = " / ".join(settings.data_allowed_extensions)
        raise DataServiceError(
            "unsupported_file_type",
            f"不支持 .{ext} 类型的文件，仅支持：{supported}。",
        )

    if not content:
        raise DataServiceError("empty_file", "上传的文件内容为空，请检查文件后重试。")

    max_bytes = settings.data_max_upload_bytes
    if len(content) > max_bytes:
        max_mb = settings.data_max_upload_mb
        raise DataServiceError(
            "file_too_large",
            f"文件大小超过限制（最大 {max_mb}MB），请压缩或抽样后上传。",
            status_code=413,
        )

    # --- 读取数据 ---
    if ext == "csv":
        df = _read_csv(content)
    else:
        df = _read_excel(content, ext)

    if len(df) == 0:
        raise DataServiceError(
            "no_rows",
            "文件中未读取到任何数据行，请确认表格包含有效数据。",
            status_code=422,
        )
    if len(df.columns) == 0:
        raise DataServiceError(
            "no_columns",
            "文件中未读取到任何数据列，请确认表格包含表头或有效数据。",
            status_code=422,
        )
    return df, ext


def analyze_upload_file(file_name: str, content: bytes) -> DataUploadResponse:
    """上传文件的完整分析流程：校验 -> 解析 -> 画像 -> 质量 -> 预览。

    仅做只读分析，不会修改或落盘用户上传的原始数据。
    """
    df, ext = _load_validated_frame(file_name, content)

    # --- 逐列画像与整体质量 ---
    column_profiles, masks = _build_column_profiles(df)
    quality, warnings = _build_quality(df, column_profiles, masks)

    # --- 组装响应 ---
    return DataUploadResponse(
        success=True,
        dataset=DatasetInfo(
            file_name=file_name,
            file_type=ext,
            file_size=len(content),
            rows=len(df),
            columns=len(df.columns),
        ),
        quality=quality,
        column_profiles=column_profiles,
        preview=_build_preview(df, settings.data_preview_rows),
        warnings=warnings,
    )


# ============================================================
# 以下为供 Dataset Session / EDA / 后续特征工程等模块复用的公开接口
# ============================================================


def read_dataframe(file_type: str, content: bytes) -> pd.DataFrame:
    """按文件类型把字节内容解析为 DataFrame（不含扩展名与大小校验）。

    供 Dataset Session 从临时目录加载会话数据时复用，保证与上传解析行为一致。
    """
    if file_type == "csv":
        return _read_csv(content)
    if file_type in ("xlsx", "xls"):
        return _read_excel(content, file_type)
    raise DataServiceError("unsupported_file_type", f"不支持 .{file_type} 类型的数据文件。")


def analyze_columns(df: pd.DataFrame) -> tuple[list[ColumnProfile], list[pd.Series]]:
    """返回 (列画像列表, 各列缺失掩码列表)，供上传画像与 EDA 复用。"""
    return _build_column_profiles(df)
