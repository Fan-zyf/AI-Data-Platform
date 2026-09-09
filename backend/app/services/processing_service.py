"""数据处理服务：Transformation Plan 的顺序执行引擎 + Preview / Apply。

设计要点：
- Plan 只允许预定义 operation；参数以普通数据处理方式使用，绝不 eval / exec；
- 原始数据不可修改：所有转换都在内存 DataFrame 上按顺序执行；
- Preview 只在内存执行，不落盘；Apply 全部成功后才原子写入新版本，任一步失败都不产生版本；
- 数值缩放/One-Hot 等转换会把拟合参数与编码映射写入版本 metadata，供未来 ML 阶段参考；
- 本阶段所有转换属于“通用数据准备 / 探索性处理”。未来 ML 阶段为防数据泄漏，
  模型相关预处理会在训练集上单独 fit，并通过 sklearn Pipeline 应用到验证/测试集
  （当前阶段不引入 sklearn Pipeline）。
"""

import logging
import math
import re
from datetime import datetime

import numpy as np
import pandas as pd
import pandas.api.types as pdt

from app.models.data import QualityOverview
from app.models.processing import (
    CONVERT_TYPE,
    DATE_FEATURES,
    DROP_COLUMNS,
    DROP_DUPLICATES,
    FILL_MISSING,
    ONE_HOT_ENCODE,
    ORIGINAL_VERSION,
    REMOVE_OUTLIERS,
    SCALE_NUMERIC,
    TEXT_TRANSFORM,
    ProcessingApplyResponse,
    ProcessingPreviewResponse,
    TransformationOperation,
    TransformationPlan,
)
from app.services import data_service as ds
from app.services import dataset_manager
from app.services import version_manager as vm

logger = logging.getLogger(__name__)

# One-Hot 高基数阈值：超过后必须显式确认（allow_high_cardinality=True）
ONE_HOT_HIGH_CARDINALITY = 50
# 单个 One-Hot 操作生成新列的绝对上限（防止字段爆炸）
ONE_HOT_MAX_GENERATED = 1000
# 预览行列数
PREVIEW_ROWS = 20

# boolean 转换严格白名单
_BOOLEAN_TRUE_TOKENS = {"true", "t", "yes", "y", "1"}
_BOOLEAN_FALSE_TOKENS = {"false", "f", "no", "n", "0"}


class ProcessingError(dataset_manager.DatasetSessionError):
    """处理引擎业务异常，携带 code / message / status_code。"""


def _perror(code: str, message: str, status_code: int = 400) -> ProcessingError:
    return ProcessingError(code, message, status_code)


def _is_numeric_dtype(series: pd.Series) -> bool:
    return pdt.is_numeric_dtype(series.dtype)


def _column_exists(frame: pd.DataFrame, column: str) -> bool:
    return column in frame.columns


def _ensure_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing_cols = [col for col in columns if not _column_exists(frame, col)]
    if missing_cols:
        raise _perror(
            "column_not_found",
            f"字段不存在：{', '.join(missing_cols)}。",
        )


def _to_float_values(series: pd.Series) -> pd.Series:
    """转为 float64 并保留缺失；非数值内容为 NaN。"""
    return pd.to_numeric(series, errors="coerce").astype("float64")


def _new_nulls_created(old: pd.Series, new: pd.Series, old_mask: pd.Series) -> int:
    """统计转换后由“原本有效”变为缺失的单元格数量。"""
    new_mask = ds.column_missing_mask(new)
    return int((old_mask.eq(False) & new_mask).sum())


# ============================================================
# 单个 operation 的执行器
# ============================================================


def _op_drop_duplicates(frame: pd.DataFrame, op: TransformationOperation) -> tuple[pd.DataFrame, dict, list[str]]:
    rows_before = len(frame)
    frame = frame.drop_duplicates(keep="first").reset_index(drop=True)
    removed = rows_before - len(frame)
    return frame, {"type": DROP_DUPLICATES, "effect": {"removed_rows": removed}}, []


def _column_family(series: pd.Series) -> str:
    """判断列在填充场景下的家族：numeric（数值）或 other（分类/文本/布尔）。"""
    inferred = ds.infer_column_type(series)
    if _is_numeric_dtype(series) or inferred == "numeric":
        return "numeric"
    return "other"


def _op_fill_missing(frame: pd.DataFrame, op: TransformationOperation) -> tuple[pd.DataFrame, dict, list[str]]:
    columns = list(op.columns)
    strategy = str(op.strategy)
    _ensure_columns(frame, columns)
    rows_before = len(frame)
    warnings: list[str] = []

    # drop_rows / drop_columns 是“删除式”的缺失处理策略
    if strategy == "drop_rows":
        masks = [ds.column_missing_mask(frame[col]) for col in columns]
        drop_mask = pd.concat(masks, axis=1).any(axis=1)
        frame = frame[~drop_mask].reset_index(drop=True)
        removed = rows_before - len(frame)
        return frame, {
            "type": FILL_MISSING,
            "strategy": strategy,
            "columns": columns,
            "effect": {"removed_rows": removed},
        }, warnings

    if strategy == "drop_columns":
        remaining = [c for c in frame.columns if c not in columns]
        if not remaining:
            raise _perror("drop_all_columns", "不允许删除全部字段，至少保留一个字段。")
        frame = frame[remaining]
        return frame, {
            "type": FILL_MISSING,
            "strategy": strategy,
            "columns": columns,
            "effect": {"removed_columns": columns},
        }, warnings

    # 逐列填充
    for column in columns:
        series = frame[column]
        missing_mask = ds.column_missing_mask(series)
        non_missing_count = int((~missing_mask).sum())
        if non_missing_count == 0:
            raise _perror(
                "column_all_missing",
                f"字段 {column} 全部为缺失，无法用 {strategy} 策略填充；"
                f"请选择 drop_rows / drop_columns 或先删除该字段。",
            )

        family = _column_family(frame[column])
        if strategy in ("mean", "median") and family != "numeric":
            raise _perror(
                "missing_strategy_incompatible",
                f"字段 {column} 不是数值类型，不能使用 {strategy} 策略填充；"
                f"请使用 mode / constant / drop_rows / drop_columns。",
            )
        if strategy == "mode" and family == "numeric":
            raise _perror(
                "missing_strategy_incompatible",
                f"数值字段 {column} 建议使用 mean / median / constant 填充。",
            )

        if strategy in ("mean", "median"):
            numeric = _to_float_values(series)
            finite = numeric[np.isfinite(numeric.to_numpy(dtype="float64"))]
            if finite.empty:
                raise _perror(
                    "column_all_missing",
                    f"字段 {column} 没有可用的有效数值，无法用 {strategy} 填充。",
                )
            fill = float(finite.mean()) if strategy == "mean" else float(finite.median())
            numeric = numeric.fillna(fill)
            frame[column] = numeric
        elif strategy == "mode":
            values = series.loc[~missing_mask]
            mode_value = values.mode(dropna=True)
            if mode_value.empty:  # pragma: no cover - 上面已保证有非缺失值
                raise _perror("column_all_missing", f"字段 {column} 无法计算众数。")
            fill = mode_value.iloc[0]
            frame[column] = series.fillna(fill)
        else:  # constant
            fill_value = op.fill_value
            if family == "numeric":
                try:
                    constant = float(fill_value)
                except (TypeError, ValueError):
                    raise _perror(
                        "missing_strategy_incompatible",
                        f"数值字段 {column} 的 constant 填充值必须是数字，不能填写字符串。",
                    ) from None
                numeric = _to_float_values(series).fillna(constant)
                frame[column] = numeric
            else:
                if fill_value is None or (isinstance(fill_value, str) and not fill_value.strip()):
                    raise _perror(
                        "missing_strategy_incompatible",
                        f"字段 {column} 的 constant 填充值不能为空。",
                    )
                frame[column] = series.fillna(fill_value)

    return frame, {"type": FILL_MISSING, "strategy": strategy, "columns": columns}, warnings


def _op_drop_columns(frame: pd.DataFrame, op: TransformationOperation) -> tuple[pd.DataFrame, dict, list[str]]:
    columns = list(op.columns)
    _ensure_columns(frame, columns)
    existing = [c for c in columns if _column_exists(frame, c)]
    remaining = [c for c in frame.columns if c not in existing]
    if not remaining:
        raise _perror("drop_all_columns", "不允许删除全部字段，至少保留一个字段。")
    frame = frame[remaining]
    return frame, {"type": DROP_COLUMNS, "columns": existing, "effect": {"removed_columns": existing}}, []


def _op_convert_type(frame: pd.DataFrame, op: TransformationOperation) -> tuple[pd.DataFrame, dict, list[str]]:
    column = str(op.column)
    _ensure_columns(frame, [column])
    target = str(op.target_type)
    series = frame[column]
    old_mask = ds.column_missing_mask(series)
    rows_before = len(frame)

    if target == "string":
        frame[column] = series.astype("string")
    elif target == "numeric":
        numeric = _to_float_values(series)
        created = _new_nulls_created(series, numeric, old_mask)
        if created > 0:
            raise _perror(
                "conversion_failed",
                f"字段 {column} 转为数值失败：有 {created} 个取值无法解析为数字。"
                f"请先检查这些取值是否混入了非数字内容。",
            )
        frame[column] = numeric
    elif target == "datetime":
        try:
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
        except (TypeError, ValueError):  # pragma: no cover - 旧 pandas
            parsed = pd.to_datetime(series, errors="coerce")
        created = _new_nulls_created(series, parsed, old_mask)
        if created > 0:
            raise _perror(
                "conversion_failed",
                f"字段 {column} 转为日期时间失败：有 {created} 个取值无法解析。"
                f"请确认它们符合常见的日期/时间格式。",
            )
        frame[column] = parsed
    else:  # boolean 严格转换
        parsed_values: list[object] = []
        invalid: list[str] = []
        for value in series.tolist():
            if value is None:
                parsed_values.append(np.nan)
                continue
            if isinstance(value, (float, int)) and (isinstance(value, float) and math.isnan(value)):
                parsed_values.append(np.nan)
                continue
            if isinstance(value, bool):
                parsed_values.append(bool(value))
                continue
            token = str(value).strip().lower()
            if token in _BOOLEAN_TRUE_TOKENS:
                parsed_values.append(True)
            elif token in _BOOLEAN_FALSE_TOKENS:
                parsed_values.append(False)
            else:
                parsed_values.append(np.nan)
                if len(invalid) < 5:
                    invalid.append(str(value)[:40])
        if invalid:
            raise _perror(
                "conversion_failed",
                f"字段 {column} 转为布尔失败：无法识别的取值（示例："
                f"{'、'.join(invalid)}）。仅支持 true/false/1/0/yes/no。",
            )
        converted = pd.Series(parsed_values, index=frame.index)
        frame[column] = converted

    return frame, {"type": CONVERT_TYPE, "column": column, "target_type": target}, []


def _op_remove_outliers(frame: pd.DataFrame, op: TransformationOperation) -> tuple[pd.DataFrame, dict, list[str]]:
    column = str(op.column)
    action = str(op.action)
    _ensure_columns(frame, [column])
    series = frame[column]
    old_mask = ds.column_missing_mask(series)
    numeric = _to_float_values(series)
    created = _new_nulls_created(series, numeric, old_mask)
    if created > 0:
        raise _perror(
            "outlier_column_not_numeric",
            f"字段 {column} 包含无法解析为数字的取值，不能执行 IQR 异常值处理。",
        )

    finite_mask = np.isfinite(numeric.to_numpy(dtype="float64"))
    finite = numeric[finite_mask]
    if finite.empty:
        raise _perror(
            "outlier_column_not_numeric",
            f"字段 {column} 没有可用于 IQR 计算的有效数值。",
        )
    arr = finite.to_numpy(dtype="float64")
    q1, q3 = np.quantile(arr, [0.25, 0.75])
    iqr = float(q3) - float(q1)
    lower = float(q1) - 1.5 * iqr
    upper = float(q3) + 1.5 * iqr
    rows_before = len(frame)

    if action == "clip":
        outlier_count = int(((numeric < lower) | (numeric > upper)).sum())
        clipped = numeric.clip(lower=lower, upper=upper)
        frame[column] = clipped
        return frame, {
            "type": REMOVE_OUTLIERS,
            "column": column,
            "method": "IQR",
            "action": "clip",
            "lower_bound": round(lower, 6),
            "upper_bound": round(upper, 6),
            "effect": {"clipped_rows": outlier_count},
        }, []

    outlier_mask = pd.Series((numeric < lower) | (numeric > upper), index=frame.index)
    frame = frame[~outlier_mask].reset_index(drop=True)
    removed = rows_before - len(frame)
    return frame, {
        "type": REMOVE_OUTLIERS,
        "column": column,
        "method": "IQR",
        "action": "remove_rows",
        "lower_bound": round(lower, 6),
        "upper_bound": round(upper, 6),
        "effect": {"removed_rows": removed},
    }, []


def _op_text_transform(frame: pd.DataFrame, op: TransformationOperation) -> tuple[pd.DataFrame, dict, list[str]]:
    columns = list(op.columns)
    action = str(op.action)
    _ensure_columns(frame, columns)
    for column in columns:
        series = frame[column]
        inferred = ds.infer_column_type(series)
        if inferred in ("numeric", "datetime", "boolean") or _is_numeric_dtype(series):
            raise _perror(
                "text_column_not_supported",
                f"字段 {column} 是 {inferred} 类型，不能执行文本处理；"
                f"仅支持 object / string 文本字段。",
            )
        text = series.astype("string")
        if action == "strip":
            text = text.str.strip()
        elif action == "lowercase":
            text = text.str.lower()
        else:
            text = text.str.upper()
        frame[column] = text
    return frame, {"type": TEXT_TRANSFORM, "columns": columns, "action": action}, []


def _op_date_features(frame: pd.DataFrame, op: TransformationOperation) -> tuple[pd.DataFrame, dict, list[str]]:
    column = str(op.column)
    features = list(op.features)
    _ensure_columns(frame, [column])
    series = frame[column]
    old_mask = ds.column_missing_mask(series)
    try:
        parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    except (TypeError, ValueError):  # pragma: no cover - 旧 pandas
        parsed = pd.to_datetime(series, errors="coerce")
    created = _new_nulls_created(series, parsed, old_mask)
    if created > 0:
        raise _perror(
            "datetime_conversion_failed",
            f"字段 {column} 无法安全识别为日期时间，不能派生日期特征"
            f"（有 {created} 个取值解析失败）。请先使用类型转换清洗该字段。",
        )

    feature_names = [f"{column}_{feature}" for feature in features]
    conflicts = [name for name in feature_names if _column_exists(frame, name)]
    if conflicts:
        raise _perror(
            "feature_column_conflict",
            f"派生字段名与现有字段冲突：{'、'.join(conflicts)}。"
            f"请先处理同名字段或更换日期字段。",
        )

    dt = parsed.dt
    accessors = {
        "year": dt.year,
        "month": dt.month,
        "day": dt.day,
        "day_of_week": dt.dayofweek,
        "quarter": dt.quarter,
    }
    generated: list[str] = []
    for feature in features:
        new_name = f"{column}_{feature}"
        frame[new_name] = accessors[feature]
        generated.append(new_name)

    return frame, {
        "type": DATE_FEATURES,
        "column": column,
        "features": features,
        "effect": {"generated_columns": generated},
    }, []


def _json_scalar(value) -> object:
    """把 numpy / pandas 标量转成可 JSON 序列化的 Python 原生类型。"""
    if value is None:
        return None
    if isinstance(value, bool):  # bool 是 int 子类，先判断
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (str, int, float)):
        return value
    try:
        item = value.item()
        return _json_scalar(item)
    except (AttributeError, ValueError):
        return str(value)


def _safe_slug(value, seen: set[str], prefix: str) -> str:
    """把分类取值转成安全的列名后缀，避免与已有生成列冲突。"""
    text = str(value).strip()
    slug = re.sub(r"[^\w]+", "_", text, flags=re.UNICODE).strip("_")
    if not slug:
        slug = "value"
    if slug[0].isdigit():
        slug = f"v_{slug}"
    if len(slug) > 40:
        slug = slug[:40].rstrip("_")
    candidate = slug
    counter = 1
    while candidate in seen:
        counter += 1
        suffix = f"_{counter}"
        candidate = f"{slug[: 40 - len(suffix)]}{suffix}"
    seen.add(candidate)
    return f"{prefix}_{candidate}"


def _op_one_hot_encode(frame: pd.DataFrame, op: TransformationOperation) -> tuple[pd.DataFrame, dict, list[str]]:
    columns = list(op.columns)
    allow_high = bool(op.allow_high_cardinality)
    _ensure_columns(frame, columns)
    warnings: list[str] = []
    encoding_map: dict[str, dict] = {}
    generated_all: list[str] = []

    for column in columns:
        series = frame[column]
        inferred = ds.infer_column_type(series)
        if inferred in ("numeric", "datetime"):
            raise _perror(
                "one_hot_not_categorical",
                f"字段 {column} 是 {inferred} 类型，One-Hot 仅支持分类 / 布尔 / 低基数文本字段。",
            )
        mask = ds.column_missing_mask(series)
        non_missing = series.loc[~mask]
        unique_count = int(non_missing.nunique(dropna=True))
        if unique_count > ONE_HOT_HIGH_CARDINALITY:
            if not allow_high:
                raise _perror(
                    "high_cardinality_confirmation_required",
                    f"字段 {column} 有 {unique_count} 个不同取值，超过 One-Hot 阈值 "
                    f"({ONE_HOT_HIGH_CARDINALITY})。为避免产生过多字段，请先确认："
                    f"设置 allow_high_cardinality=true 后再次执行。",
                )
            warnings.append(
                f"字段 {column} 为高基数字段（{unique_count} 个取值），已按你的确认执行 One-Hot 编码。"
            )
        if unique_count == 0:
            raise _perror("one_hot_not_categorical", f"字段 {column} 没有任何有效分类取值，无法编码。")

        # 稳定排序：统一按字符串展示排序，保证生成列名可复现
        unique_values = sorted(non_missing.drop_duplicates().tolist(), key=lambda v: str(v))
        if len(unique_values) > ONE_HOT_MAX_GENERATED - len(generated_all):
            raise _perror(
                "high_cardinality_confirmation_required",
                f"字段 {column} 的 One-Hot 将生成过多字段（超过上限 "
                f"{ONE_HOT_MAX_GENERATED}），为避免字段爆炸已拒绝执行。",
            )

        seen: set[str] = {str(c) for c in frame.columns} | set(generated_all)
        mapping: dict[str, str] = {}  # new_col -> original value
        new_columns: dict[str, pd.Series] = {}
        index = frame.index
        for value in unique_values:
            new_name = _safe_slug(value, seen, str(column))
            seen.add(new_name)
            generated_all.append(new_name)
            mapping[new_name] = _json_scalar(value)
            equal = (series == value).fillna(False).astype(bool)
            new_columns[new_name] = pd.Series(
                np.where(equal.to_numpy(), 1, 0), index=index, dtype="int8"
            )
        encoding_map[column] = mapping
        frame = frame.drop(columns=[column])
        for name, col_series in new_columns.items():
            frame[name] = col_series

    op_record = {
        "type": ONE_HOT_ENCODE,
        "columns": columns,
        "allow_high_cardinality": allow_high,
        "effect": {"generated_columns": generated_all, "removed_columns": columns},
    }
    return frame, op_record, warnings, encoding_map


def _op_scale_numeric(frame: pd.DataFrame, op: TransformationOperation) -> tuple[pd.DataFrame, dict, list[str]]:
    columns = list(op.columns)
    method = str(op.scale_method)
    _ensure_columns(frame, columns)
    warnings: list[str] = []
    fit: dict[str, dict] = {}

    for column in columns:
        series = frame[column]
        old_mask = ds.column_missing_mask(series)
        numeric = _to_float_values(series)
        created = _new_nulls_created(series, numeric, old_mask)
        if created > 0:
            raise _perror(
                "scale_column_not_numeric",
                f"字段 {column} 包含无法解析为数字的取值，不能执行数值缩放。",
            )
        arr = numeric.to_numpy(dtype="float64")
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            raise _perror(
                "scale_column_not_numeric",
                f"字段 {column} 没有可用于缩放的有效数值。",
            )

        # 无限值：为避免标准化/归一化产生 Inf/NaN，先替换为缺失并记录警告
        if bool(np.isinf(arr).any()):
            warnings.append(f"字段 {column} 含有无限值，已将其替换为缺失值后再缩放。")

        scaled = np.full(arr.shape, np.nan, dtype="float64")
        finite_mask = np.isfinite(arr)
        if method == "standardization":
            mean = float(arr[finite_mask].mean())
            std = float(arr[finite_mask].std(ddof=0))
            fit[column] = {"method": "standardization", "mean": round(mean, 10), "std": round(std, 10)}
            if std == 0 or finite_mask.sum() < 2:
                scaled[finite_mask] = 0.0
                warnings.append(
                    f"字段 {column} 为常量字段（std=0），标准化后置为 0，未发生除零错误。"
                )
            else:
                scaled[finite_mask] = (arr[finite_mask] - mean) / std
        else:
            vmin = float(arr[finite_mask].min())
            vmax = float(arr[finite_mask].max())
            fit[column] = {"method": "min_max", "min": round(vmin, 10), "max": round(vmax, 10)}
            if vmax == vmin:
                scaled[finite_mask] = 0.0
                warnings.append(
                    f"字段 {column} 为常量字段（max=min），归一化后置为 0，未发生除零错误。"
                )
            else:
                scaled[finite_mask] = (arr[finite_mask] - vmin) / (vmax - vmin)

        frame[column] = pd.Series(scaled, index=frame.index)

    return frame, {
        "type": SCALE_NUMERIC,
        "columns": columns,
        "scale_method": method,
        "effect": {"fit": fit},
    }, warnings


# ============================================================
# Plan 执行引擎（含顺序执行与原子性保证）
# ============================================================

# dispatch 返回值：op 若额外产出 metadata（encoding map）会放在第 4 个返回值
_OP_DISPATCH = {
    DROP_DUPLICATES: _op_drop_duplicates,
    FILL_MISSING: _op_fill_missing,
    DROP_COLUMNS: _op_drop_columns,
    CONVERT_TYPE: _op_convert_type,
    REMOVE_OUTLIERS: _op_remove_outliers,
    TEXT_TRANSFORM: _op_text_transform,
    DATE_FEATURES: _op_date_features,
    ONE_HOT_ENCODE: _op_one_hot_encode,
    SCALE_NUMERIC: _op_scale_numeric,
}


def _run_plan(frame: pd.DataFrame, operations: list[TransformationOperation]) -> tuple[pd.DataFrame, list[dict], list[str], dict]:
    """按顺序执行整个 Plan；任一步抛错即整体失败（由调用方保证不落盘）。"""
    applied: list[dict] = []
    warnings: list[str] = []
    extra_metadata: dict = {}

    for op in operations:
        rows_before = len(frame)
        handler = _OP_DISPATCH.get(op.type)
        if handler is None:  # pragma: no cover - Literal 已约束
            raise _perror("unknown_operation", f"不支持的操作类型：{op.type}。")
        result = handler(frame, op)
        frame = result[0]
        record = dict(result[1])
        # 记录 effect 里的行数变化以便审计
        record["effect"] = dict(record.get("effect") or {})
        record["effect"]["rows_before"] = rows_before
        record["effect"]["rows_after"] = len(frame)
        applied.append(record)
        if len(result) >= 4 and isinstance(result[3], dict):
            extra_metadata.update(result[3])
        warnings.extend(result[2] if len(result) > 2 else [])

    return frame, applied, warnings, extra_metadata


def _metrics(frame: pd.DataFrame) -> dict:
    overview, _ = ds.quality_overview(frame)
    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "missing": overview.total_missing,
        "duplicates": overview.duplicate_rows,
        "quality": overview,
    }


def _validate_result(frame: pd.DataFrame) -> None:
    """转换结果有效性校验（0 行 / 0 列必须拒绝）。"""
    if len(frame) == 0:
        raise _perror(
            "empty_result",
            "转换后数据集变为 0 行，请检查是否删除了过多记录。",
            status_code=422,
        )
    if len(frame.columns) == 0:
        raise _perror(
            "drop_all_columns",
            "转换后不存在任何字段，请至少保留一个字段。",
            status_code=422,
        )


def _common_warnings(generated: list[str], removed: list[str]) -> list[str]:
    text: list[str] = []
    if generated:
        text.append(f"将新增 {len(generated)} 个字段（示例：{'、'.join(generated[:5])}）")
    if removed:
        text.append(f"将移除 {len(removed)} 个字段（{'、'.join(removed[:8])}）")
    return text


def preview_plan(dataset_id: str, plan: TransformationPlan) -> ProcessingPreviewResponse:
    """Preview：仅在内存执行一次 Plan，绝不写入新版本。"""
    source_version = plan.source_version_id or ORIGINAL_VERSION
    source_frame, source_info = vm.load_dataset_version(dataset_id, source_version)
    # 深拷贝，防止对来源 DataFrame 的任何原地修改
    working = source_frame.copy(deep=True)

    before = _metrics(source_frame)
    transformed, applied, warnings, _ = _run_plan(working, plan.operations)
    _validate_result(transformed)
    after = _metrics(transformed)

    columns_before = set(str(c) for c in source_frame.columns)
    columns_after = set(str(c) for c in transformed.columns)
    generated = sorted(columns_after - columns_before)
    removed = sorted(columns_before - columns_after)

    all_warnings = list(warnings) + _common_warnings(generated, removed)
    if not all_warnings:
        all_warnings.append("预览结果无额外警告；请确认 Before / After 数据后点击应用。")

    return ProcessingPreviewResponse(
        success=True,
        dataset_id=dataset_id,
        source_version_id=source_version,
        rows_before=before["rows"],
        rows_after=after["rows"],
        columns_before=before["columns"],
        columns_after=after["columns"],
        missing_before=before["missing"],
        missing_after=after["missing"],
        duplicates_before=before["duplicates"],
        duplicates_after=after["duplicates"],
        generated_columns=generated,
        removed_columns=removed,
        warnings=all_warnings,
        preview_before=ds.build_preview_rows(source_frame, PREVIEW_ROWS),
        preview_after=ds.build_preview_rows(transformed, PREVIEW_ROWS),
    )


def apply_plan(dataset_id: str, plan: TransformationPlan) -> ProcessingApplyResponse:
    """Apply：全部步骤成功后才生成新版本；任一步失败都不保存“半成品版本”。"""
    source_version = plan.source_version_id or ORIGINAL_VERSION
    source_frame, source_info = vm.load_dataset_version(dataset_id, source_version)
    working = source_frame.copy(deep=True)

    transformed, applied, warnings, extra_metadata = _run_plan(working, plan.operations)
    _validate_result(transformed)

    result_metrics = _metrics(transformed)
    metadata_payload = {
        "transform_warnings": list(warnings),
        "transform_fit": extra_metadata,
    }

    meta = vm.persist_new_version(
        dataset_id=dataset_id,
        parent_version_id=source_version,
        frame=transformed,
        operations=applied,
        quality=result_metrics["quality"],
        metadata_payload=metadata_payload,
    )

    created_at = datetime.fromisoformat(meta["created_at"])
    return ProcessingApplyResponse(
        success=True,
        dataset_id=dataset_id,
        version_id=meta["version_id"],
        parent_version_id=meta["parent_version_id"],
        source_version_id=meta["source_version"],
        created_at=created_at,
        rows=result_metrics["rows"],
        columns=result_metrics["columns"],
        operations_applied=applied,
        quality=result_metrics["quality"],
        warnings=list(warnings),
        preview=ds.build_preview_rows(transformed, PREVIEW_ROWS),
    )
