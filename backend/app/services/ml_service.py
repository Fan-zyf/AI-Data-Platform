"""v0.5 Machine Learning Service：训练 / CV / 模型比较 / 最终评估 / 预测。

防止 Data Leakage 的核心原则（硬性约束，全部在训练请求内执行）：
    版本数据 -> 选 Target / Features -> train/test split
        -> 仅在训练集内 fit preprocessing（Pipeline + ColumnTransformer）
        -> 仅在训练集内做 CV -> 按 CV 主指标选最佳模型
        -> 在完整训练集上 refit 最佳模型 -> 对 untouched Test Set 做最后一次评估

实现约定：
- 所有统计型预处理（median / most_frequent 填充、OneHot、StandardScaler）只出现在
  sklearn Pipeline 中，随 CV fold / refit 一起 fit，绝不先在全量数据上 fit 再拆分；
- 若所选派生版本在 v0.4 中执行过全量统计型操作（mean/median/mode 填充、IQR、
  One-Hot、标准化/归一化），返回 potential_data_leakage warning（不阻止训练）；
- 候选模型逐个训练，允许部分失败并记录 model_errors；全部失败才整体失败且不落盘；
- 实验目录与 joblib 由 experiment_manager 原子持久化，任一步失败不留半成品。
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pandas.api.types as pdt
import sklearn
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from app.core.config import settings
from app.models.data import BOOLEAN, CATEGORICAL, DATETIME, NUMERIC, TEXT
from app.models.ml import MLPredictRequest, MLTrainRequest
from app.services import data_service as ds
from app.services import dataset_manager
from app.services import experiment_manager as em
from app.services import version_manager as vm
from app.services.dataset_manager import DatasetSessionError
from app.services.preprocessing_factory import (
    build_preprocessing_pipeline,
    extract_feature_names,
)

logger = logging.getLogger(__name__)

ML_APP_VERSION = "0.5.0"

# 每个任务类型使用的主评估指标（用于模型比较与择优）
_PRIMARY_METRICS = {"classification": "f1_macro", "regression": "rmse"}
# 各任务支持的候选模型
_MODEL_SUPPORT = {
    "classification": {"dummy", "logistic_regression", "random_forest"},
    "regression": {"dummy", "ridge", "random_forest"},
}
# 分类模型的所有指标
_CLASSIF_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
)
_REGRESSION_METRICS = ("mae", "rmse", "r2")
_RF_N_ESTIMATORS = 150


class MLServiceError(dataset_manager.DatasetSessionError):
    """ML 训练 / 预测的业务异常（携带 code / message / HTTP status）。"""


def _merror(code: str, message: str, status_code: int = 400) -> MLServiceError:
    return MLServiceError(code, message, status_code)


# ============================================================
# JSON 安全与数值处理
# ============================================================


def _json_safe(value):
    """把 numpy / pandas 标量与嵌套结构转成 JSON 安全的 Python 原生类型。"""
    if value is None:
        return None
    if isinstance(value, bool):  # bool 是 int 子类，必须先判断
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.ndarray, list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Series, pd.Index)):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (str, int, float)):
        return value
    try:
        item = value.item()
        if item is not value:
            return _json_safe(item)
    except (AttributeError, ValueError):
        pass
    return str(value)


def _rnd(value, digits: int = 6):
    """四舍五入到指定位数；非有限数返回 None（JSON 安全）。"""
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return round(number, digits)


def _stats(values: list[float]) -> dict:
    """计算 (values, mean, std)，全部四舍五入为 JSON 安全浮点。"""
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    mean = float(np.mean(finite)) if finite else None
    std = float(np.std(finite)) if finite else None
    return {
        "values": [_rnd(v) for v in values],
        "mean": _rnd(mean),
        "std": _rnd(std),
    }


# ============================================================
# 候选模型 / 预处理 Pipeline
# ============================================================


def _build_estimator(model_key: str, task_type: str, random_state: int):
    """构建模型估计器（不带预处理）。"""
    if task_type == "classification":
        if model_key == "dummy":
            return DummyClassifier(strategy="most_frequent")
        if model_key == "logistic_regression":
            return LogisticRegression(max_iter=1000, random_state=random_state)
        if model_key == "random_forest":
            return RandomForestClassifier(
                n_estimators=_RF_N_ESTIMATORS,
                random_state=random_state,
                n_jobs=-1,
            )
    else:
        if model_key == "dummy":
            return DummyRegressor(strategy="mean")
        if model_key == "ridge":
            return Ridge()
        if model_key == "random_forest":
            return RandomForestRegressor(
                n_estimators=_RF_N_ESTIMATORS,
                random_state=random_state,
                n_jobs=-1,
            )
    raise _merror("invalid_model", f"未知的候选模型：{model_key}")


def _build_pipeline(
    model_key: str,
    task_type: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    random_state: int,
) -> Pipeline:
    """构造完整 Pipeline：类型规整 + ColumnTransformer + 模型。

    preprocessing 只会随 Pipeline 在训练数据上 fit，预测复用已保存的 Pipeline。
    """
    preprocessing = build_preprocessing_pipeline(numeric_cols, categorical_cols)
    estimator = _build_estimator(model_key, task_type, random_state)
    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("model", estimator),
        ]
    )


def _validate_candidate_models(candidates: list[str] | None, task_type: str) -> list[str]:
    """校验候选模型与任务类型匹配，返回去重后的模型 key 列表。"""
    if not candidates:
        return sorted(_MODEL_SUPPORT[task_type])
    allowed = _MODEL_SUPPORT[task_type]
    for name in candidates:
        if name not in allowed:
            raise _merror(
                "model_not_supported_for_task",
                f"候选模型 {name} 不适用于 {task_type} 任务"
                f"（可用：{', '.join(sorted(allowed))}）。",
            )
    return list(dict.fromkeys(candidates))


# ============================================================
# 列画像 / 任务类型推断
# ============================================================


def _inspect_column(series: pd.Series, column: str) -> dict:
    """单列的画像信息（用于特征选择与目标校验）。"""
    missing_mask = ds.column_missing_mask(series)
    n_valid = int((~missing_mask).sum())
    valid = series.loc[~missing_mask]
    n_unique = int(valid.nunique(dropna=True)) if n_valid else 0
    return {
        "column": column,
        "inferred": ds.infer_column_type(series),
        "dtype": str(series.dtype),
        "n_valid": n_valid,
        "n_missing": int(missing_mask.sum()),
        "n_unique": n_unique,
        "unique_ratio": (n_unique / n_valid) if n_valid else 0.0,
    }


def _resolve_task_type(requested: str, target_info: dict, target: str) -> tuple[str, str]:
    """返回 (task_type, reason)。task_type 必须为 classification / regression。"""
    inferred = target_info["inferred"]
    uniq = target_info["n_unique"]
    n_valid = target_info["n_valid"]

    if n_valid == 0:
        raise _merror("target_all_missing", f"目标字段 {target} 全部为缺失，无法训练。", 422)
    if uniq < 2:
        raise _merror(
            "target_constant",
            f"目标字段 {target} 仅包含 {uniq} 个不同取值（目标值单一），无法训练。"
            "请选择至少有 2 个不同取值的字段作为目标。",
        )

    if requested != "auto":
        task_type = requested
        if task_type == "classification":
            if inferred == DATETIME:
                raise _merror(
                    "target_not_supported",
                    f"目标字段 {target} 是日期时间类型，不支持分类训练。",
                )
            if uniq < 2:
                raise _merror(
                    "target_constant",
                    f"目标字段 {target} 仅包含 {uniq} 个不同取值，无法进行分类训练"
                    "（至少需要 2 个类别）。",
                )
            if uniq > settings.ml_max_unique_classes:
                raise _merror(
                    "target_too_many_classes",
                    f"目标字段 {target} 有 {uniq} 个不同类别，超过分类上限 "
                    f"{settings.ml_max_unique_classes}。",
                )
            return task_type, "按请求指定为分类任务"
        # regression
        if inferred == DATETIME:
            raise _merror(
                "target_not_supported",
                f"目标字段 {target} 是日期时间类型，不支持回归训练。",
            )
        if uniq < 2:
            raise _merror(
                "target_constant",
                f"目标字段 {target} 仅包含 {uniq} 个不同取值，无法进行回归训练。",
            )
        return task_type, "按请求指定为回归任务"

    # ---- auto 推断（保守规则）----
    if inferred == DATETIME:
        raise _merror(
            "target_not_supported",
            f"目标字段 {target} 是日期时间类型，无法自动推断任务；"
            "请选择分类 / 回归目标字段或显式指定 task_type。",
        )
    if inferred in (BOOLEAN, CATEGORICAL):
        reason = f"目标字段 {target} 为{('布尔' if inferred == BOOLEAN else '分类')}类型，"
        if uniq > settings.ml_max_unique_classes:
            raise _merror(
                "target_too_many_classes",
                f"目标字段 {target} 有 {uniq} 个不同类别，超过分类上限 "
                f"{settings.ml_max_unique_classes}。",
            )
        return "classification", reason + "自动推断为分类任务"
    if inferred == TEXT:
        if uniq <= settings.ml_max_unique_classes:
            return (
                "classification",
                f"目标字段 {target} 为低基数字符串（{uniq} 个取值），自动推断为分类任务",
            )
        raise _merror(
            "target_not_supported",
            f"目标字段 {target} 是高基数自由文本（{uniq} 个取值），无法用作预测目标；"
            "请选择类别数较少或数值型的目标字段。",
        )
    # numeric / 数值字符串：综合唯一值个数与占比判断分类/回归
    if uniq <= 10:
        return (
            "classification",
            f"目标字段 {target} 为低基数数值（{uniq} 个取值），自动推断为分类任务",
        )
    if n_valid >= 30 and uniq >= 20 and target_info["unique_ratio"] >= 0.1:
        return (
            "regression",
            f"目标字段 {target} 为连续数值（{uniq} 个取值 / 唯一率 "
            f"{target_info['unique_ratio']:.2f}），自动推断为回归任务",
        )
    return (
        "regression",
        f"目标字段 {target} 具有较多离散取值（{uniq} 个），自动推断为回归任务",
    )


# ============================================================
# 特征选择（无任何统计参数被学习，仅基于类型 / 基数判断）
# ============================================================


def _feature_plan(
    frame: pd.DataFrame,
    target: str,
    requested_features: list[str] | None,
    excluded_by_user: list[str],
) -> tuple[dict, list[str], list[str], list[dict], list[str]]:
    """选择可用特征。

    返回 (column_info, numeric_cols, categorical_cols, excluded, warnings)：
    - numeric_cols / categorical_cols：进入预处理 Pipeline 的字段（按原表顺序）;
    - excluded：被排除字段列表 [{column, reason}]，供 metadata 追溯。
    """
    warnings: list[str] = []
    excluded: list[dict] = []
    info_by_column: dict[str, dict] = {}
    for col in frame.columns:
        info_by_column[str(col)] = _inspect_column(frame[col], str(col))

    columns = [str(c) for c in frame.columns]
    exclude_set = {str(c) for c in (excluded_by_user or [])}

    if requested_features:
        missing = [c for c in requested_features if c not in columns]
        if missing:
            raise _merror(
                "column_not_found",
                f"特征字段不存在：{', '.join(missing)}。",
            )
        if target in requested_features:
            raise _merror(
                "target_in_features",
                f"目标字段 {target} 不能同时作为特征参与训练。",
            )
        candidate_order = [str(c) for c in requested_features]
        explicit = True
    else:
        candidate_order = [c for c in columns if c != target]
        explicit = False

    selected: list[str] = []
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []

    def exclude(col: str, reason: str) -> None:
        excluded.append({"column": col, "reason": reason})

    for col in candidate_order:
        if col in exclude_set:
            exclude(col, "用户显式排除")
            continue
        info = info_by_column[col]
        inferred = info["inferred"]
        n_valid = info["n_valid"]
        n_unique = info["n_unique"]

        if n_valid == 0:
            if explicit:
                raise _merror(
                    "feature_all_missing",
                    f"特征字段 {col} 全部为缺失，无法参与训练。",
                )
            exclude(col, "字段完全为空")
            continue
        if n_unique <= 1:
            if explicit:
                # 显式常量特征保留（记 warning），但必须落入 numeric/categorical 列表，
                # 否则后续 feature 子集会丢失该列
                warnings.append(f"字段 {col} 取值单一（常量），对模型信息量很低。")
                selected.append(col)
                if inferred == NUMERIC or pdt.is_numeric_dtype(frame[col].dtype):
                    numeric_cols.append(col)
                else:
                    categorical_cols.append(col)
            else:
                exclude(col, "取值单一的常量字段")
            continue

        if inferred == NUMERIC or pdt.is_numeric_dtype(frame[col].dtype):
            # 高基数标识符（如整型 ID）自动排除：仅当整型、且唯一率极高时才判为 ID，
            # 避免把普通连续数值特征（浮点唯一率同样接近 1）误杀
            if (
                not explicit
                and n_valid >= 10
                and n_unique >= 10
                and info["unique_ratio"] >= settings.ml_id_like_unique_ratio
                and pdt.is_integer_dtype(frame[col].dtype)
            ):
                exclude(col, "疑似整型高基数标识符（如 ID），对预测缺少泛化意义")
                continue
            if (
                explicit
                and pdt.is_integer_dtype(frame[col].dtype)
                and info["unique_ratio"] >= settings.ml_id_like_unique_ratio
            ):
                warnings.append(f"字段 {col} 疑似整型标识符（唯一率极高），请评估其有效性。")
            selected.append(col)
            numeric_cols.append(col)
            continue

        if inferred == DATETIME:
            if explicit:
                raise _merror(
                    "unsupported_feature_type",
                    f"特征字段 {col} 是日期时间类型；请先通过日期特征工程生成可用字段。",
                )
            exclude(col, "日期时间字段暂不支持直接建模")
            continue

        if inferred == TEXT:
            low_card = (
                n_unique <= settings.ml_low_cardinality_text_unique
                and info["unique_ratio"] <= settings.ml_low_cardinality_text_ratio
            )
            if not low_card:
                if explicit:
                    raise _merror(
                        "unsupported_feature_type",
                        f"特征字段 {col} 是高基数自由文本（{n_unique} 个取值），无法直接建模。",
                    )
                exclude(col, "高基数自由文本字段")
                continue
            # 低基数文本按分类特征处理
            categorical_cols.append(col)
            selected.append(col)
            continue

        # boolean / categorical
        categorical_cols.append(col)
        selected.append(col)

    # 编码维度估计：数值 1 维 + One-Hot 每类取值 1 维（防特征爆炸）
    encoded_dim = len(numeric_cols)
    for col in categorical_cols:
        encoded_dim += max(info_by_column[col]["n_unique"], 1)
    if encoded_dim > settings.ml_max_encoded_features:
        raise _merror(
            "feature_explosion",
            f"特征 One-Hot 展开后约 {encoded_dim} 维，超过上限 "
            f"{settings.ml_max_encoded_features}。请减少高基数分类特征或改用特征工程。",
        )
    if len(selected) > settings.ml_max_features:
        raise _merror(
            "too_many_features",
            f"选中特征数 {len(selected)} 超过上限 {settings.ml_max_features}，请精简特征。",
        )
    if not selected:
        raise _merror(
            "no_features",
            "没有可用于训练的特征字段。请确认除目标外存在数值 / 分类 / 布尔 / 低基数文本字段，"
            "或通过特征工程生成可用字段。",
        )
    return info_by_column, numeric_cols, categorical_cols, excluded, warnings


# ============================================================
# Data Leakage 检测（对 v0.4 派生版本）
# ============================================================


def _leakage_check(version_info: dict) -> tuple[bool, list[str], list[str]]:
    """检测所选派生版本是否执行过全量统计型 v0.4 操作。

    返回 (flagged, sensitive_operation_labels, warnings)。
    """
    if version_info.get("is_original"):
        return False, [], []
    ops = version_info.get("operations") or []
    flagged_labels: list[str] = []
    for op in ops:
        op_type = str(op.get("type", ""))
        if op_type == "fill_missing":
            strategy = str(op.get("strategy", ""))
            if strategy in ("mean", "median", "mode"):
                flagged_labels.append(f"fill_missing({strategy})")
        elif op_type == "remove_outliers":
            flagged_labels.append("remove_outliers(IQR)")
        elif op_type == "one_hot_encode":
            flagged_labels.append("one_hot_encode")
        elif op_type == "scale_numeric":
            flagged_labels.append(
                f"scale_numeric({op.get('scale_method', '')})"
            )
    if not flagged_labels:
        return False, [], []
    warnings = [
        "所选数据版本在 v0.4 中执行过全量统计型预处理（"
        + "、".join(sorted(set(flagged_labels)))
        + "），其统计参数基于整份数据（含本应属于测试集的行）计算，可能引入 data leakage，"
        "CV / 测试评估结果可能偏乐观。为避免数据泄漏，建议从 original 版本或仅含非统计型操作"
        "（删除 / 类型转换 / 文本清洗等）的版本训练。本次训练将继续执行，但请谨慎解读指标。"
    ]
    return True, list(sorted(set(flagged_labels))), warnings


# ============================================================
# 目标清理：整行按目标缺失剔除（无任何统计学习）
# ============================================================


def _clean_target_rows(
    frame: pd.DataFrame, target: str, task_type: str
) -> tuple[pd.DataFrame, pd.Series, int]:
    """按目标字段有效性剔除行，返回 (clean_frame, y, dropped)。

    回归目标做纯数值解析；解析失败的明确报错，绝不静默丢弃。
    """
    series = frame[target]
    missing_mask = ds.column_missing_mask(series)

    if task_type == "regression":
        parsed = pd.to_numeric(series, errors="coerce")
        invalid = int((~missing_mask & parsed.isna()).sum())
        if invalid:
            raise _merror(
                "target_not_numeric",
                f"目标字段 {target} 有 {invalid} 个取值无法解析为数字，不能执行回归。"
                "请先清洗数据或选择分类任务。",
                422,
            )
        keep = parsed.notna()
        y = parsed[keep].astype("float64").reset_index(drop=True)
    else:
        keep = ~missing_mask
        y = series[keep]
        if pdt.is_string_dtype(y.dtype):
            y = y.astype(object)
        if pdt.is_bool_dtype(y.dtype):
            y = y.astype(int)
        elif y.dtype == object:
            # 修复 v0.7.x: 含缺失值的 bool 列 dtype 会变成 object（值仍是 bool 真值），
            # 仍要转 int，否则 sklearn type_of_target 返回 'unknown'。
            non_null = y.dropna()
            if len(non_null) > 0 and all(
                isinstance(v, (bool, np.bool_)) for v in non_null
            ):
                y = y.astype(int)
        y = y.reset_index(drop=True)

    dropped = int((~keep).sum())
    clean = frame.loc[keep.to_numpy()].reset_index(drop=True)
    if len(y) < settings.ml_min_total_rows:
        raise _merror(
            "not_enough_samples",
            f"剔除目标缺失行后仅剩 {len(y)} 行，少于训练所需最少样本数 "
            f"{settings.ml_min_total_rows}。请提供更多数据或重新选择目标。",
            422,
        )
    return clean, y, dropped


# ============================================================
# 模型评估：CV + 最终测试
# ============================================================


def _compute_classification_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }


def _compute_regression_metrics(y_true, y_pred) -> dict:
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
    }


def _run_cv(
    model_keys: list[str],
    task_type: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    folds: int,
    random_state: int,
) -> tuple[dict, dict]:
    """在训练集内对每个候选模型做 CV，返回 (results, model_errors)。"""
    results: dict = {}
    model_errors: dict = {}
    metric_names = _CLASSIF_METRICS if task_type == "classification" else _REGRESSION_METRICS

    for key in model_keys:
        collected: dict[str, list[float]] = {name: [] for name in metric_names}
        fit_times: list[float] = []
        # 每个候选模型重新生成折（生成器不能跨模型复用）
        if task_type == "classification":
            cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
            split_indices = cv.split(X_train, y_train)
        else:
            cv = KFold(n_splits=folds, shuffle=True, random_state=random_state)
            split_indices = cv.split(X_train)
        try:
            for train_idx, valid_idx in split_indices:
                X_fold, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
                y_fold, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]
                pipeline = _build_pipeline(
                    key, task_type, numeric_cols, categorical_cols, random_state
                )
                start = time.perf_counter()
                pipeline.fit(X_fold, y_fold)
                fit_times.append(time.perf_counter() - start)
                y_pred = pipeline.predict(X_val)
                metrics = (
                    _compute_classification_metrics(y_val, y_pred)
                    if task_type == "classification"
                    else _compute_regression_metrics(y_val, y_pred)
                )
                for name in metric_names:
                    collected[name].append(metrics[name])
            results[key] = {
                "folds": folds,
                "metrics": {name: _stats(collected[name]) for name in metric_names},
                "fit_time_mean_seconds": _rnd(float(np.mean(fit_times)) if fit_times else 0.0),
            }
        except Exception as exc:  # noqa: BLE001 - 允许单模型失败
            logger.warning("candidate model %s failed in CV: %s", key, exc)
            model_errors[key] = " ".join(str(exc).split())[:400]
    return results, model_errors


def _rank_models(results: dict, primary: str, maximize: bool) -> list[dict]:
    """按主指标对候选模型排名（desc for 分类，asc for 回归）。"""
    ranked: list[dict] = []
    for key, detail in results.items():
        mean = detail["metrics"].get(primary, {}).get("mean")
        if mean is None:
            continue
        ranked.append({"model": key, "primary_mean": mean})
    ranked.sort(key=lambda item: item["primary_mean"], reverse=maximize)
    return ranked


def _compress_curve(x_values, y_values, limit: int) -> tuple[list[float], list[float]]:
    """把 ROC 曲线压缩到最多 limit 个点（保留端点，等距抽样）。"""
    xs = [float(v) for v in x_values]
    ys = [float(v) for v in y_values]
    if len(xs) <= limit:
        return xs, ys
    indices = np.unique(np.linspace(0, len(xs) - 1, limit).astype(int))
    return [xs[int(i)] for i in indices], [ys[int(i)] for i in indices]


def _sample_pairs(actual, predicted, limit: int) -> list[dict]:
    """回归散点：最多 limit 个 (actual, predicted) 数据点。"""
    points = [
        {"actual": _rnd(a), "predicted": _rnd(p)}
        for a, p in zip(actual, predicted)
    ]
    if len(points) <= limit:
        return points
    indices = np.unique(np.linspace(0, len(points) - 1, limit).astype(int))
    return [points[int(i)] for i in indices]


def _evaluate_test(
    pipeline: Pipeline,
    task_type: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """对 untouched Test Set 执行唯一一次最终评估。"""
    model = pipeline.named_steps["model"]
    y_pred = pipeline.predict(X_test)

    if task_type == "classification":
        metrics = _compute_classification_metrics(y_test, y_pred)
        result: dict = {"metrics": {k: _rnd(v) for k, v in metrics.items()}}
        class_labels = [c for c in model.classes_]
        labels_json = [_json_safe(c) for c in class_labels]
        result["class_labels"] = labels_json
        # 混淆矩阵 + 每类指标
        matrix = confusion_matrix(y_test, y_pred, labels=class_labels)
        result["confusion_matrix"] = {
            "labels": labels_json,
            "matrix": _json_safe(matrix.tolist()),
        }
        p, r, f1, support = precision_recall_fscore_support(
            y_test, y_pred, labels=class_labels, zero_division=0
        )
        result["per_class_metrics"] = [
            {
                "class": labels_json[i],
                "precision": _rnd(float(p[i])),
                "recall": _rnd(float(r[i])),
                "f1": _rnd(float(f1[i])),
                "support": int(support[i]),
            }
            for i in range(len(class_labels))
        ]
        # 二分类 + 模型支持概率时输出 ROC / AUC（压缩曲线点）
        result["roc"] = None
        if len(class_labels) == 2 and hasattr(model, "predict_proba"):
            try:
                proba = pipeline.predict_proba(X_test)
                positive_class = class_labels[1]
                positive_col = list(model.classes_).index(positive_class)
                probs = proba[:, positive_col]
                # sklearn>=1.9 的 roc_auc_score 不再接收 pos_label（自动推断正类）
                auc = float(roc_auc_score(y_test, probs))
                fpr, tpr, _ = roc_curve(y_test, probs, pos_label=positive_class)
                fx, fy = _compress_curve(fpr, tpr, settings.ml_roc_max_points)
                result["roc"] = {
                    "auc": _rnd(auc),
                    "positive_class": labels_json[1],
                    "fpr": fx,
                    "tpr": fy,
                }
            except ValueError:
                result["roc"] = None
        return result

    # regression
    residuals = np.asarray(y_test.to_numpy(dtype=float)) - np.asarray(y_pred, dtype=float)
    metrics = _compute_regression_metrics(y_test, y_pred)
    return {
        "metrics": {k: _rnd(v) for k, v in metrics.items()},
        "residuals": {
            "count": int(len(residuals)),
            "mean": _rnd(float(np.mean(residuals))),
            "std": _rnd(float(np.std(residuals))),
            "min": _rnd(float(np.min(residuals))),
            "max": _rnd(float(np.max(residuals))),
        },
        "scatter": _sample_pairs(
            y_test.to_numpy(dtype=float),
            np.asarray(y_pred, dtype=float),
            settings.ml_scatter_max_points,
        ),
    }


# ============================================================
# 公开接口：训练 / 列表 / 详情 / 删除 / 预测
# ============================================================


def run_training(dataset_id: str, request: MLTrainRequest) -> dict:
    """执行一次完整训练并持久化实验；任何一步失败都不产生实验文件。"""
    # 1) 加载所选版本数据（同时校验 Session 与版本合法性）
    frame, version_info = vm.load_dataset_version(dataset_id, request.source_version_id)
    source_version_id = str(version_info.get("version_id") or "original")
    if len(frame) > settings.ml_max_rows:
        raise _merror(
            "too_many_rows",
            f"数据集共 {len(frame)} 行，超过单次训练上限 "
            f"{settings.ml_max_rows} 行。请先采样或筛选后再训练。",
            422,
        )

    warnings: list[str] = []
    leaked, leaky_ops, leak_warnings = _leakage_check(version_info)
    warnings.extend(leak_warnings)

    # 2) 目标解析（auto / 显式）
    target = str(request.target_column).strip()
    if not target or target not in frame.columns:
        raise _merror(
            "column_not_found",
            f"目标字段 {target} 不存在于当前数据版本中。",
        )
    target_info = _inspect_column(frame[target], target)
    task_type, task_reason = _resolve_task_type(request.task_type, target_info, target)

    # 3) 剔除目标缺失行（目标列缺失不参与训练）
    clean_frame, y, dropped_missing_target_rows = _clean_target_rows(
        frame, target, task_type
    )

    # 4) 特征选择
    feature_request = (
        [str(c) for c in request.feature_columns]
        if request.feature_columns is not None
        else None
    )
    (
        _info_by_col,
        numeric_cols,
        categorical_cols,
        excluded,
        feature_warnings,
    ) = _feature_plan(clean_frame, target, feature_request, request.exclude_columns)
    warnings.extend(feature_warnings)
    features = numeric_cols + categorical_cols

    # 5) train / test split（stratify 仅分类且类别足够时启用）
    requested_cv = int(request.cv_folds)
    random_state = int(request.random_state)
    test_size = float(request.test_size)
    stratify = None
    if task_type == "classification":
        class_counts = y.value_counts()
        if int(class_counts.min()) < 2:
            raise _merror(
                "not_enough_samples_per_class",
                "某些类别样本数不足 2 个，无法进行分层划分与交叉验证。",
                422,
            )
        stratify = y
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            clean_frame[features],
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        # 分层失败（极端类别分布）回退为普通随机切分
        X_train, X_test, y_train, y_test = train_test_split(
            clean_frame[features],
            y,
            test_size=test_size,
            random_state=random_state,
        )
        warnings.append(
            "数据类别分布特殊，train/test 分层切分失败，已回退为随机切分。"
        )

    # 6) 有效折数：分类按最小类别数收敛，回归按样本量收敛
    if task_type == "classification":
        effective_folds = min(requested_cv, int(y_train.value_counts().min()))
    else:
        effective_folds = min(requested_cv, int(len(y_train)))
    if effective_folds < 2:
        raise _merror(
            "cv_folds_unavailable",
            "交叉验证折数不足（有效折数 < 2）。请增大样本量或减少类别数。",
            422,
        )

    # 7) 候选模型（校验任务匹配）
    model_keys = _validate_candidate_models(request.candidate_models, task_type)

    # 8) 训练集内 CV
    cv_results, model_errors = _run_cv(
        model_keys,
        task_type,
        numeric_cols,
        categorical_cols,
        X_train,
        y_train,
        effective_folds,
        random_state,
    )
    if not cv_results:
        raise _merror(
            "all_models_failed",
            "所有候选模型均训练失败，未保存任何实验。"
            f"最后错误：{_json_safe(list(model_errors.values())[-1])}",
            422,
        )
    for key, err in model_errors.items():
        warnings.append(f"候选模型 {key} 训练失败（已跳过）：{err}")

    primary_metric = _PRIMARY_METRICS[task_type]
    maximize = task_type == "classification"
    ranking = _rank_models(cv_results, primary_metric, maximize)
    if not ranking:
        raise _merror("all_models_failed", "候选模型指标计算失败，未保存实验。", 422)

    best_key = ranking[0]["model"]
    best_primary_mean = float(ranking[0]["primary_mean"])
    baseline_mean = None
    if "dummy" in cv_results:
        baseline_mean = cv_results["dummy"]["metrics"][primary_metric].get("mean")
    beats_baseline = False
    improvement_over_baseline = None
    if baseline_mean is not None:
        if maximize:
            beats_baseline = best_primary_mean > float(baseline_mean)
            improvement_over_baseline = best_primary_mean - float(baseline_mean)
        else:
            beats_baseline = best_primary_mean < float(baseline_mean)
            if float(baseline_mean) != 0:
                improvement_over_baseline = (
                    float(baseline_mean) - best_primary_mean
                ) / abs(float(baseline_mean))
    if not beats_baseline:
        warnings.append(
            f"最佳模型 {best_key} 的{('f1_macro' if maximize else 'RMSE')} "
            f"({_rnd(best_primary_mean)}) 未明确优于 Dummy 基线"
            + (f"（基线 {_rnd(baseline_mean)}）" if baseline_mean is not None else "")
            + "；数据可能缺乏可学模式，请谨慎对待预测结果。"
        )

    # 9) 在完整训练集上 refit 最佳模型（评估前的唯一一次正式拟合）
    best_pipeline = _build_pipeline(
        best_key, task_type, numeric_cols, categorical_cols, random_state
    )
    best_pipeline.fit(X_train, y_train)
    feature_names_out = extract_feature_names(best_pipeline)
    if not feature_names_out:
        warnings.append("无法从 Pipeline 提取编码后特征名（feature_names_out 不可用）。")

    # 10) 最终评估 untouched Test Set
    test_eval = _evaluate_test(best_pipeline, task_type, X_test, y_test)

    # 11) 组装元数据与评估结果
    class_labels: list = []
    if task_type == "classification":
        class_labels = test_eval.get("class_labels") or []

    primary_test_score = test_eval["metrics"].get(primary_metric)
    test_score_summary: dict = {"primary_metric": primary_metric, "score": _rnd(primary_test_score)}
    if task_type == "classification":
        test_score_summary["accuracy"] = test_eval["metrics"].get("accuracy")
        test_score_summary["balanced_accuracy"] = test_eval["metrics"].get(
            "balanced_accuracy"
        )
    else:
        test_score_summary["r2"] = test_eval["metrics"].get("r2")
        test_score_summary["mae"] = test_eval["metrics"].get("mae")

    experiment_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    metadata = {
        "experiment_id": experiment_id,
        "dataset_id": dataset_id,
        "source_version_id": source_version_id,
        "dataset_sha256": _frame_sha256(frame),
        "app_version": ML_APP_VERSION,
        "sklearn_version": sklearn.__version__,
        "created_at": created_at,
        "target_column": target,
        "task_type": task_type,
        "task_type_reason": task_reason,
        "preprocessing_fit_scope": "train_only",
        "total_rows": len(clean_frame),
        "dropped_missing_target_rows": dropped_missing_target_rows,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_size": round(test_size, 4),
        "random_state": random_state,
        "requested_cv_folds": requested_cv,
        "effective_cv_folds": effective_folds,
        "features": features,
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "excluded_columns": excluded,
        "candidate_models": model_keys,
        "primary_metric": primary_metric,
        "best_model": best_key,
        "best_cv_primary_mean": _rnd(best_primary_mean),
        "baseline_primary_mean": _rnd(baseline_mean),
        "beats_baseline": bool(beats_baseline),
        "improvement_over_baseline": _rnd(improvement_over_baseline),
        "cv_ranking": [
            {
                "model": item["model"],
                "primary_mean": _rnd(item["primary_mean"]),
            }
            for item in ranking
        ],
        "class_labels": class_labels,
        "feature_names_out": feature_names_out,
        "potential_data_leakage": bool(leaked),
        "leakage_sensitive_operations": leaky_ops,
        "test_score_summary": test_score_summary,
        "warnings": warnings,
    }

    evaluation = {
        "cv": {
            "primary_metric": primary_metric,
            "best_model": best_key,
            "best_primary_mean": _rnd(best_primary_mean),
            "beats_baseline": bool(beats_baseline),
            "model_errors": model_errors,
            "results": cv_results,
            "ranking": metadata["cv_ranking"],
        },
        "test": test_eval,
        "model_errors": model_errors,
    }

    # 12) 原子持久化（model.joblib + metadata.json + evaluation.json）
    em.persist_experiment(
        dataset_id=dataset_id,
        experiment_id=experiment_id,
        pipeline=best_pipeline,
        metadata=metadata,
        evaluation=evaluation,
    )
    response = dict(metadata)
    response.update(evaluation)
    response["success"] = True
    return response


def _frame_sha256(frame: pd.DataFrame) -> str:
    """对版本数据做规范化 SHA-256，用于实验可追溯。"""
    canonical = frame.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def list_experiments(dataset_id: str) -> list[dict]:
    """实验列表（created_at 降序）。"""
    return em.list_experiments(dataset_id)


def get_experiment_detail(dataset_id: str, experiment_id: str) -> dict:
    """实验详情：metadata + evaluation 的合并结果。"""
    return em.get_experiment(dataset_id, experiment_id)


def delete_experiment(dataset_id: str, experiment_id: str) -> None:
    """删除实验及其模型文件。"""
    em.delete_experiment(dataset_id, experiment_id)


def run_prediction(
    dataset_id: str, experiment_id: str, request: MLPredictRequest
) -> dict:
    """用已保存实验 Pipeline 对新记录批量预测（不重新训练 / 不重新 fit）。"""
    if len(request.records) > settings.ml_max_prediction_records:
        raise _merror(
            "too_many_records",
            f"预测记录数 {len(request.records)} 超过单次上限 "
            f"{settings.ml_max_prediction_records} 条。",
            422,
        )
    metadata = em.get_experiment(dataset_id, experiment_id)
    features: list[str] = [str(c) for c in (metadata.get("features") or [])]
    if not features:
        raise _merror(
            "experiment_invalid",
            "实验元数据缺少特征信息，无法执行预测。",
            500,
        )
    task_type = str(metadata.get("task_type") or "classification")

    warnings: list[str] = []
    feature_set = set(features)
    extra_fields: set[str] = set()
    for index, record in enumerate(request.records, start=1):
        unknown = [str(f) for f in record.keys() if str(f) not in feature_set]
        if unknown:
            extra_fields.update(unknown)
    if extra_fields:
        warnings.append(
            "以下字段不在训练特征中，已忽略："
            + "、".join(sorted(extra_fields)[:8])
            + (" 等" if len(extra_fields) > 8 else "")
            + "。"
        )

    rows: list[dict] = []
    missing_fields: set[str] = set()
    for record in request.records:
        missing = [f for f in features if f not in record]
        if missing:
            missing_fields.update(missing)
            continue
        rows.append({f: record[f] for f in features})
    if missing_fields:
        raise _merror(
            "missing_feature_columns",
            "部分记录缺少训练必需字段：" + "、".join(sorted(missing_fields)[:8])
            + "。请补全字段后重试。",
            422,
        )

    X_new = pd.DataFrame(rows, columns=features)
    pipeline = em.load_experiment_model(dataset_id, experiment_id)
    model = pipeline.named_steps["model"]

    predictions_raw = pipeline.predict(X_new)
    result: dict = {
        "success": True,
        "dataset_id": dataset_id,
        "experiment_id": experiment_id,
        "task_type": task_type,
        "count": len(rows),
        "features": features,
        "warnings": warnings,
    }

    if task_type == "classification":
        class_labels = [_json_safe(c) for c in model.classes_]
        result["class_labels"] = class_labels
        predictions = [
            {"index": i, "prediction": _json_safe(raw)}
            for i, raw in enumerate(predictions_raw)
        ]
        result["predictions"] = predictions
        if hasattr(model, "predict_proba"):
            proba = pipeline.predict_proba(X_new)
            result["probabilities"] = [
                {class_labels[j]: _rnd(float(row[j])) for j in range(len(class_labels))}
                for row in proba
            ]
        else:  # pragma: no cover - 当前分类模型均支持概率
            result["probabilities"] = None
    else:
        result["class_labels"] = []
        result["predictions"] = [
            {"index": i, "prediction": _rnd(float(raw))}
            for i, raw in enumerate(predictions_raw)
        ]
        result["probabilities"] = None
    return result
