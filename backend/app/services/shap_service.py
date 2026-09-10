"""v0.6 模型可解释性（SHAP）服务。

设计目标：
- 复用 v0.5 已落盘的完整 Pipeline（preprocessing + estimator）作为 SHAP 解释对象；
- 不修改任何 v0.5 模型文件，仅在 experiment 目录下新增 shap_result.json；
- 沿用 v0.5 的「Dataset Session」磁盘布局，所有产物始终位于
  ``backend/runtime/``（即 D:\\ai\\AI-Data-Platform\\backend\\runtime\\），
  绝不写入 C 盘。

SHAP 解释对象选择策略（按 estimator 类型自动路由）：

1. 树模型（RandomForestClassifier / RandomForestRegressor） → ``shap.TreeExplainer``
2. 线性模型（LogisticRegression / Ridge）                      → ``shap.LinearExplainer``
3. 其余（DummyClassifier / DummyRegressor / 其它）             → ``shap.KernelExplainer``
   背景摘要采用 kmeans 摘要（最多 ``ML_SHAP_BACKGROUND_SIZE``）以控制耗时。

分类任务约定：始终对 ``class_labels[1]``（即「正类」）计算 SHAP 值；
base_value 同样选取 ``expected_value[1]``。这样二分类与多分类走同一条
输出路径，前端无需为不同类别数写分支。
"""

from __future__ import annotations

import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

from app.core.config import settings
from app.services import experiment_manager as em
from app.services import version_manager as vm

logger = logging.getLogger(__name__)

EXPLAINABILITY_APP_VERSION = "0.6.0"
_KERNEL_NSAMPLES = 100  # KernelExplainer 每条样本的采样次数（控制精度 / 速度）


# ============================================================
# 业务异常
# ============================================================


class ExplainabilityError(Exception):
    """SHAP 解释服务的业务异常（code / message / HTTP status）。"""

    def __init__(self, code: str, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def _err(code: str, message: str, status_code: int = 400) -> ExplainabilityError:
    return ExplainabilityError(code, message, status_code)


# ============================================================
# JSON 安全 / 数值处理
# ============================================================


def _json_safe(value):
    if value is None:
        return None
    if isinstance(value, (str, bool)):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        n = float(value)
        return n if math.isfinite(n) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.ndarray, list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (AttributeError, ValueError):
            pass
    return str(value)


def _rnd(value, digits: int = 6):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, digits)


# ============================================================
# 公共入口
# ============================================================


def run_explain(experiment_id: str, request) -> dict:
    """为已落盘的 experiment 生成 / 读取 SHAP 解释。"""
    if not experiment_id:
        raise _err("invalid_experiment_id", "experiment_id 不能为空。", 400)
    try:
        normalized = em.normalize_experiment_id(experiment_id)
    except em.ExperimentError as exc:
        raise _err(exc.code, exc.message, exc.status_code) from exc

    location = em.find_experiment_location(normalized)
    if not location:
        raise _err(
            "experiment_not_found",
            "实验不存在或已被删除，请重新训练后再生成解释。",
            404,
        )
    dataset_id, exp_dir = location

    metadata = em.get_experiment(dataset_id, normalized)
    # 缓存命中：除非 caller 显式要求 regenerate
    if not getattr(request, "regenerate", False):
        cached = em.read_shap_result(exp_dir)
        if cached:
            cached.setdefault("success", True)
            cached["cached"] = True
            cached["experiment_id"] = normalized
            cached["dataset_id"] = dataset_id
            return cached

    # 关键路径：模型文件缺失（存储损坏）应明确报错
    if not (exp_dir / em.MODEL_FILE).is_file():
        raise _err(
            "experiment_model_missing",
            "实验目录中缺少模型文件，无法生成 SHAP 解释。",
            500,
        )

    pipeline = em.load_experiment_model(dataset_id, normalized)
    source_version_id = str(metadata.get("source_version_id") or "original")
    try:
        frame, _version_info = vm.load_dataset_version(dataset_id, source_version_id)
    except Exception as exc:  # noqa: BLE001 - 源版本缺失时给清晰错误
        logger.warning("source version unavailable for %s: %s", normalized, exc)
        raise _err(
            "source_version_unavailable",
            "实验所引用的源数据版本已不可用，请删除实验后重新训练。",
            422,
        ) from exc

    features = [str(c) for c in (metadata.get("features") or [])]
    if not features:
        raise _err(
            "experiment_invalid",
            "实验元数据缺少训练特征列表，无法生成 SHAP 解释。",
            500,
        )
    if any(f not in frame.columns for f in features):
        missing = [f for f in features if f not in frame.columns]
        raise _err(
            "experiment_invalid",
            f"源版本缺少实验训练时使用的特征字段：{', '.join(missing)}。",
            500,
        )
    X_raw = frame[features].copy()

    preprocessor = _extract_preprocessor(pipeline)
    model = pipeline.named_steps["model"]
    task_type = str(metadata.get("task_type") or "classification")
    class_labels = [_json_safe(c) for c in (metadata.get("class_labels") or [])]
    class_index_used = 1 if (task_type == "classification" and len(class_labels) > 1) else 0
    class_label_used = class_labels[class_index_used] if class_labels else None

    # 转换 + 采样
    X_transformed = preprocessor.transform(X_raw)
    n_total = int(X_transformed.shape[0])
    n_features_post = int(X_transformed.shape[1])
    if n_total < 1:
        raise _err("empty_dataset", "源版本无可用行，无法生成 SHAP 解释。", 422)
    if n_total > settings.ml_max_rows:
        # 超过 ML 上限时仍允许 SHAP（仅采样部分行），给出 warning
        pass
    max_rows = int(getattr(request, "max_summary_rows", None) or settings.ml_shap_sample_size)
    max_rows = max(1, min(max_rows, settings.ml_shap_sample_size * 5))
    sample_size = min(n_total, max_rows)
    rng = np.random.RandomState(int(metadata.get("random_state") or settings.ml_default_random_state))
    if sample_size < n_total:
        sample_indices = np.sort(rng.choice(n_total, size=sample_size, replace=False))
    else:
        sample_indices = np.arange(n_total, dtype=int)
    X_sample = X_transformed[sample_indices]
    X_raw_sample = X_raw.iloc[sample_indices].reset_index(drop=True)
    sample_index_list = [int(i) for i in sample_indices]

    # 特征名（优先使用 v0.5 已落盘的 feature_names_out）
    feature_names = _resolve_feature_names(metadata, preprocessor, n_features_post)

    warnings: list[str] = []
    if isinstance(model, (DummyClassifier, DummyRegressor)):
        warnings.append(
            "当前最佳模型为 Dummy 基线（多数类 / 均值预测），所有特征的 SHAP 贡献都接近 0，"
            "建议训练真正的候选模型以获得有意义的解释。"
        )

    # 选择 explainer 并计算 SHAP
    start = time.perf_counter()
    try:
        explainer, explainer_type, expected_value_raw = _build_explainer(
            model=model,
            X_background=X_sample,
            task_type=task_type,
        )
        shap_values = _compute_shap_values(
            explainer=explainer,
            model=model,
            X_sample=X_sample,
            task_type=task_type,
        )
    except Exception as exc:  # noqa: BLE001 - SHAP 计算异常统一转 500
        logger.exception("SHAP computation failed for experiment %s", normalized)
        raise _err(
            "shap_computation_failed",
            f"SHAP 计算失败：{exc}",
            500,
        ) from exc
    elapsed = time.perf_counter() - start
    if elapsed > settings.ml_shap_timeout_seconds:
        warnings.append(
            f"SHAP 计算耗时 {elapsed:.1f}s 超过建议阈值 "
            f"{settings.ml_shap_timeout_seconds}s，建议减少采样行数或换用更快的 explainer。"
        )

    sv_2d, base_value_used = _select_class_shap(shap_values, expected_value_raw, class_index_used, n_features_post)
    n_rows_used = int(sv_2d.shape[0])
    importance = np.abs(sv_2d).mean(axis=0)

    # summary.values 使用 post-preprocessing 矩阵，与 feature_names / shap_values 同维对齐
    # （原始值在 samples 解释中作为 raw_value 提供，方便人读）
    summary_values_json = [
        [_json_safe(v) for v in row] for row in X_sample.tolist()
    ]
    summary_shap_json = [
        [_rnd(v) for v in row] for row in sv_2d.tolist()
    ]
    summary_raw_values_json = [
        [_json_safe(v) for v in row] for row in X_raw_sample.values.tolist()
    ]

    # 单点解释：默认 3 个；按 caller 提供的 sample_indices 过滤
    sample_explanations = _build_sample_explanations(
        sv_2d=sv_2d,
        X_raw_sample=X_raw_sample,
        X_transformed_sample=X_sample,
        model=model,
        feature_names=feature_names,
        task_type=task_type,
        class_labels=class_labels,
        class_index_used=class_index_used,
        requested_indices=list(getattr(request, "sample_indices", None) or []),
    )

    top_k = min(settings.ml_shap_top_features, len(feature_names))
    top_pairs = sorted(
        zip(feature_names, [_rnd(x) for x in importance]),
        key=lambda item: (item[1] is None, -(item[1] or 0.0)),
    )[:top_k]
    top_feature_names = [name for name, _ in top_pairs]
    top_importance = [imp for _, imp in top_pairs]

    created_at = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    response = {
        "success": True,
        "experiment_id": normalized,
        "dataset_id": dataset_id,
        "source_version_id": source_version_id,
        "task_type": task_type,
        "best_model": metadata.get("best_model"),
        "model_class": type(model).__name__,
        "explainer_type": explainer_type,
        "class_label_used": _json_safe(class_label_used),
        "class_labels": class_labels,
        "base_value": _rnd(base_value_used),
        "n_features_post_preprocessing": n_features_post,
        "feature_names": feature_names,
        "global": {
            "feature_names": feature_names,
            "importance": [_rnd(x) for x in importance],
            "n_rows_used": n_rows_used,
            "top_feature_names": top_feature_names,
            "top_importance": top_importance,
        },
        "summary": {
            "feature_names": feature_names,
            "sample_indices": sample_index_list,
            "values": summary_values_json,
            "raw_values": summary_raw_values_json,
            "raw_feature_names": features,
            "shap_values": summary_shap_json,
        },
        "samples": sample_explanations,
        "warnings": warnings,
        "created_at": created_at,
        "app_version": EXPLAINABILITY_APP_VERSION,
        "cached": False,
    }

    em.write_shap_result(
        exp_dir,
        response,
        {
            "created_at": created_at,
            "explainer_type": explainer_type,
            "model_class": type(model).__name__,
            "task_type": task_type,
            "class_label_used": response["class_label_used"],
            "n_rows_used": n_rows_used,
            "warnings": warnings,
        },
    )
    logger.info(
        "SHAP explainer=%s rows=%d features=%d in %.1fs for experiment %s",
        explainer_type,
        n_rows_used,
        n_features_post,
        elapsed,
        normalized,
    )
    return response


# ============================================================
# 内部：Pipeline 拆分 / explainer 路由 / SHAP 归一化
# ============================================================


def _extract_preprocessor(pipeline: Pipeline) -> Pipeline:
    """从完整 Pipeline 中提取出「model 步骤之前」的预处理子 Pipeline。"""
    steps = []
    for name, step in pipeline.steps:
        if name == "model":
            break
        steps.append((name, step))
    if not steps:
        raise _err("pipeline_invalid", "Pipeline 缺少预处理步骤。", 500)
    return Pipeline(steps=steps)


def _resolve_feature_names(metadata: dict, preprocessor: Pipeline, n_features_post: int) -> list[str]:
    """优先复用 v0.5 已写入的 feature_names_out，失败则重新提取。"""
    stored = metadata.get("feature_names_out") or []
    if isinstance(stored, list) and len(stored) == n_features_post and all(
        isinstance(x, str) for x in stored
    ):
        return [str(x) for x in stored]
    # 兜底：尝试从 ColumnTransformer 直接提取
    try:
        step = preprocessor
        if hasattr(step, "named_steps") and "preprocessing" in step.named_steps:
            step = step.named_steps["preprocessing"]
        return [str(x) for x in step.get_feature_names_out()]
    except Exception:  # noqa: BLE001
        return [f"feature_{i}" for i in range(n_features_post)]


def _build_explainer(model, X_background: np.ndarray, task_type: str):
    """根据 estimator 类型选择 explainer。

    返回 (explainer, explainer_type, expected_value_raw)。
    """
    import shap

    if isinstance(model, (RandomForestClassifier, RandomForestRegressor)):
        return shap.TreeExplainer(model), "tree", np.asarray(
            shap.TreeExplainer(model).expected_value
        )

    if isinstance(model, (LogisticRegression, Ridge)):
        try:
            # LinearExplainer 接受 (model, data) 走 interventional 路径
            explainer = shap.LinearExplainer(model, X_background)
        except Exception:  # noqa: BLE001 - 旧版本 shap 兼容性兜底
            explainer = shap.LinearExplainer(model, X_background, feature_perturbation="interventional")
        return explainer, "linear", np.asarray(explainer.expected_value)

    # 兜底：KernelExplainer（Dummy 或其它）
    background = _summarize_background(X_background, settings.ml_shap_background_size)
    if task_type == "classification" and hasattr(model, "predict_proba"):
        explainer = shap.KernelExplainer(model.predict_proba, background)
    else:
        explainer = shap.KernelExplainer(model.predict, background)
    return explainer, "kernel", np.asarray(explainer.expected_value)


def _summarize_background(X: np.ndarray, n: int) -> np.ndarray:
    """对背景矩阵做 kmeans 摘要（sklearn KMeans）。失败时退回到随机抽样。"""
    if X.shape[0] <= n:
        return X
    try:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=n, random_state=0, n_init=3)
        km.fit(X)
        return km.cluster_centers_.astype(X.dtype, copy=False)
    except Exception:  # noqa: BLE001
        rng = np.random.RandomState(0)
        idx = rng.choice(X.shape[0], size=n, replace=False)
        return X[idx]


def _compute_shap_values(explainer, model, X_sample: np.ndarray, task_type: str):
    """调用 explainer 计算 SHAP 值，统一不同 explainer 的输出差异。"""
    if isinstance(explainer, type(explainer)) and explainer.__class__.__name__ == "TreeExplainer":
        # TreeExplainer：shap_values 对分类任务返回 list 或 3D
        return explainer.shap_values(X_sample)
    if explainer.__class__.__name__ == "LinearExplainer":
        return explainer.shap_values(X_sample)
    # KernelExplainer
    if task_type == "classification" and hasattr(model, "predict_proba"):
        return explainer.shap_values(X_sample, nsamples=_KERNEL_NSAMPLES)
    return explainer.shap_values(X_sample, nsamples=_KERNEL_NSAMPLES)


def _select_class_shap(shap_values, expected_value, class_index: int, n_features: int):
    """把各种格式的 SHAP 输出归一为 2D (n_samples, n_features) 并返回 base_value。"""
    sv = np.asarray(shap_values)
    if sv.ndim == 2:
        sv_2d = sv
    elif sv.ndim == 3:
        # (n_samples, n_features, n_classes) — 较新 shap 统一格式
        sv_2d = sv[..., class_index]
    elif isinstance(shap_values, list):
        sv_2d = np.asarray(shap_values[class_index])
    else:
        sv_2d = sv.reshape((-1, n_features))
    ev = np.asarray(expected_value).flatten()
    if ev.size == 0:
        base_value = 0.0
    elif ev.size == 1:
        base_value = float(ev[0])
    else:
        idx = min(class_index, ev.size - 1)
        base_value = float(ev[idx])
    return sv_2d, base_value


def _build_sample_explanations(
    sv_2d: np.ndarray,
    X_raw_sample: pd.DataFrame,
    X_transformed_sample: np.ndarray,
    model,
    feature_names: list[str],
    task_type: str,
    class_labels: list,
    class_index_used: int,
    requested_indices: list[int],
) -> list[dict]:
    """生成单点解释（默认 3 个，或按 caller 指定的索引）。"""
    n_rows = int(sv_2d.shape[0])
    if n_rows == 0:
        return []

    if requested_indices:
        indices = []
        for sidx in requested_indices:
            if 0 <= int(sidx) < n_rows:
                indices.append(int(sidx))
        # 同一索引去重并保持顺序
        seen = set()
        deduped = []
        for idx in indices:
            if idx not in seen:
                deduped.append(idx)
                seen.add(idx)
        indices = deduped[:5]
        if not indices:
            indices = [0]
    else:
        # 默认：第 0 / 中间 / 最后 各一个（最多 3 个）
        default = [0, n_rows // 2, n_rows - 1]
        indices = []
        for idx in default:
            if 0 <= idx < n_rows and idx not in indices:
                indices.append(int(idx))
        indices = indices[:3]

    # 预测：使用 model 直接对 post-preprocessing 矩阵预测
    explanations: list[dict] = []
    for idx in indices:
        contributions = []
        row_sv = sv_2d[idx]
        for j, fname in enumerate(feature_names):
            raw_value = X_raw_sample.iloc[idx, j] if j < X_raw_sample.shape[1] else None
            contributions.append(
                {
                    "feature": fname,
                    "value": _json_safe(raw_value),
                    "shap": _rnd(float(row_sv[j])),
                }
            )
        contributions_sorted = sorted(
            contributions,
            key=lambda item: (item["shap"] is None, -abs(item["shap"] or 0.0)),
        )
        try:
            if task_type == "classification" and hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_transformed_sample[idx : idx + 1])[0]
                if class_labels:
                    pred_label = class_labels[class_index_used]
                    if class_index_used < len(proba):
                        pred_value = _rnd(float(proba[class_index_used]))
                    else:
                        pred_value = None
                else:
                    pred_label = _json_safe(proba.argmax())
                    pred_value = _rnd(float(proba.max()))
                prediction = {
                    "kind": "classification",
                    "class_label": _json_safe(pred_label),
                    "probability": pred_value,
                }
            else:
                pred = model.predict(X_transformed_sample[idx : idx + 1])[0]
                prediction = {
                    "kind": "regression",
                    "value": _json_safe(pred),
                }
        except Exception as exc:  # noqa: BLE001
            logger.warning("sample %d prediction failed: %s", idx, exc)
            prediction = {"kind": "unknown", "error": str(exc)}

        # base_value 在每条 sample 解释中重复返回（方便前端免对齐）
        # 通过 _select_class_shap 在外层拿到，这里直接通过 row 的 sum 校验一致性
        sample_base = float(row_sv.sum())  # 临时，后续由 caller 注入更稳的值
        explanations.append(
            {
                "index": int(idx),
                "sample_source_index": int(X_raw_sample.index[idx]) if hasattr(X_raw_sample, "index") else int(idx),
                "prediction": prediction,
                "base_value_delta": _rnd(sample_base),  # 本样本 SHAP 之和（对分类表示 log-odds 偏移）
                "contributions": contributions,
                "top_contributions": contributions_sorted[: min(10, len(contributions_sorted))],
            }
        )
    return explanations
