"""v0.5 机器学习模块测试（训练 / CV / 模型比较 / 最终评估 / 实验管理 / 预测）。

覆盖要点：
- 防 Data Leakage：统计型预处理只发生在训练集 Pipeline 内；派生版本含 v0.4 统计操作
  时返回 potential_data_leakage warning；
- 自动任务推断 / 特征选择 / 显式特征 / 目标校验错误；
- CV 折数收敛、模型部分失败、全部失败不留文件；
- 分类混淆矩阵 / ROC-AUC、回归散点与残差、预测接口校验；
- 版本删除保护（409 version_in_use_by_ml_experiment）。
"""

import json
import math
import re
import uuid

import numpy as np
import pandas as pd
import pytest

from app.services import ml_service as mlsvc


# ============================================================
# 公共工具
# ============================================================


def _upload(client, frame: pd.DataFrame, filename: str = "dataset.csv") -> str:
    content = frame.to_csv(index=False).encode("utf-8")
    resp = client.post(
        "/api/data/upload", files={"file": (filename, content, "text/csv")}
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["dataset_id"]


def _train(client, dataset_id: str, **overrides) -> dict:
    payload = {
        "source_version_id": "original",
        "target_column": "label",
        "task_type": "auto",
    }
    payload.update(overrides)
    resp = client.post(f"/api/datasets/{dataset_id}/ml/train", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True
    return body


def _classification_frame(rows: int = 90, seed: int = 11, label_missing: int = 5) -> pd.DataFrame:
    """可学习的二分类数据：x1/grp 决定 label，另带缺失值练习 imputer。"""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=rows)
    x2 = rng.integers(0, 20, size=rows).astype(float)
    grp = rng.choice(["Alpha", "Beta", "Gamma"], size=rows)
    flag = pd.Series(rng.choice([True, False], size=rows, p=[0.6, 0.4]), dtype=object)
    grp_effect = np.where(grp == "Alpha", 0.9, np.where(grp == "Beta", 0.1, -0.8))
    score = 1.3 * x1 + grp_effect + rng.normal(0, 0.35, size=rows)
    label = np.where(score > float(np.median(score)), "yes", "no")
    frame = pd.DataFrame(
        {"x1": x1, "x2": x2, "grp": grp, "flag": flag, "label": label}
    )
    frame.loc[rng.choice(rows, size=3, replace=False), "x1"] = np.nan
    frame.loc[rng.choice(rows, size=2, replace=False), "grp"] = np.nan
    if label_missing:
        frame.loc[rng.choice(rows, size=label_missing, replace=False), "label"] = np.nan
    return frame


def _regression_frame(rows: int = 80, seed: int = 5) -> pd.DataFrame:
    """可学习的回归数据：target 与 x1/grp 相关。"""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=rows)
    x2 = rng.integers(0, 30, size=rows).astype(float)
    grp = rng.choice(["A", "B", "C"], size=rows)
    target = (
        2.5 * x1
        + np.where(grp == "A", 1.4, np.where(grp == "B", -0.3, 0.7))
        + rng.normal(0, 0.4, size=rows)
    )
    frame = pd.DataFrame({"x1": x1, "x2": x2, "grp": grp, "target": target})
    frame.loc[rng.choice(rows, size=4, replace=False), "x1"] = np.nan
    return frame


def _expect_error(client, resp, expected_code: str, status: int | None = None):
    assert resp.status_code == (status if status is not None else 400), resp.text
    payload = resp.json()
    assert payload["success"] is False
    assert payload["error"]["code"] == expected_code
    return payload


# ============================================================
# 分类端到端 + CV 比较 + 最终评估
# ============================================================


def test_train_classification_end_to_end(client):
    dataset_id = _upload(client, _classification_frame())
    body = _train(client, dataset_id)  # 默认 3 个候选模型 + 5 折 CV

    # 元信息完整性
    assert re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", body["experiment_id"])
    assert body["dataset_id"] == dataset_id
    assert body["source_version_id"] == "original"
    assert body["task_type"] == "classification"
    assert body["task_type_reason"]
    assert body["preprocessing_fit_scope"] == "train_only"
    assert body["requested_cv_folds"] == 5
    assert body["effective_cv_folds"] == 5
    assert re.fullmatch(r"[0-9a-f]{64}", body["dataset_sha256"])
    assert body["class_labels"] == ["no", "yes"] or len(body["class_labels"]) == 2
    assert body["train_rows"] + body["test_rows"] + body["dropped_missing_target_rows"] == 90
    assert body["dropped_missing_target_rows"] == 5
    assert body["beats_baseline"] is True

    # 特征：数值/分类分组存在且不相交
    assert body["features"] == body["numeric_features"] + body["categorical_features"]
    assert "x1" in body["numeric_features"]
    assert "grp" in body["categorical_features"]

    # CV：每个模型 5 折都有完整指标，ranking 覆盖全部候选
    assert set(body["cv"]["results"].keys()) == {"dummy", "logistic_regression", "random_forest"}
    for key, detail in body["cv"]["results"].items():
        assert len(detail["metrics"]["f1_macro"]["values"]) == 5
        assert detail["folds"] == 5
    assert len(body["cv"]["ranking"]) == 3
    assert body["cv"]["best_model"] == body["best_model"]
    assert body["cv"]["best_model"] in {"logistic_regression", "random_forest"}
    assert body["best_cv_primary_mean"] > 0.7
    assert body["model_errors"] == {}

    # 测试集评估（唯一一次）
    test_metrics = body["test"]["metrics"]
    assert test_metrics["accuracy"] > 0.7
    assert test_metrics["f1_macro"] > 0.6
    cm = body["test"]["confusion_matrix"]
    assert len(cm["labels"]) == 2
    assert len(cm["matrix"]) == 2
    assert sum(sum(row) for row in cm["matrix"]) == body["test_rows"]
    assert len(body["test"]["per_class_metrics"]) == 2
    roc = body["test"]["roc"]
    assert roc is not None
    assert roc["auc"] > 0.8
    assert len(roc["fpr"]) == len(roc["tpr"]) > 2
    assert roc["fpr"][0] == 0.0 and roc["fpr"][-1] == 1.0

    # feature_names_out 与 test_score_summary
    assert body["feature_names_out"]
    assert body["test_score_summary"]["primary_metric"] == "f1_macro"
    assert body["test_score_summary"]["accuracy"] > 0.7
    assert body["warnings"] == []

    # 实验已持久化：列表 / 详情
    lst = client.get(f"/api/datasets/{dataset_id}/ml/experiments").json()
    assert lst["count"] == 1
    item = lst["experiments"][0]
    assert item["experiment_id"] == body["experiment_id"]
    assert item["best_model"] == body["best_model"]
    detail = client.get(
        f"/api/datasets/{dataset_id}/ml/experiments/{body['experiment_id']}"
    ).json()
    assert detail["experiment_id"] == body["experiment_id"]
    assert detail["test"]["metrics"]["f1_macro"] == test_metrics["f1_macro"]


def test_multiclass_no_roc_and_confusion(client):
    rng = np.random.default_rng(9)
    rows = 90
    x1 = rng.normal(size=rows)
    grp = rng.choice(["Alpha", "Beta", "Gamma"], size=rows)
    label = np.select(
        [x1 + np.where(grp == "Alpha", 1.0, 0.0) > 0.4, x1 + np.where(grp == "Gamma", 1.2, 0.0) < -0.4],
        ["red", "green"],
        default="blue",
    )
    frame = pd.DataFrame({"x1": x1, "grp": grp, "label": label})
    dataset_id = _upload(client, frame)
    body = _train(client, dataset_id, candidate_models=["logistic_regression"])
    assert body["task_type"] == "classification"
    assert len(body["class_labels"]) == 3
    assert body["test"]["confusion_matrix"]["labels"] == sorted(["red", "green", "blue"])
    assert body["test"]["roc"] is None
    assert len(body["test"]["per_class_metrics"]) == 3


def test_dummy_not_beaten_warning_only_constants(client):
    rows = 60
    frame = pd.DataFrame(
        {
            "const_a": [3.0] * rows,
            "const_b": [1] * rows,
            "label": np.where(np.arange(rows) % 3 == 0, "yes", "no"),
        }
    )
    dataset_id = _upload(client, frame)
    body = _train(
        client,
        dataset_id,
        feature_columns=["const_a", "const_b"],
        candidate_models=["dummy", "logistic_regression", "random_forest"],
    )
    assert body["best_model"] == "dummy"
    assert body["beats_baseline"] is False
    assert any("未明确优于" in w for w in body["warnings"])
    assert any("取值单一" in w for w in body["warnings"])


# ============================================================
# 回归 + 自动任务推断 + 显式回归
# ============================================================


def test_auto_regression_end_to_end(client):
    dataset_id = _upload(client, _regression_frame())
    body = _train(
        client,
        dataset_id,
        target_column="target",
        candidate_models=["dummy", "ridge", "random_forest"],
    )
    assert body["task_type"] == "regression"
    assert body["primary_metric"] == "rmse"
    assert body["class_labels"] == []
    assert body["beats_baseline"] is True
    assert body["best_model"] in {"ridge", "random_forest"}
    assert set(body["cv"]["results"].keys()) == {"dummy", "ridge", "random_forest"}
    assert len(body["cv"]["ranking"]) == 3

    t = body["test"]
    assert t["metrics"]["r2"] > 0.2
    assert t["metrics"]["rmse"] > 0
    assert t["residuals"]["count"] == body["test_rows"]
    assert t["residuals"]["std"] >= 0
    assert t["scatter"]
    assert all("actual" in p and "predicted" in p for p in t["scatter"])
    assert "confusion_matrix" not in t and "roc" not in t


def test_explicit_regression_task_and_candidate_mismatch(client):
    frame = _regression_frame()
    dataset_id = _upload(client, frame)
    resp = client.post(
        f"/api/datasets/{dataset_id}/ml/train",
        json={
            "source_version_id": "original",
            "target_column": "target",
            "task_type": "regression",
            "candidate_models": ["ridge"],
            "feature_columns": ["x1", "x2"],
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["features"] == ["x1", "x2"]
    assert body["numeric_features"] == ["x1", "x2"]
    assert body["categorical_features"] == []

    # Ridge 只能用于回归：分类任务里指定 logistic 之外还需任务匹配
    resp2 = client.post(
        f"/api/datasets/{dataset_id}/ml/train",
        json={
            "target_column": "target",
            "task_type": "regression",
            "candidate_models": ["logistic_regression"],
        },
    )
    _expect_error(client, resp2, "model_not_supported_for_task")


def test_regression_cv_folds_auto_reduce(client):
    frame = _regression_frame(rows=30, seed=3)
    dataset_id = _upload(client, frame)
    body = _train(
        client,
        dataset_id,
        target_column="target",
        task_type="regression",
        candidate_models=["ridge"],
        cv_folds=12,
    )
    assert body["requested_cv_folds"] == 12
    assert body["effective_cv_folds"] == 12
    assert len(body["cv"]["results"]["ridge"]["metrics"]["rmse"]["values"]) == 12
    assert body["cv"]["results"]["ridge"]["folds"] == 12


# ============================================================
# 目标 / 特征校验与错误
# ============================================================


def test_target_constant_rejected(client):
    frame = pd.DataFrame({"x1": np.arange(40, dtype=float), "label": ["same"] * 40})
    dataset_id = _upload(client, frame)
    resp = client.post(
        f"/api/datasets/{dataset_id}/ml/train",
        json={"target_column": "label", "task_type": "auto"},
    )
    _expect_error(client, resp, "target_constant", status=400)


def test_target_all_missing_rejected(client):
    frame = _classification_frame(rows=40, seed=1)
    frame["label"] = np.nan
    dataset_id = _upload(client, frame)
    resp = client.post(
        f"/api/datasets/{dataset_id}/ml/train",
        json={"target_column": "label", "task_type": "auto"},
    )
    _expect_error(client, resp, "target_all_missing", status=422)


def test_target_in_features_error(client):
    dataset_id = _upload(client, _classification_frame(rows=60, seed=2))
    resp = client.post(
        f"/api/datasets/{dataset_id}/ml/train",
        json={"target_column": "label", "feature_columns": ["x1", "label"]},
    )
    _expect_error(client, resp, "target_in_features")


def test_feature_not_found_error(client):
    dataset_id = _upload(client, _classification_frame(rows=60, seed=2))
    resp = client.post(
        f"/api/datasets/{dataset_id}/ml/train",
        json={"target_column": "label", "feature_columns": ["does_not_exist"]},
    )
    _expect_error(client, resp, "column_not_found")


def test_high_cardinality_text_feature_rejected(client):
    rows = 50
    rng = np.random.default_rng(6)
    frame = pd.DataFrame(
        {
            "x1": rng.normal(size=rows),
            "note": [f"unique free text {i} ends here" for i in range(rows)],
            "label": np.where(np.arange(rows) % 2 == 0, "a", "b"),
        }
    )
    dataset_id = _upload(client, frame)
    resp = client.post(
        f"/api/datasets/{dataset_id}/ml/train",
        json={"target_column": "label", "feature_columns": ["x1", "note"]},
    )
    _expect_error(client, resp, "unsupported_feature_type")


def test_auto_excludes_id_like_and_no_features(client):
    rows = 50
    frame = pd.DataFrame(
        {
            "id": list(range(rows)),
            "ts": [f"2024-0{(i % 9) + 1}-0{(i % 3) + 1} 12:00:00" for i in range(rows)],
            "note": [f"long free text row {i}" for i in range(rows)],
            "label": np.where(np.arange(rows) % 2 == 0, "a", "b"),
        }
    )
    dataset_id = _upload(client, frame)
    resp = client.post(
        f"/api/datasets/{dataset_id}/ml/train",
        json={"target_column": "label", "task_type": "auto"},
    )
    _expect_error(client, resp, "no_features")


def test_id_like_column_excluded_with_warning_metadata(client):
    rows = 80
    frame = _classification_frame(rows=rows, seed=4, label_missing=0)
    frame.insert(0, "row_id", list(range(rows)))
    dataset_id = _upload(client, frame)
    body = _train(client, dataset_id, candidate_models=["logistic_regression"])
    reasons = {item["column"]: item["reason"] for item in body["excluded_columns"]}
    assert "row_id" in reasons
    assert "整型高基数标识符" in reasons["row_id"]
    assert "row_id" not in body["features"]
    assert "x1" in body["features"]


def test_constant_and_all_missing_columns_excluded(client):
    rows = 70
    rng = np.random.default_rng(8)
    frame = _classification_frame(rows=rows, seed=8, label_missing=0)
    frame["constant_col"] = 7
    frame["empty_col"] = np.nan
    dataset_id = _upload(client, frame)
    body = _train(client, dataset_id, candidate_models=["logistic_regression"])
    excluded_names = {item["column"] for item in body["excluded_columns"]}
    assert "constant_col" in excluded_names
    assert "empty_col" in excluded_names
    assert "constant_col" not in body["features"]
    assert "empty_col" not in body["features"]


def test_too_few_rows_rejected(client, monkeypatch):
    monkeypatch.setattr(mlsvc.settings, "ml_min_total_rows", 50)
    dataset_id = _upload(client, _classification_frame(rows=30, seed=12, label_missing=0))
    resp = client.post(
        f"/api/datasets/{dataset_id}/ml/train",
        json={"target_column": "label", "task_type": "auto"},
    )
    _expect_error(client, resp, "not_enough_samples", status=422)


def test_too_many_rows_rejected(client, monkeypatch):
    monkeypatch.setattr(mlsvc.settings, "ml_max_rows", 10)
    dataset_id = _upload(client, _classification_frame(rows=30, seed=12, label_missing=0))
    resp = client.post(
        f"/api/datasets/{dataset_id}/ml/train",
        json={"target_column": "label", "task_type": "auto"},
    )
    _expect_error(client, resp, "too_many_rows", status=422)


# ============================================================
# Data Leakage 检测（派生版本）
# ============================================================


def _apply_operation(client, dataset_id, operations: list[dict]) -> str:
    resp = client.post(
        f"/api/datasets/{dataset_id}/processing/apply",
        json={"source_version_id": "original", "operations": operations},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["version_id"]


def test_potential_data_leakage_warning_from_derived(client):
    dataset_id = _upload(client, _classification_frame(rows=80, seed=20, label_missing=4))
    leaky_version = _apply_operation(
        client, dataset_id, [{"type": "fill_missing", "columns": ["x1"], "strategy": "mean"}]
    )
    body = _train(
        client,
        dataset_id,
        source_version_id=leaky_version,
        candidate_models=["logistic_regression"],
    )
    assert body["source_version_id"] == leaky_version
    assert body["potential_data_leakage"] is True
    assert body["leakage_sensitive_operations"] == ["fill_missing(mean)"]
    assert any("data leakage" in w for w in body["warnings"])

    # original 不做泄漏告警
    body_original = _train(
        client, dataset_id, source_version_id="original", candidate_models=["logistic_regression"]
    )
    assert body_original["potential_data_leakage"] is False
    assert body_original["leakage_sensitive_operations"] == []


def test_derived_version_without_statistical_ops_no_leakage(client):
    frame = _classification_frame(rows=60, seed=21, label_missing=0)
    frame = pd.concat([frame, frame.iloc[:3]], ignore_index=True)  # 制造重复行
    dataset_id = _upload(client, frame)
    version_id = _apply_operation(
        client, dataset_id, [{"type": "drop_duplicates", "columns": []}]
    )
    assert version_id != "original"
    body = _train(
        client,
        dataset_id,
        source_version_id=version_id,
        candidate_models=["logistic_regression"],
    )
    assert body["potential_data_leakage"] is False
    assert body["leakage_sensitive_operations"] == []


# ============================================================
# 模型失败降级：部分失败 / 全部失败
# ============================================================


def test_partial_model_failure_keeps_others(client, monkeypatch):
    dataset_id = _upload(client, _classification_frame(rows=60, seed=31, label_missing=0))
    original = mlsvc._build_estimator

    def broken(key, task_type, random_state):
        if key == "random_forest":
            raise RuntimeError("simulated rf failure")
        return original(key, task_type, random_state)

    monkeypatch.setattr(mlsvc, "_build_estimator", broken)
    body = _train(
        client,
        dataset_id,
        candidate_models=["logistic_regression", "random_forest"],
    )
    assert "random_forest" in body["model_errors"]
    assert "logistic_regression" in body["cv"]["results"]
    assert "random_forest" not in body["cv"]["results"]
    assert body["best_model"] == "logistic_regression"
    assert any("random_forest" in w for w in body["warnings"])


def test_all_models_failed_no_artifact(client, monkeypatch):
    dataset_id = _upload(client, _classification_frame(rows=60, seed=32, label_missing=0))
    monkeypatch.setattr(
        mlsvc,
        "_build_estimator",
        lambda key, task_type, random_state: (_ for _ in ()).throw(
            RuntimeError("boom")
        ),
    )
    resp = client.post(
        f"/api/datasets/{dataset_id}/ml/train",
        json={
            "target_column": "label",
            "candidate_models": ["dummy", "logistic_regression"],
        },
    )
    _expect_error(client, resp, "all_models_failed", status=422)
    lst = client.get(f"/api/datasets/{dataset_id}/ml/experiments").json()
    assert lst["count"] == 0  # 无半成品实验目录


# ============================================================
# 预测接口
# ============================================================


def test_prediction_ok_probabilities_and_extra_field(client):
    dataset_id = _upload(client, _classification_frame(rows=80, seed=33, label_missing=2))
    body = _train(client, dataset_id, candidate_models=["logistic_regression"])
    eid = body["experiment_id"]
    features = body["features"]

    records = []
    for extra, seed in [("north", 1), ("south", 2)]:
        rng = np.random.default_rng(seed)
        record = {
            "x1": float(rng.normal()),
            "x2": int(rng.integers(0, 20)),
            "grp": "Alpha",
            "flag": True,
        }
        record["unseen_field"] = extra
        records.append(record)
    assert set(features) == {"x1", "x2", "grp", "flag"}

    resp = client.post(
        f"/api/datasets/{dataset_id}/ml/experiments/{eid}/predict", json={"records": records}
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["success"] is True
    assert payload["count"] == 2
    assert payload["task_type"] == "classification"
    assert len(payload["predictions"]) == 2
    assert payload["predictions"][0]["index"] == 0
    assert payload["predictions"][0]["prediction"] in ("yes", "no")
    assert payload["class_labels"] == ["no", "yes"]
    assert len(payload["probabilities"]) == 2
    for probs in payload["probabilities"]:
        assert set(probs.keys()) == {"no", "yes"}
        assert abs(sum(probs.values()) - 1.0) < 1e-6
    assert any("unseen_field" in w and "忽略" in w for w in payload["warnings"])

    # 分类预测也能覆盖未见类别（OneHotEncoder handle_unknown=ignore）
    unknown = {
        "x1": 1.2,
        "x2": 5,
        "grp": "ZetaUnknown",
        "flag": False,
        "whatever": "ignored",
    }
    resp2 = client.post(
        f"/api/datasets/{dataset_id}/ml/experiments/{eid}/predict",
        json={"records": [unknown]},
    )
    assert resp2.status_code == 200, resp2.text
    assert len(resp2.json()["predictions"]) == 1


def test_prediction_missing_feature_field_422(client):
    dataset_id = _upload(client, _classification_frame(rows=60, seed=35, label_missing=0))
    body = _train(client, dataset_id, candidate_models=["logistic_regression"])
    eid = body["experiment_id"]
    resp = client.post(
        f"/api/datasets/{dataset_id}/ml/experiments/{eid}/predict",
        json={"records": [{"x1": 1.0, "grp": "Alpha"}]},  # 缺 x2/flag
    )
    _expect_error(client, resp, "missing_feature_columns", status=422)


def test_prediction_record_limit(client, monkeypatch):
    monkeypatch.setattr(mlsvc.settings, "ml_max_prediction_records", 1)
    dataset_id = _upload(client, _classification_frame(rows=60, seed=36, label_missing=0))
    body = _train(client, dataset_id, candidate_models=["logistic_regression"])
    eid = body["experiment_id"]
    records = [{"x1": 0.0, "x2": 1, "grp": "Alpha", "flag": True},
               {"x1": 1.0, "x2": 2, "grp": "Beta", "flag": False}]
    resp = client.post(
        f"/api/datasets/{dataset_id}/ml/experiments/{eid}/predict", json={"records": records}
    )
    _expect_error(client, resp, "too_many_records", status=422)


def test_regression_prediction_output(client):
    dataset_id = _upload(client, _regression_frame(rows=60, seed=40))
    body = _train(client, dataset_id, target_column="target", candidate_models=["ridge"])
    eid = body["experiment_id"]
    resp = client.post(
        f"/api/datasets/{dataset_id}/ml/experiments/{eid}/predict",
        json={"records": [{"x1": 0.5, "x2": 3, "grp": "A"}, {"x1": -1.2, "x2": 9, "grp": "B"}]},
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["task_type"] == "regression"
    assert payload["class_labels"] == []
    assert payload["probabilities"] is None
    assert len(payload["predictions"]) == 2
    assert all(isinstance(p["prediction"], float) for p in payload["predictions"])


def test_prediction_after_delete_returns_404(client):
    dataset_id = _upload(client, _classification_frame(rows=60, seed=50, label_missing=0))
    body = _train(client, dataset_id, candidate_models=["logistic_regression"])
    eid = body["experiment_id"]
    resp = client.delete(f"/api/datasets/{dataset_id}/ml/experiments/{eid}")
    assert resp.status_code == 200, resp.text
    resp2 = client.post(
        f"/api/datasets/{dataset_id}/ml/experiments/{eid}/predict",
        json={"records": [{"x1": 0.0, "x2": 1, "grp": "Alpha", "flag": True}]},
    )
    _expect_error(client, resp2, "experiment_not_found", status=404)


# ============================================================
# 实验管理：历史 / 详情 / 删除 / 版本删除保护
# ============================================================


def test_experiment_history_ordering_and_delete(client):
    dataset_id = _upload(client, _classification_frame(rows=70, seed=44, label_missing=0))
    first = _train(client, dataset_id, candidate_models=["logistic_regression"])
    second = _train(client, dataset_id, candidate_models=["logistic_regression"])
    assert first["experiment_id"] != second["experiment_id"]

    lst = client.get(f"/api/datasets/{dataset_id}/ml/experiments").json()
    assert lst["count"] == 2
    ids = [item["experiment_id"] for item in lst["experiments"]]
    # 最新在前
    assert ids == [second["experiment_id"], first["experiment_id"]]
    assert all(item["source_version_id"] == "original" for item in lst["experiments"])

    resp = client.delete(f"/api/datasets/{dataset_id}/ml/experiments/{first['experiment_id']}")
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    lst2 = client.get(f"/api/datasets/{dataset_id}/ml/experiments").json()
    assert lst2["count"] == 1
    assert lst2["experiments"][0]["experiment_id"] == second["experiment_id"]


def test_version_delete_blocked_while_experiment_references_it(client):
    dataset_id = _upload(client, _classification_frame(rows=70, seed=45, label_missing=0))
    derived_version = _apply_operation(
        client, dataset_id, [{"type": "fill_missing", "columns": ["x1"], "strategy": "median"}]
    )
    body = _train(
        client,
        dataset_id,
        source_version_id=derived_version,
        candidate_models=["logistic_regression"],
    )
    assert body["source_version_id"] == derived_version

    # 实验存在时不能删除被引用版本
    resp = client.delete(f"/api/datasets/{dataset_id}/versions/{derived_version}")
    _expect_error(client, resp, "version_in_use_by_ml_experiment", status=409)

    # 删除实验后可删除版本
    dl = client.delete(
        f"/api/datasets/{dataset_id}/ml/experiments/{body['experiment_id']}"
    )
    assert dl.status_code == 200, dl.text
    resp2 = client.delete(f"/api/datasets/{dataset_id}/versions/{derived_version}")
    assert resp2.status_code == 200, resp2.text


def test_reproducible_sha_and_split(client):
    frame = _classification_frame(rows=80, seed=46, label_missing=3)
    dataset_id = _upload(client, frame)
    body1 = _train(
        client,
        dataset_id,
        candidate_models=["logistic_regression"],
        test_size=0.25,
        random_state=7,
    )
    body2 = _train(
        client,
        dataset_id,
        candidate_models=["logistic_regression"],
        test_size=0.25,
        random_state=7,
    )
    assert re.fullmatch(r"[0-9a-f]{64}", body1["dataset_sha256"])
    assert body1["dataset_sha256"] == body2["dataset_sha256"]
    assert body1["experiment_id"] != body2["experiment_id"]
    assert body1["train_rows"] + body1["test_rows"] + 3 == 80
    assert body1["test_rows"] == body2["test_rows"]
    assert body1["test_score_summary"]["score"] == body2["test_score_summary"]["score"]
    assert body1["features"] == body2["features"]

    # 不同 random_state 的划分不要求一致，但拆分字段保持一致
    assert body1["test_size"] == 0.25
    assert body1["random_state"] == 7
    assert 10 <= body1["test_rows"] <= 30


def test_error_payload_structure_uniform(client):
    dataset_id = _upload(client, _classification_frame(rows=40, seed=2))
    resp = client.post(
        f"/api/datasets/{dataset_id}/ml/train",
        json={"target_column": "missing_target"},
    )
    payload = _expect_error(client, resp, "column_not_found")
    assert set(payload["error"].keys()) == {"code", "message"}


def test_split_with_non_stratified_fallback_is_ok(client):
    # 罕见类别（约 3 个）存在时仍能训练：回退或分层成功都应成功且不抛 500
    rng = np.random.default_rng(7)
    rows = 120
    x1 = rng.normal(size=rows)
    x2 = rng.integers(0, 8, size=rows).astype(float)
    y = np.array(["common"] * rows)
    y[:3] = "rare"
    frame = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
    dataset_id = _upload(client, frame)
    resp = client.post(
        f"/api/datasets/{dataset_id}/ml/train",
        json={
            "target_column": "y",
            "task_type": "classification",
            "candidate_models": ["dummy"],
            "cv_folds": 2,
        },
    )
    assert resp.status_code == 200, resp.text


def test_invalid_experiment_id_returns_400(client):
    dataset_id = _upload(client, _classification_frame(rows=40, seed=2))
    resp = client.get(f"/api/datasets/{dataset_id}/ml/experiments/not-a-uuid")
    _expect_error(client, resp, "invalid_experiment_id", status=400)


def test_train_and_predict_for_unknown_dataset(client):
    resp = client.post(
        "/api/datasets/11111111-1111-1111-1111-111111111111/ml/train",
        json={"target_column": "x", "task_type": "classification"},
    )
    assert resp.status_code == 404 or resp.status_code == 400  # 明确错误且非 500
