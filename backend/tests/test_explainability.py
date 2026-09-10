"""v0.6 模型可解释性（SHAP）测试。

覆盖要点（与用户 spec 一一对应）：
1. experiment 不存在 → 404 experiment_not_found
2. 非法 experiment_id → 400 invalid_experiment_id
3. 模型文件缺失 → 500 experiment_model_missing
4. classification 解释（TreeExplainer 路径）→ 200 + 完整结构
5. regression 解释（TreeExplainer 路径）→ 200 + 完整结构
6. JSON schema：response 含 required 字段（global/summary/samples/feature_names/class_label_used/base_value）
7. 结果保存路径：runtime/datasets/<id>/ml/experiments/<exp_id>/shap_result.json 存在
8. 删除 experiment 后 SHAP 文件被级联清理
9. KernelExplainer 兜底：Dummy 模型也能给出（无信息量但 0 不报错）解释
10. dataset-scoped 入口与跨 dataset 校验
11. 缓存命中：重复请求不会重新计算
12. explainer 类型正确识别
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.services import experiment_manager as em


# ============================================================
# 工具
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


def _classification_frame(rows: int = 80, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=rows)
    x2 = rng.integers(0, 20, size=rows).astype(float)
    grp = rng.choice(["Alpha", "Beta", "Gamma"], size=rows)
    grp_effect = np.where(grp == "Alpha", 0.9, np.where(grp == "Beta", 0.1, -0.8))
    score = 1.3 * x1 + grp_effect + rng.normal(0, 0.3, size=rows)
    label = np.where(score > float(np.median(score)), "yes", "no")
    return pd.DataFrame({"x1": x1, "x2": x2, "grp": grp, "label": label})


def _regression_frame(rows: int = 80, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=rows)
    x2 = rng.integers(0, 30, size=rows).astype(float)
    grp = rng.choice(["A", "B", "C"], size=rows)
    target = (
        2.5 * x1
        + np.where(grp == "A", 1.4, np.where(grp == "B", -0.3, 0.7))
        + rng.normal(0, 0.4, size=rows)
    )
    return pd.DataFrame({"x1": x1, "x2": x2, "grp": grp, "target": target})


def _expect_error(client, resp, expected_code: str, status: int | None = None):
    assert resp.status_code == (status if status is not None else 400), resp.text
    payload = resp.json()
    assert payload["success"] is False
    assert payload["error"]["code"] == expected_code
    return payload


# ============================================================
# 1. experiment 不存在
# ============================================================


def test_explain_unknown_experiment_returns_404(client):
    resp = client.post(f"/api/ml/experiments/{uuid.uuid4()}/explain")
    _expect_error(client, resp, "experiment_not_found", 404)


# ============================================================
# 2. 非法 experiment_id（不是 UUID）
# ============================================================


def test_explain_invalid_experiment_id_returns_400(client):
    resp = client.post("/api/ml/experiments/not-a-uuid/explain")
    _expect_error(client, resp, "invalid_experiment_id", 400)


# ============================================================
# 3. 模型文件缺失（实验目录被人为删掉 model.joblib）
# ============================================================


def test_explain_without_model_file_returns_500(client):
    dataset_id = _upload(client, _classification_frame())
    body = _train(client, dataset_id, candidate_models=["random_forest"])
    experiment_id = body["experiment_id"]

    # 拿到实验目录并删掉 model.joblib
    location = em.find_experiment_location(experiment_id)
    assert location is not None
    _, exp_dir = location
    model_path = Path(exp_dir) / em.MODEL_FILE
    model_path.unlink()

    resp = client.post(f"/api/ml/experiments/{experiment_id}/explain")
    _expect_error(client, resp, "experiment_model_missing", 500)


# ============================================================
# 4. 分类 SHAP 解释（TreeExplainer）
# ============================================================


def test_explain_classification_random_forest(client):
    dataset_id = _upload(client, _classification_frame())
    body = _train(client, dataset_id, candidate_models=["random_forest"])
    experiment_id = body["experiment_id"]
    assert body["task_type"] == "classification"
    assert body["best_model"] == "random_forest"

    resp = client.post(f"/api/ml/experiments/{experiment_id}/explain")
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["success"] is True
    assert payload["task_type"] == "classification"
    assert payload["model_class"] == "RandomForestClassifier"
    assert payload["explainer_type"] == "tree"
    assert payload["class_label_used"] in payload["class_labels"]
    assert payload["class_labels"] == body["class_labels"]
    # global / summary / samples / base_value 都应存在
    assert payload["global"]["feature_names"]
    assert len(payload["global"]["importance"]) == len(payload["global"]["feature_names"])
    assert payload["global"]["n_rows_used"] > 0
    assert payload["summary"]["feature_names"] == payload["global"]["feature_names"]
    assert len(payload["summary"]["values"]) == payload["global"]["n_rows_used"]
    assert len(payload["summary"]["shap_values"]) == payload["global"]["n_rows_used"]
    assert len(payload["summary"]["values"][0]) == len(payload["feature_names"])
    assert payload["samples"], "应当至少返回一条 sample 解释"
    for sample in payload["samples"]:
        assert sample["prediction"]["kind"] == "classification"
        assert sample["prediction"]["class_label"] == payload["class_label_used"]
        assert "contributions" in sample and sample["contributions"]
        contribs = sample["contributions"]
        assert len(contribs) == len(payload["feature_names"])
        # 每条 contribution 至少有 feature / shap 字段
        for c in contribs:
            assert "feature" in c and "shap" in c
    # base_value 必须是有限数
    assert payload["base_value"] is not None


# ============================================================
# 5. 回归 SHAP 解释（RandomForestRegressor → TreeExplainer）
# ============================================================


def test_explain_regression_random_forest(client):
    dataset_id = _upload(client, _regression_frame())
    body = _train(
        client,
        dataset_id,
        target_column="target",
        task_type="regression",
        candidate_models=["random_forest"],
    )
    experiment_id = body["experiment_id"]
    assert body["task_type"] == "regression"

    resp = client.post(f"/api/ml/experiments/{experiment_id}/explain")
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["success"] is True
    assert payload["task_type"] == "regression"
    assert payload["model_class"] == "RandomForestRegressor"
    assert payload["explainer_type"] == "tree"
    assert payload["class_label_used"] is None
    assert payload["global"]["n_rows_used"] > 0
    for sample in payload["samples"]:
        assert sample["prediction"]["kind"] == "regression"
        assert "value" in sample["prediction"]


# ============================================================
# 6. JSON schema 完整性（含 top_k + warnings + app_version）
# ============================================================


def test_explain_response_schema_complete(client):
    dataset_id = _upload(client, _classification_frame())
    body = _train(client, dataset_id, candidate_models=["random_forest"])
    experiment_id = body["experiment_id"]

    resp = client.post(f"/api/ml/experiments/{experiment_id}/explain")
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    # 关键字段必现
    for key in [
        "success",
        "experiment_id",
        "dataset_id",
        "source_version_id",
        "task_type",
        "best_model",
        "model_class",
        "explainer_type",
        "class_label_used",
        "class_labels",
        "base_value",
        "feature_names",
        "global",
        "summary",
        "samples",
        "warnings",
        "created_at",
        "app_version",
    ]:
        assert key in payload, f"missing key: {key}"
    assert payload["app_version"] == "0.6.0"
    assert payload["experiment_id"] == experiment_id
    assert payload["dataset_id"] == dataset_id
    # global 结构
    g = payload["global"]
    for gkey in ["feature_names", "importance", "n_rows_used", "top_feature_names", "top_importance"]:
        assert gkey in g
    assert len(g["top_feature_names"]) == len(g["top_importance"])
    # summary 结构
    s = payload["summary"]
    for skey in ["feature_names", "sample_indices", "values", "shap_values"]:
        assert skey in s


# ============================================================
# 7. SHAP 结果落盘到正确路径（D 盘 / 业务规范）
# ============================================================


def test_explain_persists_shap_result_in_expected_path(client):
    dataset_id = _upload(client, _classification_frame())
    body = _train(client, dataset_id, candidate_models=["random_forest"])
    experiment_id = body["experiment_id"]

    resp = client.post(f"/api/ml/experiments/{experiment_id}/explain")
    assert resp.status_code == 200, resp.text

    location = em.find_experiment_location(experiment_id)
    assert location is not None
    found_dataset_id, exp_dir = location
    assert found_dataset_id == dataset_id
    shap_path = Path(exp_dir) / em.SHAP_RESULT_FILE
    assert shap_path.is_file(), f"SHAP 结果未落盘：{shap_path}"
    # 落盘内容必须是合法 JSON 且 success
    saved = json.loads(shap_path.read_text(encoding="utf-8"))
    assert saved["success"] is True
    assert saved["experiment_id"] == experiment_id
    # 元数据也应记录 explainability 状态
    metadata = json.loads((Path(exp_dir) / em.METADATA_FILE).read_text(encoding="utf-8"))
    assert "explainability" in metadata
    assert metadata["explainability"]["status"] == "computed"
    assert metadata["explainability"]["result_path"] == em.SHAP_RESULT_FILE
    assert metadata["explainability"]["explainer_type"] in {"tree", "linear", "kernel"}


# ============================================================
# 8. 删除 experiment 后 SHAP 文件级联清理
# ============================================================


def test_delete_experiment_cascades_shap_files(client):
    dataset_id = _upload(client, _classification_frame())
    body = _train(client, dataset_id, candidate_models=["random_forest"])
    experiment_id = body["experiment_id"]

    client.post(f"/api/ml/experiments/{experiment_id}/explain")
    location = em.find_experiment_location(experiment_id)
    _, exp_dir = location
    assert (Path(exp_dir) / em.SHAP_RESULT_FILE).is_file()

    # 删除实验
    resp = client.delete(f"/api/datasets/{dataset_id}/ml/experiments/{experiment_id}")
    assert resp.status_code == 200, resp.text
    # 实验目录应整体消失，shap_result.json 也随之消失
    assert not Path(exp_dir).exists()
    # 再调用 explain 应得到 404
    again = client.post(f"/api/ml/experiments/{experiment_id}/explain")
    _expect_error(client, again, "experiment_not_found", 404)


# ============================================================
# 9. Dummy 走 KernelExplainer 兜底（仍返回结果，仅 SHAP ≈ 0）
# ============================================================


def test_explain_dummy_classifier_uses_kernel_fallback(client):
    dataset_id = _upload(client, _classification_frame())
    body = _train(client, dataset_id, candidate_models=["dummy"])
    experiment_id = body["experiment_id"]
    assert body["best_model"] == "dummy"

    resp = client.post(f"/api/ml/experiments/{experiment_id}/explain")
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["explainer_type"] == "kernel"
    # Dummy 的解释几乎全为 0
    flat = [abs(v) for row in payload["summary"]["shap_values"] for v in row]
    assert max(flat) < 1e-3
    # 应附带警告
    assert any("Dummy" in w for w in payload["warnings"])


# ============================================================
# 10. dataset-scoped 入口 + 跨 dataset 校验
# ============================================================


def test_dataset_scoped_explain_endpoint(client):
    dataset_id = _upload(client, _classification_frame())
    body = _train(client, dataset_id, candidate_models=["random_forest"])
    experiment_id = body["experiment_id"]

    # 正确的 dataset_id
    resp = client.post(
        f"/api/datasets/{dataset_id}/ml/experiments/{experiment_id}/explain"
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["success"] is True

    # 错误的 dataset_id 应报错
    wrong = client.post(
        f"/api/datasets/{uuid.uuid4()}/ml/experiments/{experiment_id}/explain"
    )
    _expect_error(client, wrong, "experiment_dataset_mismatch", 400)


# ============================================================
# 11. 缓存命中：重复请求不重算
# ============================================================


def test_explain_caches_and_avoids_recomputation(client):
    dataset_id = _upload(client, _classification_frame())
    body = _train(client, dataset_id, candidate_models=["random_forest"])
    experiment_id = body["experiment_id"]

    first = client.post(f"/api/ml/experiments/{experiment_id}/explain").json()
    second = client.post(f"/api/ml/experiments/{experiment_id}/explain").json()
    assert second["cached"] is True
    assert second["created_at"] == first["created_at"]

    # 显式 regenerate=True 强制重算
    third = client.post(
        f"/api/ml/experiments/{experiment_id}/explain",
        json={"regenerate": True},
    ).json()
    assert third["cached"] is False


# ============================================================
# 12. ExplainRequest 字段校验
# ============================================================


def test_explain_request_validation(client):
    dataset_id = _upload(client, _classification_frame())
    body = _train(client, dataset_id, candidate_models=["random_forest"])
    experiment_id = body["experiment_id"]

    # max_summary_rows 超出范围
    bad = client.post(
        f"/api/ml/experiments/{experiment_id}/explain",
        json={"max_summary_rows": 5000},
    )
    assert bad.status_code == 422

    # sample_indices 超过 20 个
    too_many = client.post(
        f"/api/ml/experiments/{experiment_id}/explain",
        json={"sample_indices": list(range(25))},
    )
    assert too_many.status_code == 422

    # 合法：自定义 sample_indices + max_summary_rows
    ok = client.post(
        f"/api/ml/experiments/{experiment_id}/explain",
        json={"max_summary_rows": 30, "sample_indices": [0, 2], "regenerate": True},
    )
    assert ok.status_code == 200, ok.text
    payload = ok.json()
    assert len(payload["samples"]) == 2
    assert payload["global"]["n_rows_used"] <= 30
