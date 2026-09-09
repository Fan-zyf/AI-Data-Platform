"""Dataset Session + 自动 EDA 接口测试。

覆盖：上传返回 UUID / Session 落盘 / EDA 结构 / 数值统计 / 缺失 / IQR /
Pearson 相关 / 分类 TopN / 非法或不存在 dataset_id / 过期 / 删除 / 边界数据。
"""

import io
import pathlib
import uuid

import numpy as np
import pandas as pd

BACKEND_DIR = pathlib.Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent

from app.core.config import settings  # noqa: E402

DEMO_CSV = PROJECT_ROOT / "data" / "sample" / "demo.csv"


def _upload(client, name: str, content: bytes):
    res = client.post("/api/data/upload", files={"file": (name, content, "text/csv")})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["success"] is True
    return body


def _upload_csv(client, name="demo.csv"):
    return _upload(client, name, DEMO_CSV.read_bytes())


def _upload_dataframe(client, df: pd.DataFrame, name: str = "frame.csv"):
    content = df.to_csv(index=False).encode("utf-8")
    return _upload(client, name, content)


def _upload_excel(client, df: pd.DataFrame, name: str = "frame.xlsx"):
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False, engine="openpyxl")
    mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    res = client.post(
        "/api/data/upload", files={"file": (name, buffer.getvalue(), mime)}
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["success"] is True
    return body


def _get_eda(client, dataset_id: str):
    return client.get(f"/api/datasets/{dataset_id}/eda")


# ---------------------------------------------------------------
# 上传与 Session
# ---------------------------------------------------------------


def test_upload_returns_valid_uuid_and_session_fields(client):
    body = _upload_csv(client)

    dataset_id = body["dataset_id"]
    assert dataset_id
    # 必须是合法 UUID
    assert str(uuid.UUID(dataset_id)) == dataset_id
    assert body["created_at"]
    assert body["expires_at"]
    # 原有画像字段全部保留
    assert body["dataset"]["rows"] == 23
    assert body["quality"]["total_missing"] == 7
    assert len(body["column_profiles"]) == 7
    assert body["warnings"]


def test_session_persisted_on_disk(client):
    body = _upload_csv(client)
    dataset_id = body["dataset_id"]

    from app.core.config import settings

    directory = pathlib.Path(settings.dataset_runtime_dir) / dataset_id
    assert directory.is_dir()
    assert (directory / "metadata.json").is_file()
    assert (directory / "source.csv").is_file()


# ---------------------------------------------------------------
# EDA 主流程
# ---------------------------------------------------------------


def test_eda_returns_full_structure_for_demo(client):
    body = _upload_csv(client)
    dataset_id = body["dataset_id"]

    res = _get_eda(client, dataset_id)
    assert res.status_code == 200, res.text
    eda = res.json()
    assert eda["success"] is True
    assert eda["dataset_id"] == dataset_id
    assert eda["dataset"]["rows"] == 23
    assert eda["dataset"]["columns"] == 7

    summary = eda["summary"]
    assert summary["total_rows"] == 23
    assert summary["total_columns"] == 7
    assert summary["numeric_columns"] == 3
    assert summary["datetime_columns"] == 1
    assert summary["categorical_columns"] == 2  # city + is_active
    assert summary["rule_based_insights"]

    numeric_columns = {item["column"] for item in eda["numeric_summaries"]}
    assert numeric_columns == {"id", "age", "score"}
    assert {item["column"] for item in eda["numeric_distributions"]} == numeric_columns
    # 分布字段一一对应且直方图数据完整
    for dist in eda["numeric_distributions"]:
        assert len(dist["counts"]) == len(dist["bin_edges"]) - 1

    cat_columns = {item["column"] for item in eda["categorical_summaries"]}
    assert cat_columns == {"city", "is_active"}

    missing = eda["missing_analysis"]
    assert missing["total_missing"] == 7
    assert missing["total_cells"] == 23 * 7
    assert set(missing["columns_with_missing"]) == {"age", "score", "joined"}
    assert set(missing["columns_without_missing"]) == {"id", "city", "is_active", "notes"}
    # 按缺失率降序
    rates = [item["missing_percentage"] for item in missing["by_column"]]
    assert rates == sorted(rates, reverse=True)
    assert rates[0] > 0

    assert len(eda["correlation"]["columns"]) == 3
    assert len(eda["correlation"]["matrix"]) == 3


def test_eda_numeric_descriptive_stats_correct(client):
    body = _upload_csv(client)
    dataset_id = body["dataset_id"]
    eda = _get_eda(client, dataset_id).json()

    # 用 pandas 独立计算期望值
    expected = pd.read_csv(io.BytesIO(DEMO_CSV.read_bytes()))
    age = pd.to_numeric(expected["age"], errors="coerce").dropna()

    row = next(item for item in eda["numeric_summaries"] if item["column"] == "age")
    assert row["count"] == len(age)
    assert row["missing_count"] == int(expected["age"].isna().sum() + (expected["age"].astype(str).str.strip() == "").sum())
    assert abs(row["mean"] - age.mean()) < 1e-4
    assert abs(row["std"] - age.std()) < 1e-4
    assert abs(row["min"] - age.min()) < 1e-6
    assert abs(row["q1"] - age.quantile(0.25)) < 1e-4
    assert abs(row["median"] - age.median()) < 1e-4
    assert abs(row["q3"] - age.quantile(0.75)) < 1e-4
    assert abs(row["max"] - age.max()) < 1e-6
    assert abs(row["range"] - (age.max() - age.min())) < 1e-4
    assert abs(row["iqr"] - (age.quantile(0.75) - age.quantile(0.25))) < 1e-4
    assert row["skewness"] is not None  # 样本量足够时偏度可计算
    assert row["unique_count"] == age.nunique()


def test_eda_missing_analysis_matches_profile(client):
    body = _upload_csv(client)
    eda = _get_eda(client, body["dataset_id"]).json()
    missing = eda["missing_analysis"]
    assert missing["overall_missing_percentage"] == 4.35
    # 与上传画像的缺失总量一致
    assert missing["total_missing"] == body["quality"]["total_missing"]
    assert len(missing["columns_with_missing"]) == body["quality"]["columns_with_missing"]


def test_eda_outlier_iqr_and_constant_column(client):
    df = pd.DataFrame(
        {
            "x": list(range(1, 21)) + [1000, 1000, 1000],  # 3 个明显 IQR 异常值
            "c": 5,  # 常量列
        }
    )
    body = _upload_dataframe(client, df)
    eda = _get_eda(client, body["dataset_id"]).json()

    outlier = next(item for item in eda["outlier_analysis"] if item["column"] == "x")
    assert outlier["detection_method"] == "IQR"
    assert outlier["outlier_count"] == 3

    # IQR 箱线边界按全量数据计算（教科书 Tukey 定义）
    x_full = np.array(list(range(1, 21)) + [1000, 1000, 1000], dtype=float)
    q1, q3 = np.quantile(x_full, [0.25, 0.75])
    iqr = q3 - q1
    assert abs(outlier["lower_bound"] - round(q1 - 1.5 * iqr, 6)) < 1e-6
    assert abs(outlier["upper_bound"] - round(q3 + 1.5 * iqr, 6)) < 1e-6

    # 常量列：std=0.0、单箱直方图、不参与相关性
    const_summary = next(
        item for item in eda["numeric_summaries"] if item["column"] == "c"
    )
    assert const_summary["std"] == 0.0
    const_dist = next(
        item for item in eda["numeric_distributions"] if item["column"] == "c"
    )
    assert len(const_dist["bin_edges"]) == 2
    assert const_dist["counts"] == [len(df)]
    assert "c" not in eda["correlation"]["columns"]


def test_eda_pearson_correlation_correct(client):
    n = 20
    x = np.arange(1, n + 1, dtype=float)
    y = 3 * x - 2
    df = pd.DataFrame({"x": x, "y": y, "noise": np.random.default_rng(0).normal(size=n)})
    body = _upload_dataframe(client, df)
    eda = _get_eda(client, body["dataset_id"]).json()

    corr = eda["correlation"]
    assert set(corr["columns"]) == {"x", "y", "noise"}
    assert len(corr["matrix"]) == 3
    for row in corr["matrix"]:
        assert len(row) == 3

    # 对角线为 1
    for i in range(3):
        assert abs(corr["matrix"][i][i] - 1.0) < 1e-6

    idx = {name: i for i, name in enumerate(corr["columns"])}
    assert abs(corr["matrix"][idx["x"]][idx["y"]] - 1.0) < 1e-4

    top = corr["top_correlations"]
    assert top[0]["variable_1"] == "x"
    assert top[0]["variable_2"] == "y"
    assert top[0]["abs_correlation"] >= 0.7
    # 不出现重复组合（反向对）
    pairs = {(item["variable_1"], item["variable_2"]) for item in top}
    assert ("y", "x") not in pairs


def test_eda_categorical_top_values_correct(client):
    df = pd.DataFrame(
        {
            "kind": ["A", "B", "A", "A", "C", "B", "A", None, "C", "C"],
            "ok": ["true", "false", "true", "true", "false", "true", "false", "true", "true", "true"],
        }
    )
    body = _upload_dataframe(client, df)
    eda = _get_eda(client, body["dataset_id"]).json()

    kind = next(item for item in eda["categorical_summaries"] if item["column"] == "kind")
    assert kind["count"] == 9
    assert kind["missing_count"] == 1
    assert kind["unique_count"] == 3
    assert kind["top_values"][0]["value"] == "A"
    assert kind["top_values"][0]["count"] == 4
    assert abs(kind["top_values"][0]["percentage"] - round(4 / 9 * 100, 2)) < 1e-6
    counts = [item["count"] for item in kind["top_values"]]
    assert counts == sorted(counts, reverse=True)


# ---------------------------------------------------------------
# Session 边界与错误
# ---------------------------------------------------------------


def test_eda_invalid_dataset_id_returns_400(client):
    res = _get_eda(client, "not-a-valid-uuid")
    assert res.status_code == 400
    body = res.json()
    assert body["success"] is False
    assert body["error"]["code"] == "invalid_dataset_id"


def test_eda_unknown_dataset_returns_404(client):
    dataset_id = str(uuid.uuid4())
    res = _get_eda(client, dataset_id)
    assert res.status_code == 404
    assert res.json()["error"]["code"] == "dataset_not_found"


def test_delete_session_success_then_gone(client):
    body = _upload_csv(client)
    dataset_id = body["dataset_id"]

    res = client.delete(f"/api/datasets/{dataset_id}")
    assert res.status_code == 200
    payload = res.json()
    assert payload["success"] is True
    assert payload["message"] == "Dataset session deleted"

    # 删除后 EDA 与再次删除都返回 404
    assert _get_eda(client, dataset_id).status_code == 404
    again = client.delete(f"/api/datasets/{dataset_id}")
    assert again.status_code == 404
    assert again.json()["error"]["code"] == "dataset_not_found"


def test_delete_unknown_dataset_returns_404(client):
    res = client.delete(f"/api/datasets/{uuid.uuid4()}")
    assert res.status_code == 404
    assert res.json()["error"]["code"] == "dataset_not_found"


def test_expired_session_returns_404(client, monkeypatch):
    monkeypatch.setattr(settings, "dataset_session_ttl_minutes", -5)
    body = _upload_csv(client)
    dataset_id = body["dataset_id"]

    res = _get_eda(client, dataset_id)
    assert res.status_code == 404
    assert res.json()["error"]["code"] == "dataset_expired"


def test_eda_repeatable_on_same_dataset(client):
    body = _upload_csv(client)
    dataset_id = body["dataset_id"]
    first = _get_eda(client, dataset_id)
    second = _get_eda(client, dataset_id)
    assert first.status_code == second.status_code == 200
    assert first.json()["numeric_summaries"] == second.json()["numeric_summaries"]


# ---------------------------------------------------------------
# 边界数据（不报错、结构化返回）
# ---------------------------------------------------------------


def test_eda_all_missing_column_graceful(client):
    # Excel 全缺失列：读取为 float64 全 NaN；为通过上传的行数校验，
    # 需同时包含至少一列有有效数据的字段。
    df = pd.DataFrame(
        {
            "ok": [1, 2, 3],
            "value": pd.Series([None, None, None], dtype="float64"),
        }
    )
    body = _upload_excel(client, df)
    res = _get_eda(client, body["dataset_id"])
    assert res.status_code == 200, res.text
    eda = res.json()
    assert eda["success"] is True
    row = next(
        item for item in eda["numeric_summaries"] if item["column"] == "value"
    )
    assert row["count"] == 0
    assert row["mean"] is None
    assert row["missing_count"] == 3
    # 缺失率 100%，应被标记为高缺失字段
    assert eda["summary"]["high_missing_columns"] == 1


def test_eda_no_numeric_columns_graceful(client):
    # 纯文本列（CSV 引号字符串），避免被解析为数值
    content = b"name\n\"alice\"\n\"bob\"\n\"alice\""
    body = _upload(client, "texts.csv", content)
    res = _get_eda(client, body["dataset_id"])
    assert res.status_code == 200, res.text
    eda = res.json()
    assert eda["numeric_summaries"] == []
    assert eda["numeric_distributions"] == []
    assert eda["correlation"]["columns"] == []
    assert eda["correlation"]["matrix"] == []
    assert eda["summary"]["numeric_columns"] == 0
    assert any("数值" in text for text in eda["summary"]["rule_based_insights"])


def test_eda_single_numeric_correlation_empty(client):
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    body = _upload_dataframe(client, df)
    eda = _get_eda(client, body["dataset_id"]).json()
    # 仅一个数值字段：相关矩阵返回空结构而不是报错
    assert eda["correlation"]["columns"] == []
    assert eda["correlation"]["matrix"] == []
    assert eda["summary"]["strong_correlations"] == 0
