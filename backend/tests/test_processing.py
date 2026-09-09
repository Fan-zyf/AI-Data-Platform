"""v0.4 数据清洗 / 特征工程 / 数据版本管理 的自动化测试。

尽量用动态 DataFrame + io.BytesIO 生成数据与边界场景，避免堆大量测试数据文件。
"""

import io
import json
import re
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.config import settings

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)


# ------------------------------------------------------------
# 动态测试数据
# ------------------------------------------------------------


def make_frame() -> pd.DataFrame:
    """构造含数值/分类/日期/文本列 + 缺失 + 重复 + 异常值 + 常量列的 DataFrame。"""
    rows = [
        {"age": 30, "score": 88.5, "city": "北京", "joined": "2024-01-15", "name": "  Alice "},
        {"age": 25, "score": 91.2, "city": "上海", "joined": "2024-02-01", "name": "Bob"},
        {"age": None, "score": None, "city": "广州", "joined": "2024-03-10", "name": " carol "},
        {"age": 40, "score": 70.0, "city": None, "joined": "2024-04-20", "name": "dave"},
        {"age": 30, "score": 88.5, "city": "北京", "joined": "2024-05-05", "name": " eve "},
        {"age": 22, "score": 55.0, "city": "上海", "joined": "2023-12-01", "name": "frank"},
        {"age": None, "score": 100.0, "city": "广州", "joined": "2023-11-11", "name": " grace"},
        {"age": 35, "score": 60.0, "city": "上海", "joined": "2023-10-21", "name": "  henry"},
    ]
    frame = pd.DataFrame(rows)
    # 追加两行完整重复行（duplicates == 2）
    frame = pd.concat([frame, frame.iloc[[0, 1]]], ignore_index=True)
    frame["income"] = [10.0, 12.0, None, 11.0, 9999.0, 13.0, 14.0, 15.0, 10.0, 12.0]
    frame["const_score"] = 5.0  # 常量列（std=0 / max=min）
    return frame


def upload_frame(client, frame: pd.DataFrame) -> str:
    buffer = io.BytesIO(frame.to_csv(index=False).encode("utf-8"))
    response = client.post(
        "/api/data/upload",
        files={"file": ("dynamic.csv", buffer, "text/csv")},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["success"] is True
    return payload["dataset_id"]


def _op(op_type: str, **kwargs) -> dict:
    operation = {"type": op_type}
    operation.update({k: v for k, v in kwargs.items() if v is not None})
    return operation


def plan(operations: list[dict], source: str | None = None) -> dict:
    body: dict = {"operations": operations}
    if source:
        body["source_version_id"] = source
    return body


def preview(client, dataset_id: str, operations, source: str | None = None):
    return client.post(
        f"/api/datasets/{dataset_id}/processing/preview", json=plan(operations, source)
    )


def apply(client, dataset_id: str, operations, source: str | None = None):
    return client.post(
        f"/api/datasets/{dataset_id}/processing/apply", json=plan(operations, source)
    )


def dataset_dir(dataset_id: str) -> Path:
    return Path(settings.dataset_runtime_dir) / dataset_id


def version_dir_count(dataset_id: str) -> int:
    versions = dataset_dir(dataset_id) / "versions"
    if not versions.is_dir():
        return 0
    return sum(1 for entry in versions.iterdir() if UUID_RE.match(entry.name))


def read_version_csv(dataset_id: str, version_id: str) -> pd.DataFrame:
    path = dataset_dir(dataset_id) / "versions" / version_id / "data.csv"
    return pd.read_csv(path)


def assert_ok(response):
    assert response.status_code == 200, response.text
    body = response.json()
    assert body.get("success") is True, body
    return body


# ------------------------------------------------------------
# 版本模型基础
# ------------------------------------------------------------


def test_original_version_present(client):
    dataset_id = upload_frame(client, make_frame())
    body = assert_ok(client.get(f"/api/datasets/{dataset_id}/versions"))
    assert body["original"]["version_id"] == "original"
    assert body["original"]["is_original"] is True
    assert body["original"]["rows"] == 10
    assert body["original"]["columns"] == 7
    assert body["original"]["duplicates"] == 2
    assert body["versions"] == []


def test_preview_does_not_create_any_file(client):
    dataset_id = upload_frame(client, make_frame())
    assert version_dir_count(dataset_id) == 0
    response = preview(client, dataset_id, [_op("drop_duplicates")])
    assert response.status_code == 200
    body = response.json()
    assert body["duplicates_before"] == 2
    assert body["rows_after"] == 8
    # preview 绝不能写入任何版本
    assert version_dir_count(dataset_id) == 0
    assert not (dataset_dir(dataset_id) / "versions").exists()


def test_apply_creates_valid_uuid_version(client):
    dataset_id = upload_frame(client, make_frame())
    response = apply(client, dataset_id, [_op("drop_duplicates")])
    body = assert_ok(response)
    assert UUID_RE.match(body["version_id"])
    assert body["parent_version_id"] == "original"
    assert body["rows"] == 8
    assert version_dir_count(dataset_id) == 1
    # 版本元信息可读，version_id 与目录名一致
    meta_path = dataset_dir(dataset_id) / "versions" / body["version_id"] / "metadata.json"
    assert meta_path.is_file()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["version_id"] == body["version_id"]
    assert meta["dataset_id"] == dataset_id


# ------------------------------------------------------------
# 数据清洗
# ------------------------------------------------------------


def test_drop_duplicates(client):
    dataset_id = upload_frame(client, make_frame())
    body = assert_ok(apply(client, dataset_id, [_op("drop_duplicates")]))
    assert body["operations_applied"][0]["type"] == "drop_duplicates"
    assert body["operations_applied"][0]["effect"]["removed_rows"] == 2
    assert body["rows"] == 8


def test_fill_missing_mean(client):
    dataset_id = upload_frame(client, make_frame())
    p = assert_ok(preview(client, dataset_id, [_op("fill_missing", columns=["age"], strategy="mean")]))
    assert p["missing_before"] == 5  # age x2 + score/city/income x1
    assert p["missing_after"] == 3
    body = assert_ok(apply(client, dataset_id, [_op("fill_missing", columns=["age"], strategy="mean")]))
    saved = read_version_csv(dataset_id, body["version_id"])
    # 非缺失均值 = 237/8 = 29.625
    filled = saved.loc[saved["age"].notna() & saved["age"].eq(saved["age"])]
    assert abs(float(filled.loc[2, "age"]) - 29.625) < 1e-9


def test_fill_missing_median(client):
    dataset_id = upload_frame(client, make_frame())
    body = assert_ok(apply(client, dataset_id, [_op("fill_missing", columns=["age"], strategy="median")]))
    saved = read_version_csv(dataset_id, body["version_id"])
    assert float(saved.loc[2, "age"]) == 30.0


def test_fill_missing_categorical_mode(client):
    dataset_id = upload_frame(client, make_frame())
    body = assert_ok(apply(client, dataset_id, [_op("fill_missing", columns=["city"], strategy="mode")]))
    saved = read_version_csv(dataset_id, body["version_id"])
    assert str(saved.loc[3, "city"]) == "上海"  # 众数：上海出现 4 次最多


def test_fill_missing_constant(client):
    dataset_id = upload_frame(client, make_frame())
    body = assert_ok(
        apply(client, dataset_id, [_op("fill_missing", columns=["score"], strategy="constant", fill_value=42)])
    )
    saved = read_version_csv(dataset_id, body["version_id"])
    assert float(saved.loc[2, "score"]) == 42.0


def test_fill_missing_numeric_constant_rejects_string(client):
    dataset_id = upload_frame(client, make_frame())
    response = apply(
        client,
        dataset_id,
        [_op("fill_missing", columns=["score"], strategy="constant", fill_value="abc")],
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "missing_strategy_incompatible"
    assert version_dir_count(dataset_id) == 0


def test_fill_missing_incompatible_strategy_on_text(client):
    dataset_id = upload_frame(client, make_frame())
    response = apply(client, dataset_id, [_op("fill_missing", columns=["city"], strategy="mean")])
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "missing_strategy_incompatible"


def test_drop_columns_and_prevent_dropping_all(client):
    dataset_id = upload_frame(client, make_frame())
    body = assert_ok(apply(client, dataset_id, [_op("drop_columns", columns=["city", "name"])]))
    assert body["columns"] == 5
    assert "city" not in read_version_csv(dataset_id, body["version_id"]).columns

    all_cols = list(make_frame().columns)
    response = apply(client, dataset_id, [_op("drop_columns", columns=all_cols)])
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "drop_all_columns"


def test_convert_type_numeric(client):
    dataset_id = upload_frame(client, make_frame())
    body = assert_ok(apply(client, dataset_id, [_op("convert_type", column="score", target_type="numeric")]))
    assert body["success"] is True


def test_convert_type_failure_reports_missing(client):
    dataset_id = upload_frame(client, make_frame())
    frame = make_frame().copy()
    frame["age_str"] = ["aa", "10", "30", "40", "30", "22", "zz", "35", "30", "25"]
    dataset_id = upload_frame(client, frame)
    response = apply(client, dataset_id, [_op("convert_type", column="age_str", target_type="numeric")])
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "conversion_failed"


def test_convert_type_datetime(client):
    dataset_id = upload_frame(client, make_frame())
    body = assert_ok(apply(client, dataset_id, [_op("convert_type", column="joined", target_type="datetime")]))
    saved = read_version_csv(dataset_id, body["version_id"])
    assert str(saved.loc[0, "joined"]).startswith("2024-01-15")


def test_convert_type_boolean(client):
    frame = make_frame().copy()
    frame["is_ok"] = ["yes", "no", "1", "0", "true", "false", "yes", "no", "yes", "no"]
    dataset_id = upload_frame(client, frame)
    body = assert_ok(apply(client, dataset_id, [_op("convert_type", column="is_ok", target_type="boolean")]))
    saved = read_version_csv(dataset_id, body["version_id"])
    assert str(saved.loc[0, "is_ok"]).strip().lower() in ("true", "1")


def test_remove_outliers_iqr_clip(client):
    dataset_id = upload_frame(client, make_frame())
    body = assert_ok(apply(client, dataset_id, [_op("remove_outliers", column="income", method="IQR", action="clip")]))
    effect = body["operations_applied"][0]["effect"]
    assert effect["clipped_rows"] >= 1  # 9999 必须被 clip
    saved = read_version_csv(dataset_id, body["version_id"])
    assert float(saved["income"].max()) < 1000.0
    assert body["rows"] == 10  # clip 不删行


def test_remove_outliers_iqr_remove_rows(client):
    dataset_id = upload_frame(client, make_frame())
    body = assert_ok(apply(client, dataset_id, [_op("remove_outliers", column="income", method="IQR", action="remove_rows")]))
    effect = body["operations_applied"][0]["effect"]
    assert effect["removed_rows"] >= 1
    saved = read_version_csv(dataset_id, body["version_id"])
    assert "9999" not in saved["income"].astype(str).tolist()
    assert body["rows"] == 10 - effect["removed_rows"]


def test_text_strip_and_lowercase(client):
    dataset_id = upload_frame(client, make_frame())
    body = assert_ok(
        apply(
            client,
            dataset_id,
            [
                _op("text_transform", columns=["name"], action="strip"),
                _op("text_transform", columns=["name"], action="lowercase"),
            ],
        )
    )
    saved = read_version_csv(dataset_id, body["version_id"])
    assert str(saved.loc[0, "name"]) == "alice"


# ------------------------------------------------------------
# 特征工程
# ------------------------------------------------------------


def test_date_features_year_month_day(client):
    dataset_id = upload_frame(client, make_frame())
    body = assert_ok(
        apply(
            client,
            dataset_id,
            [
                _op("convert_type", column="joined", target_type="datetime"),
                _op(
                    "date_features",
                    column="joined",
                    features=["year", "month", "day", "day_of_week", "quarter"],
                ),
            ],
        )
    )
    saved = read_version_csv(dataset_id, body["version_id"])
    for feature in ("year", "month", "day", "day_of_week", "quarter"):
        assert f"joined_{feature}" in saved.columns
    assert int(saved.loc[0, "joined_year"]) == 2024
    assert int(saved.loc[0, "joined_month"]) == 1
    # 周一(0) / 2024-01-15
    assert int(saved.loc[0, "joined_day_of_week"]) == 0


def test_date_features_column_conflict(client):
    frame = make_frame().copy()
    frame["joined_year"] = 1999
    dataset_id = upload_frame(client, frame)
    response = apply(
        client,
        dataset_id,
        [_op("date_features", column="joined", features=["year"])],
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "feature_column_conflict"


def test_one_hot_encoding(client):
    dataset_id = upload_frame(client, make_frame())
    body = assert_ok(apply(client, dataset_id, [_op("one_hot_encode", columns=["city"])]))
    saved = read_version_csv(dataset_id, body["version_id"])
    assert "city_北京" in saved.columns
    assert "city_上海" in saved.columns
    assert "city" not in saved.columns
    assert int(saved.loc[0, "city_北京"]) == 1


def test_one_hot_high_cardinality_requires_confirmation(client):
    frame = pd.DataFrame({"code": [f"cat_{i:03d}" for i in range(60)]})
    dataset_id = upload_frame(client, frame)
    response = apply(client, dataset_id, [_op("one_hot_encode", columns=["code"])])
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "high_cardinality_confirmation_required"
    # 明确确认后可执行（warnings 提示）
    response2 = apply(
        client,
        dataset_id,
        [_op("one_hot_encode", columns=["code"], allow_high_cardinality=True)],
    )
    body = assert_ok(response2)
    assert len(body["warnings"]) >= 1
    assert body["columns"] == 60


def test_scale_standardization(client):
    dataset_id = upload_frame(client, make_frame())
    body = assert_ok(apply(client, dataset_id, [_op("scale_numeric", columns=["income"], scale_method="standardization")]))
    saved = read_version_csv(dataset_id, body["version_id"])
    vals = pd.to_numeric(saved["income"], errors="coerce").dropna()
    assert abs(float(vals.mean())) < 1e-9
    assert abs(float(vals.std(ddof=0)) - 1.0) < 1e-6


def test_scale_min_max(client):
    dataset_id = upload_frame(client, make_frame())
    body = assert_ok(apply(client, dataset_id, [_op("scale_numeric", columns=["income"], scale_method="min_max")]))
    saved = read_version_csv(dataset_id, body["version_id"])
    vals = pd.to_numeric(saved["income"], errors="coerce").dropna()
    assert float(vals.min()) == 0.0
    assert float(vals.max()) == 1.0


def test_scale_constant_column_no_divide_by_zero(client):
    dataset_id = upload_frame(client, make_frame())
    body = assert_ok(apply(client, dataset_id, [_op("scale_numeric", columns=["const_score"], scale_method="standardization")]))
    assert any("常量字段" in text for text in body["warnings"])
    saved = read_version_csv(dataset_id, body["version_id"])
    assert set(pd.to_numeric(saved["const_score"], errors="coerce").dropna().tolist()) == {0.0}


# ------------------------------------------------------------
# Plan 顺序 / 原子性
# ------------------------------------------------------------


def test_multi_operation_order_preserved(client):
    dataset_id = upload_frame(client, make_frame())
    ops = [
        _op("drop_duplicates"),
        _op("fill_missing", columns=["age"], strategy="median"),
        _op("one_hot_encode", columns=["city"]),
        _op("drop_columns", columns=["name"]),
    ]
    body = assert_ok(apply(client, dataset_id, ops))
    applied = [item["type"] for item in body["operations_applied"]]
    assert applied == ["drop_duplicates", "fill_missing", "one_hot_encode", "drop_columns"]
    saved = read_version_csv(dataset_id, body["version_id"])
    # 先填充后 OHE：city 被编码；name 被删除
    assert "city" not in saved.columns
    assert "city_北京" in saved.columns
    assert "name" not in saved.columns
    assert len(saved) == 8


def test_apply_atomic_when_middle_operation_fails(client):
    dataset_id = upload_frame(client, make_frame())
    ops = [
        _op("drop_duplicates"),
        _op("drop_columns", columns=["not_exist_column"]),
    ]
    response = apply(client, dataset_id, ops)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "column_not_found"
    # 中途失败：绝不能留下任何版本
    assert version_dir_count(dataset_id) == 0
    assert not (dataset_dir(dataset_id) / "versions").exists()


def test_original_source_never_modified(client):
    dataset_id = upload_frame(client, make_frame())
    original_csv_path = None
    for candidate in dataset_dir(dataset_id).iterdir():
        if candidate.name.lower().endswith((".csv", ".xlsx", ".xls")):
            original_csv_path = candidate
            break
    assert original_csv_path is not None
    before = original_csv_path.read_bytes()

    assert_ok(apply(client, dataset_id, [_op("drop_duplicates")]))
    assert_ok(apply(client, dataset_id, [_op("fill_missing", columns=["age"], strategy="median")]))
    assert_ok(
        apply(
            client,
            dataset_id,
            [_op("remove_outliers", column="income", method="IQR", action="clip")],
            source="original",
        )
    )

    after = original_csv_path.read_bytes()
    assert before == after  # 原始 source 文件字节完全一致


# ------------------------------------------------------------
# 版本管理
# ------------------------------------------------------------


def test_versions_list_after_applies(client):
    dataset_id = upload_frame(client, make_frame())
    v1 = assert_ok(apply(client, dataset_id, [_op("drop_duplicates")]))["version_id"]
    v2 = assert_ok(apply(client, dataset_id, [_op("fill_missing", columns=["age"], strategy="median")]))["version_id"]
    body = assert_ok(client.get(f"/api/datasets/{dataset_id}/versions"))
    listed = [item["version_id"] for item in body["versions"]]
    assert set(listed) == {v1, v2}
    # 按创建时间升序排列
    times = [item["created_at"] for item in body["versions"]]
    assert times == sorted(times)
    rows_by_id = {item["version_id"]: item for item in body["versions"]}
    assert rows_by_id[v1]["parent_version_id"] == "original"
    assert rows_by_id[v1]["rows"] == 8
    assert rows_by_id[v1]["operation_count"] == 1
    assert body["original"]["version_id"] == "original"
    assert len(body["original"]["operations_summary"]) == 0


def test_version_detail(client):
    dataset_id = upload_frame(client, make_frame())
    body = assert_ok(apply(client, dataset_id, [_op("drop_duplicates"), _op("fill_missing", columns=["age"], strategy="median")]))
    vid = body["version_id"]
    detail = assert_ok(client.get(f"/api/datasets/{dataset_id}/versions/{vid}"))
    assert detail["version_id"] == vid
    assert detail["is_original"] is False
    assert detail["rows"] == 8
    assert len(detail["operations"]) == 2
    assert len(detail["column_profiles"]) == 7
    assert len(detail["preview"]) <= 20
    assert detail["quality"]["duplicate_rows"] == 0


def test_eda_supports_version_id(client):
    dataset_id = upload_frame(client, make_frame())
    body = assert_ok(apply(client, dataset_id, [_op("fill_missing", columns=["age"], strategy="median")]))
    vid = body["version_id"]
    eda_original = assert_ok(client.get(f"/api/datasets/{dataset_id}/eda"))
    eda_version = assert_ok(client.get(f"/api/datasets/{dataset_id}/eda?version_id={vid}"))
    # 行数一致（填充不删行）
    assert eda_original["dataset"]["rows"] == eda_version["dataset"]["rows"] == 10
    # 缺省调用不带 version 参数时 behavior 不变
    assert eda_original["success"] is True
    # age 的缺失被填充 → 缺失从 5 降到 3
    assert eda_original["missing_analysis"]["total_missing"] == 5
    assert eda_version["missing_analysis"]["total_missing"] == 3


def test_delete_derived_version(client):
    dataset_id = upload_frame(client, make_frame())
    vid = assert_ok(apply(client, dataset_id, [_op("drop_duplicates")]))["version_id"]
    assert version_dir_count(dataset_id) == 1
    response = client.delete(f"/api/datasets/{dataset_id}/versions/{vid}")
    assert response.status_code == 200
    assert version_dir_count(dataset_id) == 0


def test_original_cannot_be_deleted(client):
    dataset_id = upload_frame(client, make_frame())
    response = client.delete(f"/api/datasets/{dataset_id}/versions/original")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "original_version_protected"


def test_delete_parent_with_child_conflict(client):
    dataset_id = upload_frame(client, make_frame())
    v1 = assert_ok(apply(client, dataset_id, [_op("drop_duplicates")]))["version_id"]
    v2 = assert_ok(
        apply(client, dataset_id, [_op("fill_missing", columns=["age"], strategy="median")], source=v1)
    )["version_id"]
    assert v2 != v1
    # 删除有子版本的父版本 → 409
    response = client.delete(f"/api/datasets/{dataset_id}/versions/{v1}")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "version_has_children"
    # 先删子版本，父版本才可删
    assert client.delete(f"/api/datasets/{dataset_id}/versions/{v2}").status_code == 200
    assert client.delete(f"/api/datasets/{dataset_id}/versions/{v1}").status_code == 200


def test_invalid_and_missing_version_id(client):
    dataset_id = upload_frame(client, make_frame())
    bad = client.get(f"/api/datasets/{dataset_id}/versions/not-a-uuid")
    assert bad.status_code == 400
    assert bad.json()["error"]["code"] == "invalid_version_id"

    fake = str(uuid.uuid4())
    missing = client.get(f"/api/datasets/{dataset_id}/versions/{fake}")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "version_not_found"

    eda_bad = client.get(f"/api/datasets/{dataset_id}/eda?version_id={fake}")
    assert eda_bad.status_code == 404


def test_preview_from_derived_version(client):
    dataset_id = upload_frame(client, make_frame())
    v1 = assert_ok(apply(client, dataset_id, [_op("drop_duplicates")]))["version_id"]
    # 基于 Version 1 继续 preview（不应产生新文件）
    response = preview(
        client,
        dataset_id,
        [_op("fill_missing", columns=["age"], strategy="median")],
        source=v1,
    )
    body = assert_ok(response)
    assert body["source_version_id"] == v1
    assert body["rows_before"] == 8
    assert version_dir_count(dataset_id) == 1  # 仍然是 v1


def test_version_compare(client):
    dataset_id = upload_frame(client, make_frame())
    v1 = assert_ok(apply(client, dataset_id, [_op("drop_duplicates")]))["version_id"]
    v2 = assert_ok(apply(client, dataset_id, [_op("one_hot_encode", columns=["city"])]))["version_id"]
    response = client.post(
        f"/api/datasets/{dataset_id}/versions/compare",
        json={"version_a": "original", "version_b": v2},
    )
    body = assert_ok(response)
    assert body["rows_a"] == 10
    assert body["rows_b"] == 10
    assert "city_北京" in body["generated_columns"]
    assert "city" in body["removed_columns"]

    # 与未删除的 v1 对比
    response2 = client.post(
        f"/api/datasets/{dataset_id}/versions/compare",
        json={"version_a": v1, "version_b": v2},
    )
    assert assert_ok(response2)["columns_a"] == 7


def test_unknown_operation_type_is_rejected(client):
    dataset_id = upload_frame(client, make_frame())
    response = preview(client, dataset_id, [{"type": "eval_code"}])
    assert response.status_code == 422
    assert version_dir_count(dataset_id) == 0
