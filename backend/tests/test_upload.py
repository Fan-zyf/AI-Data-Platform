"""数据上传接口的基础自动化测试。

覆盖：CSV 成功 / Excel 成功 / 非法扩展名 / 空文件 / 损坏文件 / 超限文件 / 健康检查。
运行方式（backend 目录）：.venv\\Scripts\\python.exe -m pytest tests -v
"""

import io
import pathlib

import pandas as pd

BACKEND_DIR = pathlib.Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent

DEMO_CSV = PROJECT_ROOT / "data" / "sample" / "demo.csv"


def _upload(client, filename: str, content: bytes, mime: str = "application/octet-stream"):
    """发起 multipart 上传请求的便捷方法。"""
    return client.post(
        "/api/data/upload",
        files={"file": (filename, content, mime)},
    )


def test_health_still_ok(client):
    """基础健康检查在新增功能后仍然正常。"""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["message"] == "AI Data Platform backend is running"


def test_csv_upload_success(client):
    """CSV 上传成功并返回完整画像结构。"""
    content = DEMO_CSV.read_bytes()
    res = _upload(client, "demo.csv", content, "text/csv")

    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True

    dataset = data["dataset"]
    assert dataset["file_name"] == "demo.csv"
    assert dataset["file_type"] == "csv"
    assert dataset["rows"] == 23
    assert dataset["columns"] == 7

    quality = data["quality"]
    assert quality["total_missing"] == 7
    assert quality["columns_with_missing"] == 3
    assert quality["duplicate_rows"] == 1
    assert quality["empty_columns"] == 0
    assert quality["constant_columns"] == 0

    # 画像结果已预先通过服务自检，此处做结构级断言
    types = {p["column_name"]: p["inferred_type"] for p in data["column_profiles"]}
    assert types["id"] == "numeric"
    assert types["city"] == "categorical"
    assert types["joined"] == "datetime"
    assert types["is_active"] == "boolean"
    assert types["notes"] == "text"

    assert len(data["preview"]) == 20  # 超过 20 行时截断预览
    for row in data["preview"]:
        assert isinstance(row, dict)
        # 确保整列字段名齐全
        assert set(row.keys()) == set(types.keys())

    assert any("重复数据" in w for w in data["warnings"])
    assert any("高基数标识符" in w for w in data["warnings"])


def test_small_csv_preview_not_truncated(client):
    """行数小于预览行数时，预览返回全部行。"""
    small = b"a,b\n1,2\n3,4\n"
    res = _upload(client, "small.csv", small)
    assert res.status_code == 200
    data = res.json()
    assert data["dataset"]["rows"] == 2
    assert len(data["preview"]) == 2


def test_excel_xlsx_upload_success(client):
    """Excel(.xlsx) 上传成功。"""
    df = pd.DataFrame(
        {
            "product": ["phone", "laptop", "phone", "tablet"],
            "sales": [100, 250, 100, None],
            "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
        }
    )
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False, engine="openpyxl")

    res = _upload(client, "sales.xlsx", buffer.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["dataset"]["file_type"] == "xlsx"
    assert data["dataset"]["rows"] == 4
    assert data["dataset"]["columns"] == 3
    # Excel 数字列缺失值被正确统计
    assert data["quality"]["total_missing"] == 1


def test_invalid_extension_rejected(client):
    """非法扩展名（如 .txt）应被拒绝。"""
    res = _upload(client, "notes.txt", b"hello world")
    assert res.status_code == 400
    body = res.json()
    assert body["success"] is False
    assert body["error"]["code"] == "unsupported_file_type"
    assert "不支持" in body["error"]["message"]


def test_empty_file_rejected(client):
    """空文件应被拒绝。"""
    res = _upload(client, "empty.csv", b"")
    assert res.status_code == 400
    body = res.json()
    assert body["success"] is False
    assert body["error"]["code"] == "empty_file"


def test_garbage_csv_rejected(client):
    """无法解析的 CSV（乱码/二进制内容）应给出友好错误，而不是 traceback。"""
    res = _upload(client, "broken.csv", b"\x00\xff\xfe\xfd\x00 garbage \x01\x02")
    assert res.status_code in (400, 422)
    body = res.json()
    assert body["success"] is False
    assert body["error"]["code"] in ("csv_encoding_error", "csv_parse_error", "csv_empty", "no_rows")
    assert "Traceback" not in res.text


def test_broken_xlsx_rejected(client):
    """损坏的 Excel 文件应被拒绝。"""
    res = _upload(client, "broken.xlsx", b"PK\x03\x04 not-a-real-zip")
    assert res.status_code == 422
    body = res.json()
    assert body["success"] is False
    assert body["error"]["code"] == "excel_parse_error"


def test_file_too_large_rejected(client, monkeypatch):
    """超过大小上限的文件应被拒绝（413）。"""
    from app.core.config import settings

    monkeypatch.setattr(settings, "data_max_upload_mb", 0)  # 上限改为 0MB
    res = _upload(client, "big.csv", b"a\n1")
    assert res.status_code == 413
    body = res.json()
    assert body["success"] is False
    assert body["error"]["code"] == "file_too_large"


def test_missing_filename_returns_400(client):
    """未携带文件名的请求应给出结构化错误。"""
    res = client.post("/api/data/upload")
    # FastAPI 参数校验失败时会返回 422 而非我们自定义的 400
    assert res.status_code in (400, 422)
