"""pytest 共享夹具：将 backend 目录加入 sys.path 并提供测试客户端。

- client：FastAPI 测试客户端；
- _isolated_dataset_runtime：每个测试使用独立临时目录作为 Dataset Session 存储，
  避免测试之间相互污染，也避免把临时用户数据写入 backend/runtime。
"""

import pathlib
import sys

import pytest

BACKEND_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_dataset_runtime(tmp_path, monkeypatch):
    """把 Dataset Session 根目录指向每个测试独立的临时目录。"""
    monkeypatch.setattr(settings, "dataset_runtime_dir", str(tmp_path / "datasets"))
    yield


@pytest.fixture()
def client():
    """FastAPI 测试客户端。"""
    with TestClient(app) as c:
        yield c
