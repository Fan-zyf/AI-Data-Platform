"""pytest 共享夹具：将 backend 目录加入 sys.path 并提供测试客户端。"""

import pathlib
import sys

import pytest

BACKEND_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture()
def client():
    """FastAPI 测试客户端。"""
    with TestClient(app) as c:
        yield c
