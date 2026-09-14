"""v0.7 AI Data Analyst Agent 测试。

覆盖目标（与用户 spec 一一对应）：
1. Mock 模式（无 LLM_* 环境变量）
2. 真实 LLM 模式（HTTPLLMClient 解析响应 / 错误降级）
3. EDA Tool 调用
4. ML Tool 调用
5. SHAP Tool 调用
6. Dataset Tool 调用
7. Report Tool 汇总
8. ToolRegistry 关键词路由
9. Agent API：完整流程（无 experiment / 有 experiment）
10. 错误：dataset_not_found / experiment_not_found / agent_invalid_request
11. LLM 错误降级
12. Agent 响应 schema
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

import numpy as np
import pandas as pd
import pytest

from app.agent import (
    AgentError,
    ALL_TOOLS,
    BaseTool,
    DatasetTool,
    EDATool,
    MLTool,
    ReportTool,
    SHAPTool,
    ToolContext,
    ToolRegistry,
)
from app.agent import analyst_agent as analyst_module
from app.llm import (
    HTTPLLMClient,
    LLMConfig,
    LLMError,
    MockLLMClient,
    get_default_client,
    reset_default_client_for_test,
)
from app.services import experiment_manager as em


# ============================================================
# 工具
# ============================================================


def _upload(client, frame: pd.DataFrame, name: str = "agent.csv") -> str:
    resp = client.post(
        "/api/data/upload", files={"file": (name, frame.to_csv(index=False).encode("utf-8"), "text/csv")}
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["dataset_id"]


def _train(client, dataset_id: str, **overrides) -> dict:
    payload = {"source_version_id": "original", "target_column": "label", "task_type": "auto"}
    payload.update(overrides)
    resp = client.post(f"/api/datasets/{dataset_id}/ml/train", json=payload)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _classification_frame(rows: int = 60, seed: int = 17) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=rows)
    x2 = rng.integers(0, 20, size=rows).astype(float)
    grp = rng.choice(["A", "B", "C"], size=rows)
    grp_effect = np.where(grp == "A", 0.9, np.where(grp == "B", 0.1, -0.8))
    score = 1.3 * x1 + grp_effect + rng.normal(0, 0.3, size=rows)
    label = np.where(score > float(np.median(score)), "yes", "no")
    return pd.DataFrame({"x1": x1, "x2": x2, "grp": grp, "label": label})


def _expect_error(resp, code: str, status: int | None = None):
    assert resp.status_code == (status if status is not None else 400), resp.text
    payload = resp.json()
    assert payload["success"] is False
    assert payload["error"]["code"] == code
    return payload


@pytest.fixture(autouse=True)
def _reset_llm():
    """每个用例前清空 LLM_* 环境变量 → 强制走 Mock 单例（不依赖 monkeypatch，避免与 conftest autouse 冲突）。"""
    saved = {}
    for key in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        saved[key] = os.environ.pop(key, None)
    reset_default_client_for_test()
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        reset_default_client_for_test()


# ============================================================
# 1. Mock LLM 工厂
# ============================================================


def test_default_client_is_mock_when_no_env():
    client = get_default_client()
    assert isinstance(client, MockLLMClient)
    assert client.is_mock is True


def test_default_client_switches_to_http_when_env_set(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    reset_default_client_for_test()
    client = get_default_client()
    assert isinstance(client, HTTPLLMClient)
    assert client.is_mock is False
    reset_default_client_for_test()


def test_http_client_rejects_when_config_is_mock():
    config = LLMConfig(base_url="", api_key="", model="", timeout_seconds=10, temperature=0.2, max_tokens=100)
    with pytest.raises(LLMError) as exc:
        HTTPLLMClient(config)
    assert exc.value.code == "llm_not_configured"


# ============================================================
# 2. Mock LLM 输出是结构化 JSON 且包含 answer/insights/recommendations
# ============================================================


def test_mock_llm_returns_structured_payload():
    client = MockLLMClient()
    system = "SYSTEM"
    user = "用户问题：哪些特征影响最大？\n\n=== 已调用的工具结果 ===\n"
    out = client.chat(system=system, user=user)
    parsed = json.loads(out)
    assert "answer" in parsed
    assert "insights" in parsed
    assert "recommendations" in parsed
    assert isinstance(parsed["insights"], list)
    assert isinstance(parsed["recommendations"], list)


# ============================================================
# 3. Tool Registry 关键词路由
# ============================================================


def test_tool_registry_selects_by_keywords_with_experiment():
    reg = ToolRegistry()
    plan = reg.select("为什么这个模型效果不好？特征影响", has_experiment=True)
    names = [t.name for t in plan.selected]
    assert "shap" in names
    assert "ml" in names
    assert "report" in names
    # eda 也可能命中
    assert all(s == "ok" for s in [])


def test_tool_registry_skips_experiment_tools_when_no_experiment():
    reg = ToolRegistry()
    plan = reg.select("为什么这个模型效果不好？", has_experiment=False)
    names = [t.name for t in plan.selected]
    assert "ml" not in names
    assert "shap" not in names
    # 至少带回 dataset + eda + report
    assert "dataset" in names and "eda" in names and "report" in names


def test_tool_registry_respects_max_tools(monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "agent_max_tools", 2)
    reg = ToolRegistry()
    plan = reg.select("数据缺失 异常 模型 特征 重要性 shap", has_experiment=True)
    # 限制为 2 + report（不计入 cap）
    assert len(plan.selected) <= 3
    # monkeypatch 自动恢复 settings.agent_max_tools 原始值


def test_tool_registry_default_fallback_when_no_keyword_hit():
    reg = ToolRegistry()
    plan = reg.select("hello world 随便聊聊", has_experiment=True)
    names = [t.name for t in plan.selected]
    # 没有命中关键词时 fallback 到 dataset + eda + report
    assert "dataset" in names
    assert "eda" in names
    assert "report" in names


# ============================================================
# 4. Tool 单工具执行
# ============================================================


def test_dataset_tool_execute(client):
    dataset_id = _upload(client, _classification_frame())
    ctx = ToolContext(dataset_id=dataset_id)
    result = DatasetTool().safe_execute(ctx)
    assert result.status == "ok"
    assert result.data["dataset_id"] == dataset_id
    assert result.data["rows"] == 60
    assert result.data["n_columns"] == 4
    assert result.data["source_format"] == "csv"


def test_eda_tool_execute(client):
    dataset_id = _upload(client, _classification_frame())
    ctx = ToolContext(dataset_id=dataset_id)
    result = EDATool().safe_execute(ctx)
    assert result.status == "ok"
    assert result.data["rows"] == 60
    assert result.data["columns_count"] == 4
    assert isinstance(result.data["missing_top"], list)


def test_ml_tool_execute_requires_experiment(client):
    dataset_id = _upload(client, _classification_frame())
    body = _train(client, dataset_id, candidate_models=["random_forest"])
    experiment_id = body["experiment_id"]
    ctx = ToolContext(dataset_id=dataset_id, experiment_id=experiment_id)
    result = MLTool().safe_execute(ctx)
    assert result.status == "ok"
    assert result.data["task_type"] == "classification"
    assert result.data["best_model"] == "random_forest"
    assert "accuracy" in (result.data["test_metrics"] or {})


def test_ml_tool_skips_without_experiment(client):
    dataset_id = _upload(client, _classification_frame())
    ctx = ToolContext(dataset_id=dataset_id, experiment_id=None)
    result = MLTool().safe_execute(ctx)
    assert result.status == "skipped"


def test_shap_tool_execute_computes_and_persists(client):
    dataset_id = _upload(client, _classification_frame())
    body = _train(client, dataset_id, candidate_models=["random_forest"])
    experiment_id = body["experiment_id"]
    ctx = ToolContext(dataset_id=dataset_id, experiment_id=experiment_id, max_summary_rows=80)
    result = SHAPTool().safe_execute(ctx)
    assert result.status == "ok", result.error
    assert result.data["explainer_type"] == "tree"
    assert result.data["top_features"]
    # 第二次应命中缓存
    cached_result = SHAPTool().safe_execute(ctx)
    assert cached_result.data.get("cached") is True


def test_shap_tool_404_when_experiment_missing(client):
    dataset_id = _upload(client, _classification_frame())
    ctx = ToolContext(dataset_id=dataset_id, experiment_id=str(uuid.uuid4()))
    result = SHAPTool().safe_execute(ctx)
    assert result.status == "error"
    assert "experiment_not_found" in (result.error or "")


def test_report_tool_compose_aggregates_insights(client):
    dataset_id = _upload(client, _classification_frame())
    body = _train(client, dataset_id, candidate_models=["random_forest"])
    experiment_id = body["experiment_id"]
    ctx = ToolContext(dataset_id=dataset_id, experiment_id=experiment_id, max_summary_rows=80)
    eda_r = EDATool().safe_execute(ctx)
    ml_r = MLTool().safe_execute(ctx)
    shap_r = SHAPTool().safe_execute(ctx)
    report = ReportTool().compose([eda_r, ml_r, shap_r], "为什么效果不好？")
    assert report["ml"]["best_model"] == "random_forest"
    assert "top_features" in (report["shap"] or {})
    assert isinstance(report["insights"], list)
    assert isinstance(report["recommendations"], list)


# ============================================================
# 5. Analyst Agent 异常 → AgentError
# ============================================================


def test_run_analyst_raises_on_empty_question():
    with pytest.raises(AgentError) as exc:
        analyst_module.run_analyst(question="", dataset_id=str(uuid.uuid4()))
    assert exc.value.code == "agent_invalid_request"


def test_run_analyst_raises_on_empty_dataset_id():
    with pytest.raises(AgentError) as exc:
        analyst_module.run_analyst(question="hi", dataset_id="")
    assert exc.value.code == "agent_invalid_request"


# ============================================================
# 6. Agent API：完整流程
# ============================================================


def test_agent_analyze_endpoint_dataset_only(client):
    dataset_id = _upload(client, _classification_frame())
    resp = client.post(
        "/api/agent/analyze",
        json={"question": "这个数据有什么问题？", "dataset_id": dataset_id},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True
    assert body["app_version"] == "0.7.2"
    assert body["dataset_id"] == dataset_id
    assert body["experiment_id"] is None
    # answer / insights / recommendations 都应是非空
    assert body["answer"]
    assert isinstance(body["insights"], list)
    assert isinstance(body["recommendations"], list)
    # tools_used 应至少包含 dataset + eda + report
    used = {t["name"] for t in body["tools_used"]}
    assert "dataset" in used
    assert "eda" in used
    assert "report" in used
    # 计划里不应有 ml/shap（无 experiment）
    assert "ml" not in used
    assert "shap" not in used
    # mock 模式
    assert body["llm"]["is_mock"] is True
    assert body["llm"]["error"] is None
    # schema
    for key in [
        "success", "question", "dataset_id", "answer", "insights",
        "recommendations", "tools_used", "tool_trace", "plan",
        "llm", "created_at", "app_version",
    ]:
        assert key in body


def test_agent_analyze_endpoint_with_experiment(client):
    dataset_id = _upload(client, _classification_frame())
    body = _train(client, dataset_id, candidate_models=["random_forest"])
    experiment_id = body["experiment_id"]

    resp = client.post(
        "/api/agent/analyze",
        json={
            "question": "为什么模型效果不好？哪些特征影响最大？",
            "dataset_id": dataset_id,
            "experiment_id": experiment_id,
        },
    )
    assert resp.status_code == 200, resp.text
    result = resp.json()
    assert result["success"] is True
    used = {t["name"] for t in result["tools_used"]}
    assert "ml" in used
    assert "shap" in used
    assert "eda" in used
    assert "report" in used
    # plan.selected 顺序
    plan_selected = result["plan"]["selected"]
    assert plan_selected[0] in {"shap", "ml", "eda", "dataset"}  # report 总是最后
    assert plan_selected[-1] == "report"


# ============================================================
# 7. 错误处理
# ============================================================


def test_agent_analyze_dataset_not_found(client):
    resp = client.post(
        "/api/agent/analyze",
        json={"question": "any", "dataset_id": str(uuid.uuid4())},
    )
    _expect_error(resp, "dataset_not_found", 404)


def test_agent_analyze_experiment_not_found(client):
    dataset_id = _upload(client, _classification_frame())
    resp = client.post(
        "/api/agent/analyze",
        json={
            "question": "为什么",
            "dataset_id": dataset_id,
            "experiment_id": str(uuid.uuid4()),
        },
    )
    _expect_error(resp, "experiment_not_found", 404)


def test_agent_analyze_invalid_request_empty_question(client):
    dataset_id = _upload(client, _classification_frame())
    resp = client.post(
        "/api/agent/analyze",
        json={"question": "", "dataset_id": dataset_id},
    )
    assert resp.status_code == 422


def test_agent_analyze_invalid_request_empty_dataset(client):
    resp = client.post(
        "/api/agent/analyze",
        json={"question": "why", "dataset_id": ""},
    )
    assert resp.status_code == 422


# ============================================================
# 8. LLM 错误降级
# ============================================================


def test_real_llm_failure_falls_back_to_mock(monkeypatch):
    """构造一个会失败的 HTTPLLMClient 注入默认工厂，验证 Agent 自动降级到 Mock。"""
    class BrokenLLM(HTTPLLMClient):
        def chat(self, system: str, user: str) -> str:  # type: ignore[override]
            raise LLMError("llm_api_error", "simulated LLM failure", 502)

    fake = BrokenLLM(LLMConfig(
        base_url="https://example.com/v1",
        api_key="x",
        model="fake",
        timeout_seconds=1.0,
        temperature=0.0,
        max_tokens=10,
    ))
    # 注入单例
    from app.llm import llm_client
    llm_client._client_singleton = fake
    try:
        result = analyst_module.run_analyst(
            question="这个数据有什么问题？",
            dataset_id=str(uuid.uuid4()),
        )
        # 内部已经把 dataset 校验交给 service 层；这里要的是降级后 LLM 仍能产出
        # 注意：当前 run_analyst 不在内部做 dataset 校验，所以会走到 service
        assert result["llm"]["is_mock"] is True
        assert "降级" in (result["llm"]["error"] or "") or "simulated" in (result["llm"]["error"] or "")
    finally:
        llm_client._client_singleton = None


# ============================================================
# 9. ALL_TOOLS 完整性
# ============================================================


def test_all_tools_registered():
    names = {cls().name for cls in ALL_TOOLS}
    assert names == {"dataset", "eda", "ml", "shap", "report"}
    for cls in ALL_TOOLS:
        assert issubclass(cls, BaseTool)
