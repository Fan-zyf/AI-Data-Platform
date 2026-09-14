"""v0.7.2 Grounding Rules 测试。

不调用真实 LLM，仅通过 Mock + compose 验证：
1. 不会产出无来源的具体性能提升数字（"F1 将提升到 X"）。
2. recommendations 全部以「【待验证】」开头。
3. SHAP feature_names 中无 missing indicator 时，不会声称存在 missing indicator。
4. 小样本（train_rows < 50 或 test_rows < 20 或 dataset.rows < 100）时，
   报告 C 节必须包含「指标不稳定」/「不能据此作确定性能判断」措辞。
5. 报告统一为 A/B/C/D 四节结构。
6. 答非所问的强断言：XGBoost / LightGBM 一定提升某指标 / CV std 下降到 X。
"""

from __future__ import annotations

import io
import json
import os

import pytest
from fastapi.testclient import TestClient

from app.agent.analyst_agent import run_analyst
from app.agent.prompts import (
    AGENT_SYSTEM_PROMPT,
    is_small_sample,
)
from app.agent.tools.base import ToolContext, ToolResult
from app.agent.tools.report_tool import ReportTool
from app.llm.llm_client import MockLLMClient, _compose_report, _tag_unverified


# ============================================================
# 工具
# ============================================================


def _ml_data_with_test_metrics(test_metrics, train_rows=18, test_rows=5):
    return {
        "task_type": "classification",
        "best_model": "logistic_regression",
        "train_rows": train_rows,
        "test_rows": test_rows,
        "test_metrics": test_metrics,
        "metrics": {"f1_macro": 0.598, "accuracy": 0.78},
        "cv": {"folds": 5, "primary_metric": "f1_macro"},
    }


def _shap_data(feat_names=None, top=None, with_missing_indicator=False):
    feat_names = feat_names or ["numeric__score", "numeric__age", "categorical__city_北京"]
    if with_missing_indicator:
        feat_names = list(feat_names) + ["missing_indicator__score"]
    return {
        "explainer_type": "linear",
        "top_features": top or ["numeric__score", "numeric__age", "categorical__city_北京"],
        "feature_names": feat_names,
        "importance": [0.744, 0.455, 0.237],
    }


def _eda_data():
    return {
        "rows": 23,
        "n_columns": 7,
        "missing_rate": 0.0435,
        "missing_top": [{"column": "score", "missing_rate": 0.174}],
        "warnings": ["a", "b"],
    }


def _dataset_data():
    return {
        "dataset_id": "ds-x",
        "name": "demo",
        "rows": 23,
        "n_columns": 7,
        "source_format": "csv",
    }


def _mock_payload(q: str, dataset, eda, ml, shap, with_experiment: bool):
    """构造一条完整 tool_results payload，喂给 MockLLMClient。"""
    structured = {
        "dataset": dataset,
        "eda": eda,
        "shap": shap,
        "report": {"ready": True},
    }
    if with_experiment:
        structured["ml"] = ml
    payload_lines = [
        f"用户问题：{q}",
        "",
        "=== 已调用的工具结果（人类可读）===",
        f"- 工具 dataset (status=ok)\n- 工具 eda (status=ok)\n- 工具 shap (status=ok)"
        + ("\n- 工具 ml (status=ok)" if with_experiment else "")
        + "\n- 工具 report (status=ok)",
        "",
        "=== 结构化工具结果（机器可读，供分析用）===",
        "<<<TOOL_RESULTS>>>",
        json.dumps(structured, ensure_ascii=False, indent=2),
        "<<<END_TOOL_RESULTS>>>",
        "",
        "请按 JSON Schema 严格输出。",
    ]
    return "\n".join(payload_lines)


# ============================================================
# 1. System prompt 包含 Grounding Rules
# ============================================================


def test_system_prompt_includes_grounding_rules():
    assert "Grounding Rules" in AGENT_SYSTEM_PROMPT
    assert "Observed Facts" in AGENT_SYSTEM_PROMPT
    assert "Interpretation" in AGENT_SYSTEM_PROMPT
    assert "Recommendations" in AGENT_SYSTEM_PROMPT
    assert "A. 已观测事实" in AGENT_SYSTEM_PROMPT
    assert "B. 谨慎解释" in AGENT_SYSTEM_PROMPT
    assert "C. 当前局限" in AGENT_SYSTEM_PROMPT
    assert "D. 建议验证的下一步" in AGENT_SYSTEM_PROMPT
    assert "不得附带未经验证的具体性能提升数字" in AGENT_SYSTEM_PROMPT
    # 禁止清单
    assert "把相关性写成因果" in AGENT_SYSTEM_PROMPT
    assert "声称「XGBoost / LightGBM / SMOTE / class_weight 一定能改善当前模型" in AGENT_SYSTEM_PROMPT
    assert "F1 将提升到 Y" in AGENT_SYSTEM_PROMPT


# ============================================================
# 2. is_small_sample 判定
# ============================================================


def test_is_small_sample_by_train_rows():
    assert is_small_sample([
        {"name": "ml", "data": {"train_rows": 30, "test_rows": 100}, "status": "ok"}
    ])


def test_is_small_sample_by_test_rows():
    assert is_small_sample([
        {"name": "ml", "data": {"train_rows": 200, "test_rows": 10}, "status": "ok"}
    ])


def test_is_small_sample_by_dataset_rows():
    assert is_small_sample([
        {"name": "dataset", "data": {"rows": 60}, "status": "ok"}
    ])


def test_not_small_sample_when_rows_large():
    assert not is_small_sample([
        {"name": "ml", "data": {"train_rows": 800, "test_rows": 200, "cv": {"folds": 5}}, "status": "ok"},
        {"name": "dataset", "data": {"rows": 1000}, "status": "ok"},
    ])


# ============================================================
# 3. Mock 报告不会生成无来源的具体性能提升数字
# ============================================================


_STRONG_CLAIM_PATTERNS = [
    "F1 将提升到",
    "AUC 将提升到",
    "CV std 将下降到",
    "F1提升到",
    "将提升到 0.",
    "F1 提升到 0.",
    "accuracy 将达到 0.",
    "一定会改善",
    "一定优于",
    "必然提升",
]


def test_mock_report_does_not_claim_specific_future_metric():
    payload = _mock_payload(
        q="为什么这个模型效果不好？",
        dataset=_dataset_data(),
        eda=_eda_data(),
        ml=_ml_data_with_test_metrics(
            {"accuracy": 0.6, "f1_macro": 0.375}, train_rows=18, test_rows=5
        ),
        shap=_shap_data(),
        with_experiment=True,
    )
    out = MockLLMClient().chat(system=AGENT_SYSTEM_PROMPT, user=payload)
    parsed = json.loads(out)
    text_blob = (
        parsed["answer"] + " "
        + " ".join(parsed["insights"]) + " "
        + " ".join(parsed["recommendations"])
    )
    for pat in _STRONG_CLAIM_PATTERNS:
        assert pat not in text_blob, f"forbidden strong claim found: {pat}\n--- blob ---\n{text_blob}"


# ============================================================
# 4. recommendations 全部以【待验证】开头
# ============================================================


def test_all_recommendations_tagged_unverified():
    for case in [
        {
            "ml": _ml_data_with_test_metrics({"accuracy": 0.5, "f1_macro": 0.3}),
            "eda": _eda_data(),
            "shap": _shap_data(),
        },
        {
            "ml": _ml_data_with_test_metrics({"accuracy": 0.92, "f1_macro": 0.91},
                                             train_rows=200, test_rows=80),
            "eda": {**_eda_data(), "rows": 1000, "missing_top": []},
            "shap": _shap_data(),
        },
        {
            "ml": None,
            "eda": _eda_data(),
            "shap": None,
        },
    ]:
        tool_results = {
            "dataset": _dataset_data(),
            "eda": case["eda"] or {},
            "ml": case["ml"] or {},
            "shap": case["shap"] or {},
        }
        out = MockLLMClient().chat(
            system=AGENT_SYSTEM_PROMPT,
            user=_mock_payload("x", _dataset_data(), case["eda"] or {}, case["ml"] or {}, case["shap"] or {}, bool(case["ml"])),
        )
        parsed = json.loads(out)
        recs = parsed["recommendations"]
        assert recs, f"recommendations should not be empty: {case}"
        for r in recs:
            assert r.startswith("【待验证】"), f"recommendation not tagged: {r!r}"


def test_tag_unverified_helper_idempotent():
    assert _tag_unverified("【待验证】x") == "【待验证】x"
    assert _tag_unverified("plain") == "【待验证】plain"
    assert _tag_unverified("") == ""
    assert _tag_unverified(None) == ""


# ============================================================
# 5. 无 missing indicator 时不声称存在
# ============================================================


def test_no_missing_indicator_claim_when_not_in_feature_names():
    out = MockLLMClient().chat(
        system=AGENT_SYSTEM_PROMPT,
        user=_mock_payload(
            "哪些特征重要？",
            _dataset_data(),
            _eda_data(),
            _ml_data_with_test_metrics({"accuracy": 0.92}),
            _shap_data(with_missing_indicator=False),
            with_experiment=True,
        ),
    )
    parsed = json.loads(out)
    text_blob = parsed["answer"] + " " + " ".join(parsed["insights"])
    # 「已观测事实」中不应出现"feature_names 中包含缺失指示"这类事实性声明
    assert "包含缺失指示类特征" not in text_blob
    assert "SHAP feature_names 中包含缺失" not in text_blob
    # 仅当 feature_names 真的含 missing/indicator 时才出现这条事实
    for r in parsed["recommendations"]:
        # recommendations 可以建议"使用缺失指示列"，但不得声称"模型已使用"
        assert "模型已使用缺失指示" not in r
        assert "已采用缺失指示" not in r


def test_acknowledges_missing_indicator_when_present():
    out = MockLLMClient().chat(
        system=AGENT_SYSTEM_PROMPT,
        user=_mock_payload(
            "哪些特征重要？",
            _dataset_data(),
            _eda_data(),
            _ml_data_with_test_metrics({"accuracy": 0.92}),
            _shap_data(with_missing_indicator=True),
            with_experiment=True,
        ),
    )
    parsed = json.loads(out)
    text_blob = parsed["answer"]
    assert "缺失指示" in text_blob


# ============================================================
# 6. 小样本时 answer 必须包含"指标不稳定"提示
# ============================================================


def test_small_sample_warning_in_mock_report():
    out = MockLLMClient().chat(
        system=AGENT_SYSTEM_PROMPT,
        user=_mock_payload(
            "模型效果如何？",
            _dataset_data(),  # rows=23
            _eda_data(),
            _ml_data_with_test_metrics({"accuracy": 0.6}, train_rows=18, test_rows=5),
            _shap_data(),
            with_experiment=True,
        ),
    )
    parsed = json.loads(out)
    answer = parsed["answer"]
    assert "### C. 当前局限" in answer
    assert "指标不稳定" in answer or "不能据此对模型性能作确定判断" in answer
    # B 节 / D 节必须使用软措辞
    assert "可能" in answer or "建议尝试" in answer or "需要进一步验证" in answer


# ============================================================
# 7. 报告统一为 A/B/C/D 四节
# ============================================================


def test_report_has_ABCD_sections():
    out = MockLLMClient().chat(
        system=AGENT_SYSTEM_PROMPT,
        user=_mock_payload(
            "总结一下",
            _dataset_data(),
            _eda_data(),
            _ml_data_with_test_metrics({"accuracy": 0.7}),
            _shap_data(),
            with_experiment=True,
        ),
    )
    parsed = json.loads(out)
    answer = parsed["answer"]
    for marker in (
        "### A. 已观测事实",
        "### B. 谨慎解释",
        "### C. 当前局限",
        "### D. 建议验证的下一步",
    ):
        assert marker in answer, f"missing section: {marker}"


# ============================================================
# 8. ReportTool.compose 也遵守 Grounding Rules
# ============================================================


def test_report_tool_compose_uses_unverified_prefix_and_ABCD():
    eda = EDATool_like = _eda_data()
    ml = _ml_data_with_test_metrics({"accuracy": 0.6, "f1_macro": 0.375})
    shap = _shap_data()
    dataset = _dataset_data()
    tool_results = [
        ToolResult(name="dataset", status="ok", data=dataset, summary="ok"),
        ToolResult(name="eda", status="ok", data=eda, summary="ok"),
        ToolResult(name="ml", status="ok", data=ml, summary="ok"),
        ToolResult(name="shap", status="ok", data=shap, summary="ok"),
    ]
    report = ReportTool().compose(tool_results, "为什么效果不好？")
    recs = report["recommendations"]
    assert recs, "recommendations should not be empty"
    for r in recs:
        assert r.startswith("【待验证】"), f"recommendation not tagged: {r!r}"
    text_blob = report["answer"] + " " + " ".join(recs) + " " + " ".join(report["insights"])
    for pat in _STRONG_CLAIM_PATTERNS:
        assert pat not in text_blob
    # 小样本应当出现在 C 节
    assert "### C. 当前局限" in report["answer"]
    assert "指标不稳定" in report["answer"]


def test_report_tool_compose_does_not_invent_missing_indicator():
    eda = _eda_data()
    ml = _ml_data_with_test_metrics({"accuracy": 0.92})
    shap = _shap_data(with_missing_indicator=False)  # 显式无 missing indicator
    tool_results = [
        ToolResult(name="dataset", status="ok", data=_dataset_data(), summary="ok"),
        ToolResult(name="eda", status="ok", data=eda, summary="ok"),
        ToolResult(name="ml", status="ok", data=ml, summary="ok"),
        ToolResult(name="shap", status="ok", data=shap, summary="ok"),
    ]
    report = ReportTool().compose(tool_results, "哪些特征影响大？")
    text_blob = report["answer"] + " " + " ".join(report["insights"])
    assert "包含缺失指示类特征" not in text_blob
    assert "SHAP feature_names 中包含缺失" not in text_blob


# ============================================================
# 9. run_analyst（Mock）也走 A/B/C/D + 【待验证】前缀
# ============================================================


def test_run_analyst_returns_grounded_report_via_api(client):
    """走完整 API 路径（Mock 模式），验证 answer / recommendations 落地形态。"""
    import io
    import uuid as _uuid

    # 直接上传一个 20 行 CSV（无 experiment）→ 强制走小样本判定路径
    csv_bytes = (
        b"x1,x2,label\n"
        + b"".join(f"{i},{0.1*i},{'yes' if i%2==0 else 'no'}\n".encode() for i in range(20))
    )
    up = client.post(
        "/api/data/upload",
        files={"file": ("grounding_demo.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert up.status_code == 200, up.text
    ds_id = up.json()["dataset_id"]

    resp = client.post(
        "/api/agent/analyze",
        json={"question": "为什么这个数据集有问题？", "dataset_id": ds_id},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True
    assert body["app_version"] == "0.7.2"
    assert body["llm"]["is_mock"] is True
    for marker in (
        "### A. 已观测事实",
        "### B. 谨慎解释",
        "### C. 当前局限",
        "### D. 建议验证的下一步",
    ):
        assert marker in body["answer"], f"missing section: {marker}\nanswer:\n{body['answer']}"
    for r in body["recommendations"]:
        assert r.startswith("【待验证】"), f"recommendation not tagged: {r!r}"


# ============================================================
# 10. v0.7.2 发布收尾：app_version / FastAPI version / llm.model / no API-key leak
# ============================================================


def test_v072_release_fastapi_app_version():
    """FastAPI app 的 OpenAPI version 必须与发布版本同步。"""
    from app.main import app
    assert app.version == "0.7.2", f"app.version should be '0.7.2', got {app.version!r}"


def test_v072_release_root_endpoint_version():
    """GET / 应返回同步后的 version 字段。"""
    from app.main import app
    with TestClient(app) as c:
        body = c.get("/").json()
    assert body.get("version") == "0.7.2", f"/ version should be '0.7.2', got {body.get('version')!r}"


def test_v072_release_openapi_info_version():
    """/openapi.json 的 info.version 必须为 0.7.2。"""
    from app.main import app
    with TestClient(app) as c:
        info = c.get("/openapi.json").json()["info"]
    assert info.get("version") == "0.7.2", f"openapi info.version should be '0.7.2', got {info.get('version')!r}"


def test_v072_release_analyze_app_version_and_llm_model_in_mock_mode():
    """analyze 响应（Mock 模式）：app_version=0.7.2、llm.model=None、不暴露 api_key。"""
    from app.llm.llm_client import MockLLMClient

    from app.main import app as _app

    class _DummyMock(MockLLMClient):
        """test-only 假 client：让 run_analyst 走 mock 路径（不调真实 LLM）。"""
        is_mock = True
        def __init__(self):
            super().__init__()

    with TestClient(_app) as c:
        # 准备一个 20 行 CSV（mock 模式判定为小样本）
        csv_bytes = (
            b"x1,x2,label\n"
            + b"".join(f"{i},{0.1*i},{'yes' if i%2==0 else 'no'}\n".encode() for i in range(20))
        )
        up = c.post(
            "/api/data/upload",
            files={"file": ("release_demo.csv", io.BytesIO(csv_bytes), "text/csv")},
        )
        assert up.status_code == 200, up.text
        ds_id = up.json()["dataset_id"]

        # 通过 monkeypatch 强制 run_analyst 使用 MockLLMClient → 不消耗真实 LLM
        from app.agent import analyst_agent as _aa
        original = _aa.get_default_client
        _aa.get_default_client = lambda: _DummyMock()  # type: ignore[assignment]
        try:
            resp = c.post(
                "/api/agent/analyze",
                json={"question": "总结一下", "dataset_id": ds_id},
            )
        finally:
            _aa.get_default_client = original  # type: ignore[assignment]

    assert resp.status_code == 200, resp.text
    body = resp.json()
    # 1) app_version
    assert body.get("app_version") == "0.7.2", f"app_version should be '0.7.2', got {body.get('app_version')!r}"
    # 2) llm.is_mock = True（mock 模式）
    assert body["llm"]["is_mock"] is True
    # 3) llm.model = None（mock 模式无真实模型名）
    assert body["llm"].get("model") is None, f"mock mode llm.model should be None, got {body['llm'].get('model')!r}"
    # 4) 不暴露 api_key（响应里所有字符串中不应出现 LLM_API_KEY 字面值）
    import json as _json
    serialized = _json.dumps(body, ensure_ascii=False)
    # os.environ 的 LLM_API_KEY 当前值
    leaked = os.environ.get("LLM_API_KEY") or ""
    if leaked:
        assert leaked not in serialized, "LLM_API_KEY value leaked in analyze response"


def test_v072_release_no_api_key_string_in_analyze_response():
    """analyze 响应中不得出现 LLM_API_KEY key 名字面（避免误传）。"""
    import io
    import json as _json

    from app.main import app as _app

    with TestClient(_app) as c:
        csv_bytes = (
            b"x1,x2,label\n"
            + b"".join(f"{i},{0.1*i},{'yes' if i%2==0 else 'no'}\n".encode() for i in range(20))
        )
        up = c.post(
            "/api/data/upload",
            files={"file": ("leakcheck.csv", io.BytesIO(csv_bytes), "text/csv")},
        )
        ds_id = up.json()["dataset_id"]
        resp = c.post(
            "/api/agent/analyze",
            json={"question": "总结", "dataset_id": ds_id},
        )
    body = resp.json()
    serialized = _json.dumps(body, ensure_ascii=False)
    # LLM_API_KEY 字符串不应该出现在响应里
    assert "LLM_API_KEY" not in serialized, "LLM_API_KEY key name should not appear in analyze response"
