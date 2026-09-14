"""v0.7 Analyst Agent 主循环。

调用流程：
1. 构造 ToolContext；
2. ToolRegistry 选工具 → 执行（同步串行）；
3. ReportTool 汇总（结构化）；
4. 调用 LLM（失败时自动回退到 Mock 重发）；
5. 解析 LLM 响应（容忍格式错误）→ 返回统一结构。
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from app.agent.prompts import build_prompt_payload
from app.agent.tool_registry import ToolRegistry
from app.agent.tools.base import ToolContext
from app.agent.tools.report_tool import ReportTool
from app.llm import LLMError, get_default_client
from app.llm.llm_client import HTTPLLMClient, MockLLMClient

logger = logging.getLogger(__name__)


def _safe_model_name(client: Any) -> str | None:
    """从 LLM 客户端安全提取已配置模型名（仅真实模式；Mock 模式返回 None）。

    v0.7.2 发布收尾：不打印 / 不暴露 api_key；不参与 real/mock 判定。
    """
    if isinstance(client, HTTPLLMClient):
        cfg = getattr(client, "_config", None)
        if cfg is not None:
            model = getattr(cfg, "model", None)
            if isinstance(model, str) and model.strip():
                return model.strip()
    return None


class AgentError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def run_analyst(
    question: str,
    dataset_id: str,
    experiment_id: str | None = None,
    source_version_id: str = "original",
    max_summary_rows: int | None = None,
    regenerate_shap: bool = False,
) -> dict[str, Any]:
    if not (question or "").strip():
        raise AgentError("agent_invalid_request", "question 不能为空。", 400)
    if not (dataset_id or "").strip():
        raise AgentError("agent_invalid_request", "dataset_id 不能为空。", 400)

    registry = ToolRegistry()
    context = ToolContext(
        dataset_id=dataset_id,
        experiment_id=experiment_id,
        source_version_id=source_version_id or "original",
        max_summary_rows=max_summary_rows,
        regenerate_shap=regenerate_shap,
    )
    plan = registry.select(question=question, has_experiment=bool(experiment_id))
    logger.info(
        "agent plan: selected=%s skipped=%s question=%r",
        [t.name for t in plan.selected],
        plan.to_payload()["skipped"],
        question,
    )

    tool_results = registry.run(plan, context)
    tool_payloads = [r.to_payload() for r in tool_results]

    # ReportTool 汇总（结构化）
    report_tool = registry.get("report")
    composed = report_tool.compose(tool_results, question) if isinstance(report_tool, ReportTool) else {}

    # 调 LLM
    client = get_default_client()
    system, user_prompt = build_prompt_payload(question, tool_payloads)

    llm_used_mock = bool(getattr(client, "is_mock", False))
    raw_response: str = ""
    llm_error: str | None = None
    try:
        raw_response = client.chat(system=system, user=user_prompt)
    except LLMError as exc:
        # 真实 LLM 失败：降级到 Mock 重发（不抛错）
        if isinstance(client, MockLLMClient):
            llm_error = f"[{exc.code}] {exc.message}"
            logger.warning("mock LLM error (unexpected): %s", exc.message)
        else:
            logger.warning("real LLM failed, fallback to mock: %s", exc.message)
            mock = MockLLMClient()
            try:
                raw_response = mock.chat(system=system, user=user_prompt)
                llm_used_mock = True
                llm_error = f"真实 LLM 失败，已降级 Mock：{exc.message}"
            except Exception as inner:  # noqa: BLE001
                llm_error = f"[{exc.code}] {exc.message}；Mock 兜底也失败：{inner}"
                raw_response = ""

    parsed = _parse_llm_output(raw_response)
    if not parsed:
        # 解析失败：使用 report 兜底
        parsed = {
            "answer": (composed.get("insights") or ["暂无数据"]) and "\n".join(
                f"- {line}" for line in (composed.get("insights") or ["暂无洞察"])
            ),
            "insights": list(composed.get("insights") or []),
            "recommendations": list(composed.get("recommendations") or []),
        }

    created_at = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    return {
        "success": True,
        "question": question,
        "dataset_id": dataset_id,
        "experiment_id": experiment_id,
        "source_version_id": context.source_version_id,
        "answer": parsed.get("answer", ""),
        "insights": list(parsed.get("insights") or []),
        "recommendations": list(parsed.get("recommendations") or []),
        "tools_used": [
            {"name": r["name"], "status": r["status"], "summary": r.get("summary", "")}
            for r in tool_payloads
        ],
        "tool_trace": tool_payloads,
        "plan": plan.to_payload(),
        "llm": {
            "is_mock": llm_used_mock,
            "model": _safe_model_name(client),
            "error": llm_error,
        },
        "raw_compose": composed,
        "created_at": created_at,
        "app_version": "0.7.2",
    }


# ============================================================
# LLM 输出解析
# ============================================================


def _parse_llm_output(raw: str) -> dict[str, Any] | None:
    if not raw or not raw.strip():
        return None
    # 1. 尝试直接 JSON
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # 2. 提取 ```json ... ``` 围栏
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    # 3. 提取首个 {...} 块
    first = re.search(r"\{.*\}", raw, re.DOTALL)
    if first:
        try:
            return json.loads(first.group(0))
        except json.JSONDecodeError:
            pass
    return None
