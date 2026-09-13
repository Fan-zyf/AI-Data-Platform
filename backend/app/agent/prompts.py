"""v0.7 Prompt 模板。

设计原则：
- 真实 LLM 与 Mock 客户端共用同一组 system / user 模板，便于测试；
- ``user`` 模板包含：
    1. 用户原始问题；
    2. 结构化工具结果（人类可读 + 末尾 ``<<<TOOL_RESULTS>>>`` 包裹的 JSON 块，
       供 MockLLMClient 反向解析拿到结构化数据）；
- 提示中明确要求 LLM 输出 ``JSON: {answer, insights, recommendations}``，
  业务层负责解析失败时回退到「仅 answer」模式。
"""

from __future__ import annotations

import json
from typing import Any

AGENT_SYSTEM_PROMPT = (
    "你是一位资深数据科学助手（AI Data Analyst），专门分析机器学习实验。\n"
    "你会收到 1) 用户原始问题，2) 已经被调用的工具结果（结构化 JSON）。\n"
    "请基于工具结果回答用户问题，给出可直接阅读的中文回答，\n"
    "并按以下 JSON Schema 返回（**严格 JSON，不要包含 JSON 之外的字符**）：\n"
    "{\n"
    '  "answer": "Markdown 格式的完整回答（包含若干 ### 小节）",\n'
    '  "insights": ["关键洞察1", "关键洞察2"],\n'
    '  "recommendations": ["可执行建议1", "可执行建议2"]\n'
    "}\n"
    "禁止编造未在工具结果中出现的数字、特征名或模型名；\n"
    "如果数据不足，请明确指出『缺少某类信息』并给出补救建议。"
)


def _compact_tool_data(data: Any) -> Any:
    """对 tool 内部的 data 做浅裁剪，防止 prompt 过大。"""
    if not isinstance(data, dict):
        return data
    out: dict[str, Any] = {}
    for key, value in data.items():
        if key == "sample_explanation" and isinstance(value, dict):
            # 截断单点 contributions，仅保留 top-5
            sample = dict(value)
            contribs = sample.get("contributions_top") or []
            if isinstance(contribs, list):
                sample["contributions_top"] = contribs[:5]
            out[key] = sample
            continue
        if key == "correlation" and isinstance(value, dict):
            # 相关性 top-5 已经够用
            out[key] = value
            continue
        if key in ("warnings", "missing_top", "top_pairs", "top_features"):
            if isinstance(value, list):
                out[key] = value[:5]
            else:
                out[key] = value
            continue
        out[key] = value
    return out


def _json_default(obj: Any) -> Any:
    """``json.dumps(..., default=_json_default)`` 兜底：datetime/Decimal/Path 等不可序列化对象转字符串。"""
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return str(obj)


def build_user_prompt(question: str, tool_results: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append(f"用户问题：{question.strip() or '（无）'}")
    lines.append("")
    lines.append("=== 已调用的工具结果（人类可读）===")
    for r in tool_results:
        name = r.get("name", "?")
        status = r.get("status", "ok")
        lines.append(f"- 工具 {name} (status={status})")
        if r.get("error"):
            lines.append(f"  error: {r['error']}")
        if r.get("summary"):
            lines.append(f"  summary: {r['summary']}")
        data = r.get("data")
        if isinstance(data, dict) and data:
            data_view = _compact_tool_data(data)
            try:
                serialized = json.dumps(data_view, ensure_ascii=False, indent=2)
            except (TypeError, ValueError):
                serialized = str(data_view)
            if len(serialized) > 6000:
                serialized = serialized[:6000] + "..."
            lines.append("  data:")
            for ln in serialized.splitlines():
                lines.append(f"    {ln}")
    lines.append("")
    lines.append("=== 结构化工具结果（机器可读，供分析用）===")
    lines.append("<<<TOOL_RESULTS>>>")
    structured: dict[str, Any] = {}
    for r in tool_results:
        if r.get("status") == "error":
            continue
        name = r.get("name", "?")
        data = r.get("data") or {}
        if isinstance(data, dict):
            structured[name] = _compact_tool_data(data)
        else:
            structured[name] = data
    try:
        structured_text = json.dumps(
            structured,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        )
    except (TypeError, ValueError):
        structured_text = str(structured)
    if len(structured_text) > 8000:
        structured_text = structured_text[:8000] + "..."
    lines.append(structured_text)
    lines.append("<<<END_TOOL_RESULTS>>>")
    lines.append("")
    lines.append("请按 JSON Schema 严格输出。")
    return "\n".join(lines)


def build_prompt_payload(question: str, tool_results: list[dict[str, Any]]) -> tuple[str, str]:
    """返回 (system, user_with_inline_results)。"""
    return AGENT_SYSTEM_PROMPT, build_user_prompt(question, tool_results)
