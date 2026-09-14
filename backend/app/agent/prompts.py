"""v0.7.2 Prompt 模板。

v0.7.2 新增：Grounding Rules（证据约束）。
- 真实 LLM 与 Mock 客户端共用同一组 system / user 模板；
- ``user`` 模板包含：
    1. 用户原始问题；
    2. 结构化工具结果（人类可读 + 末尾 ``<<<TOOL_RESULTS>>>`` 包裹的 JSON 块，
       供 MockLLMClient 反向解析拿到结构化数据）；
- 提示中明确要求 LLM 输出 ``JSON: {answer, insights, recommendations}``；
- 业务层负责解析失败时回退到「仅 answer」模式。

Grounding Rules（v0.7.2 强制约束）：
- 三类内容必须清晰区分：
    A. Observed Facts    —— 来自 tool 结果的客观事实（必须可回溯到工具输出）
    B. Interpretation    —— 基于事实的谨慎解释（用「可能」「倾向于」等措辞）
    C. Recommendations  —— 未经实验验证的改进建议（明确标注「待验证」）
- 禁止：
    1. 编造未在 tool output 中出现的指标 / 特征名 / 模型名；
    2. 预测「采取某措施后 F1/AUC/std 将提升/下降到 X」；
    3. 声称「某算法一定优于当前模型」；
    4. 把相关性描述为因果；
    5. 把未在 tool output 中出现的 preprocessing / missing indicator
       描述为「模型已采用的事实」；
    6. 在小样本下做确定的性能判断。
- 措辞：
    未经实验验证的内容必须使用「建议尝试」「可能」「需要进一步验证」
    「当前数据不足以确定」「待验证」等措辞。
- 报告结构统一为 A / B / C / D 四节。
"""

from __future__ import annotations

import json
from typing import Any


# ============================================================
# System Prompt
# ============================================================

GROUNDING_RULES = (
    "【Grounding Rules v0.7.2 — 证据约束（必须严格遵守）】\n"
    "1) 三类内容严格区分，禁止混淆：\n"
    "   A. 已观测事实（Observed Facts）—— 只能引用工具结果里实际出现的数字、字段、模型名、特征名。\n"
    "      必须可回溯到「已调用的工具结果」中相应字段。\n"
    "   B. 谨慎解释（Interpretation）—— 对事实的合理解读，必须使用「可能」「倾向于」\n"
    "      「结合现有数据看」等措辞，不要写成事实。\n"
    "   C. 建议（Recommendations）—— 尚未实验验证的改进方向。\n"
    "      必须以「建议尝试」「可考虑」「需要进一步验证」开头，\n"
    "      不得附带未经验证的具体性能提升数字。\n"
    "2) 严禁：\n"
    "   - 编造工具结果中没有的指标、特征名、模型名；\n"
    "   - 预测「采取 X 措施后，F1 将提升到 Y」「CV std 将下降到 Z」等具体数值；\n"
    "   - 声称「XGBoost / LightGBM / SMOTE / class_weight 一定能改善当前模型」；\n"
    "   - 把工具结果中未出现的 preprocessing 行为（如 missing indicator、归一化方式）\n"
    "     描述为「模型已使用的事实」；\n"
    "   - 把相关性写成因果。\n"
    "3) 小样本判断：\n"
    "   如果工具结果中 train_rows < 50 或 test_rows < 20，\n"
    "   必须在 answer 与 insights 中明确说明「测试集过小，指标不稳定，不能据此作确定性能判断」。\n"
    "4) SHAP 解释：\n"
    "   只能引用工具结果中真实出现的 feature_names / importance / shap_values / base_value。\n"
    "   不得自行推断「模型为缺失值创建了独立维度」之类的内容，除非 tool output 中确实出现。\n"
    "5) 报告结构（必须严格按此四节顺序，answer 字段内使用 Markdown 小节）：\n"
    "   A. 已观测事实（来自工具结果的可回溯数据）\n"
    "   B. 谨慎解释（基于事实的合理解读，使用「可能」等措辞）\n"
    "   C. 当前局限（数据规模 / 类别分布 / 指标稳定性等不足）\n"
    "   D. 建议验证的下一步（候选方案，必须标注「待验证」「建议尝试」）\n"
)

AGENT_SYSTEM_PROMPT = (
    "你是一位资深数据科学助手（AI Data Analyst），专门分析机器学习实验。\n"
    "你会收到 1) 用户原始问题，2) 已经被调用的工具结果（结构化 JSON）。\n"
    "请基于工具结果回答用户问题，给出可直接阅读的中文回答。\n"
    "\n"
    + GROUNDING_RULES
    + "\n"
    "请按以下 JSON Schema 严格输出（**严格 JSON，不要包含 JSON 之外的字符**）：\n"
    "{\n"
    '  "answer": "Markdown 格式的完整回答，必须按 A/B/C/D 四节组织（### A. 已观测事实 / '
    '### B. 谨慎解释 / ### C. 当前局限 / ### D. 建议验证的下一步）",\n'
    '  "insights": ["关键洞察1（仅事实或谨慎解释）", "关键洞察2"],\n'
    '  "recommendations": ["【待验证】可执行建议1（不附具体提升数字）", "【待验证】可执行建议2"]\n'
    "}\n"
    "如果数据不足，请在 answer 的 C 节明确说明「缺少哪类信息」，并保持 recommendations 留空或仅给出验证方向。"
)


# ============================================================
# 小样本判定工具
# ============================================================


def is_small_sample(tool_results: list[dict[str, Any]]) -> bool:
    """判定当前数据是否属于小样本。

    依据：
    - train_rows < 50
    - 或 test_rows < 20
    - 或 dataset.rows < 100
    """
    for r in tool_results or []:
        data = r.get("data") or {}
        if not isinstance(data, dict):
            continue
        # 训练 / 测试样本
        for key in ("train_rows", "test_rows"):
            v = data.get(key)
            if isinstance(v, (int, float)) and v < (50 if key == "train_rows" else 20):
                return True
        # dataset 总行数
        for k in ("rows", "n_rows", "total_rows"):
            v = data.get(k)
            if isinstance(v, (int, float)) and v < 100:
                return True
        # ML 工具返回的 metrics 维度
        cv = data.get("cv") or {}
        if isinstance(cv, dict):
            for k in ("folds",):
                v = cv.get(k)
                if isinstance(v, (int, float)) and v <= 3:
                    return True
    return False


# 统一小样本警告措辞（被 prompts.py / report_tool.py / llm_client.py 共享）
SMALL_SAMPLE_NOTE = (
    "测试集过小，指标不稳定，"
    "不能据此对模型性能作确定判断；需要扩大样本后再评估。"
)


# ============================================================
# Compact / Serialize
# ============================================================


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


# ============================================================
# User Prompt
# ============================================================


def build_user_prompt(question: str, tool_results: list[dict[str, Any]]) -> str:
    small_sample_note = ""
    if is_small_sample(tool_results):
        small_sample_note = (
            "\n【小样本提示】当前工具结果显示数据规模较小（训练/测试样本或总行数偏低）。\n"
            "请在 answer 的 C 节「当前局限」中明确指出「测试集过小，指标不稳定，"
            "不能据此作确定性能判断」，并在 D 节给出「扩大样本后再评估」的待验证方向。\n"
        )

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
    lines.append(small_sample_note)
    lines.append("请按 JSON Schema 严格输出（A/B/C/D 四节、禁止编造、措辞软化）。")
    return "\n".join(lines)


def build_prompt_payload(question: str, tool_results: list[dict[str, Any]]) -> tuple[str, str]:
    """返回 (system, user_with_inline_results)。"""
    return AGENT_SYSTEM_PROMPT, build_user_prompt(question, tool_results)
