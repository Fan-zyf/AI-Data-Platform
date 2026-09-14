"""v0.7.2 Report Tool：在所有其他工具执行完毕后统一整理最终报告。

v0.7.2 调整：
- 严格遵守 Grounding Rules：recommendations 全部以「【待验证】」前缀开头，
  不附未经验证的具体性能提升数字；
- 命中「小样本」时，answer 中追加「指标不稳定」提示。
- Report Tool 不会调用任何 LLM（避免循环）；它仅做结构化输出整理。
"""

from __future__ import annotations

from typing import Any

from app.agent.prompts import is_small_sample, SMALL_SAMPLE_NOTE
from app.agent.tools.base import BaseTool, ToolContext, ToolResult


_SMALL_SAMPLE_NOTE = SMALL_SAMPLE_NOTE  # 共享 prompts.py 的措辞


def _tag_unverified(text: str) -> str:
    """给单条建议加『【待验证】』前缀，确保不被误读为已验证结论。"""
    text = (text or "").strip()
    if not text:
        return text
    if text.startswith("【待验证】"):
        return text
    return f"【待验证】{text}"


class ReportTool(BaseTool):
    name = "report"
    description = "汇总所有工具结果，输出结构化分析报告（数据概况 / 模型表现 / 关键洞察 / 改进建议）。"
    keywords = (
        "报告", "总结", "汇总", "report", "summary", "结论", "建议",
        "分析", "整体", "综合", "总览", "report",
    )

    def execute(self, context: ToolContext) -> ToolResult:  # noqa: D401
        # Report Tool 实际并不读取数据，汇总由 agent 层完成。
        return ToolResult(
            name=self.name,
            status="ok",
            data={"ready": True},
            summary="报告工具准备就绪（由 Agent 在编排阶段统一调用）。",
        )

    def compose(self, tool_results: list[ToolResult], question: str) -> dict[str, Any]:
        """由 Agent 调用，把多个 tool 的结果整合成最终结构化报告（不含 LLM）。"""
        datasets = []
        eda = ml = shap = None
        for r in tool_results:
            if r.status == "error":
                continue
            if r.name == "dataset":
                datasets.append(r.data)
            elif r.name == "eda":
                eda = r.data
            elif r.name == "ml":
                ml = r.data
            elif r.name == "shap":
                shap = r.data

        insights: list[str] = []
        recommendations: list[str] = []
        limitations: list[str] = []
        facts: list[str] = []

        # ---- 训练 / 测试规模判定（grounding 必备） ----
        train_rows = None
        test_rows = None
        if ml:
            train_rows = ml.get("train_rows")
            test_rows = ml.get("test_rows")
        small = is_small_sample(
            [
                {"name": "ml", "data": ml or {}, "status": "ok"},
                {"name": "eda", "data": eda or {}, "status": "ok"},
                {"name": "dataset", "data": datasets[0] if datasets else {}, "status": "ok"},
            ]
        )
        if small:
            limitations.append(_SMALL_SAMPLE_NOTE)

        # ---- ML：只描述事实，不预测提升 ----
        if ml:
            tm = ml.get("test_metrics") or {}
            try:
                acc = float(tm.get("accuracy"))
                facts.append(f"测试集 accuracy = {acc:.2%}（来自工具结果的 test_metrics.accuracy）")
                if acc < 0.75:
                    insights.append(
                        f"测试集准确率仅 {acc:.2%}，可能反映当前模型在小样本下表现不稳。"
                    )
                    # 注意：不再写「XGBoost 一定能提升到 X%」之类的话
                    recommendations.append(
                        "建议尝试在更大样本上重新评估 baseline 模型，并对比更复杂模型（如 LightGBM / XGBoost），"
                        "但在当前小样本下其实际提升幅度无法被验证。"
                    )
            except (TypeError, ValueError):
                pass

            # 类别分布（事实）
            try:
                pos = ml.get("class_labels")
                if pos:
                    facts.append(f"任务类别标签：{pos}（来自工具结果）")
            except Exception:
                pass

        # ---- SHAP：只引用真实字段 ----
        if shap:
            top = shap.get("top_features") or []
            if top:
                facts.append(
                    f"SHAP 全局重要性 Top1 特征：{top[0]}（来自 shap.top_features 字段）"
                )
                insights.append(
                    f"SHAP 显示 {top[0]} 是当前全局重要性最高的特征；"
                    f"是否构成因果影响需要进一步业务验证。"
                )
            warnings = shap.get("warnings") or []
            for w in warnings[:2]:
                facts.append(f"SHAP 工具告警：{w}")

            # 严禁自行推断 missing indicator —— 仅在 tool output 显式出现时引用
            # 判定标准：feature_names 中是否出现含 "missing" / "is_missing" / "indicator" 字样
            try:
                feat_names = shap.get("feature_names") or []
                if any(
                    isinstance(f, str) and ("missing" in f.lower() or "indicator" in f.lower())
                    for f in feat_names
                ):
                    facts.append("SHAP feature_names 中包含缺失指示类特征（来自工具结果）。")
            except Exception:
                pass

        # ---- EDA：仅事实 ----
        if eda:
            eda_warnings = eda.get("warnings") or []
            if eda_warnings:
                facts.append(
                    f"EDA 共报告 {len(eda_warnings)} 项数据质量警告（来自 eda.warnings 字段）。"
                )
            missing_top = eda.get("missing_top") or []
            if missing_top:
                top_str = "、".join(
                    f"{m.get('column')}({m.get('missing_rate')})" for m in missing_top[:3]
                    if isinstance(m, dict)
                )
                if top_str:
                    facts.append(f"缺失最严重的列：{top_str}（来自 eda.missing_top）")

        # ---- 默认建议（全部加待验证前缀 + 不附数字） ----
        if not recommendations:
            recommendations.append(
                "建议尝试结合 SHAP Top 特征与业务经验构造交叉特征，但具体增益需要在新数据上重新训练后验证。"
            )
            recommendations.append(
                "建议尝试在更大样本（含更平衡的类别分布）上重新训练，"
                "在引入 XGBoost / LightGBM 之前先确认 class_weight 或重采样对指标的实质影响。"
            )

        # 统一加【待验证】前缀
        recommendations = [_tag_unverified(r) for r in recommendations]

        # ---- 组装 answer（A/B/C/D 四节结构，Mock / 真实 LLM 兜底统一） ----
        answer_lines: list[str] = []
        answer_lines.append("### A. 已观测事实（来自工具结果）")
        if facts:
            for f in facts:
                answer_lines.append(f"- {f}")
        else:
            answer_lines.append("- 当前工具结果中未提供可回溯的具体数据。")

        answer_lines.append("")
        answer_lines.append("### B. 谨慎解释（基于事实的解读）")
        if insights:
            for it in insights:
                answer_lines.append(f"- {it}")
        else:
            answer_lines.append("- 当前样本与指标不足以做出额外解释。")

        answer_lines.append("")
        answer_lines.append("### C. 当前局限")
        if limitations:
            for lm in limitations:
                answer_lines.append(f"- {lm}")
        else:
            answer_lines.append("- 当前工具结果未显式报告样本规模相关的局限。")

        answer_lines.append("")
        answer_lines.append("### D. 建议验证的下一步（候选方案，未经验证）")
        for r in recommendations:
            answer_lines.append(f"- {r}")

        return {
            "dataset": datasets[0] if datasets else None,
            "eda": eda,
            "ml": ml,
            "shap": shap,
            "insights": insights,
            "recommendations": recommendations,
            "limitations": limitations,
            "facts": facts,
            "answer": "\n".join(answer_lines),
        }
