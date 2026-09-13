"""v0.7 Report Tool：在所有其他工具执行完毕后统一整理最终报告。

Report Tool 不会调用任何 LLM（避免循环）；它仅做结构化输出整理。
"""

from __future__ import annotations

from typing import Any

from app.agent.tools.base import BaseTool, ToolContext, ToolResult


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

        if ml:
            tm = ml.get("test_metrics") or {}
            try:
                acc = float(tm.get("accuracy"))
                if acc < 0.75:
                    insights.append(f"测试集准确率仅 {acc:.2%}，明显低于业务可接受水平。")
                    recommendations.append("考虑更复杂模型（XGBoost / LightGBM）或更细粒度特征工程。")
            except (TypeError, ValueError):
                pass
        if shap:
            top = shap.get("top_features") or []
            if top:
                insights.append(f"SHAP 显示 {top[0]} 是最关键的特征。")
            warnings = shap.get("warnings") or []
            for w in warnings[:2]:
                insights.append(f"SHAP 警告：{w}")
        if eda:
            eda_warnings = eda.get("warnings") or []
            if eda_warnings:
                insights.append("数据存在 " + str(len(eda_warnings)) + " 项质量警告。")
        if not recommendations:
            recommendations.append("结合 SHAP Top 特征与业务经验补充交叉特征。")
            recommendations.append("在引入更复杂模型前，先验证数据质量与样本量是否充足。")

        return {
            "dataset": datasets[0] if datasets else None,
            "eda": eda,
            "ml": ml,
            "shap": shap,
            "insights": insights,
            "recommendations": recommendations,
        }
