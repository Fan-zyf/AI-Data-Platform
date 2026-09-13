"""v0.7 Tool Registry。

职责：
- 实例化全部 5 个 Tool；
- 根据用户问题关键词 + 默认权重，自动选择要调用的工具子集；
- 保证 ``experiment_id`` 缺失时自动跳过 ML / SHAP 工具。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.agent.tools import ALL_TOOLS, BaseTool, ReportTool
from app.agent.tools.base import ToolContext
from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ToolPlan:
    selected: list[BaseTool]
    skipped: list[tuple[BaseTool, str]]

    def to_payload(self) -> dict:
        return {
            "selected": [t.name for t in self.selected],
            "skipped": [{"name": t.name, "reason": reason} for t, reason in self.skipped],
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {cls().name: cls() for cls in ALL_TOOLS}

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def all_names(self) -> list[str]:
        return list(self._tools.keys())

    def select(self, question: str, has_experiment: bool) -> ToolPlan:
        """根据问题关键词 + 是否提供 experiment_id 选择要执行的工具。

        规则：
        1. ReportTool 总是被加入（但其 ``execute`` 仅占位，真正的报告由 Agent 汇总）；
        2. 实验类工具（ML / SHAP）在缺少 experiment_id 时跳过；
        3. 关键词命中数 ≥ 1 即视为「相关」，否则只保留 DatasetTool + ReportTool。
        4. 工具数量上限 :attr:`settings.agent_max_tools`。
        """
        lowered = (question or "").strip().lower()
        # 计算关键词命中
        scored: list[tuple[int, BaseTool]] = []
        for tool in self._tools.values():
            if tool.name == "report":
                continue
            score = sum(1 for kw in tool.keywords if kw.lower() in lowered)
            scored.append((score, tool))
        # 按命中数降序
        scored.sort(key=lambda x: (-x[0], x[1].name))

        selected: list[BaseTool] = []
        skipped: list[tuple[BaseTool, str]] = []
        for score, tool in scored:
            if tool.requires_experiment and not has_experiment:
                skipped.append((tool, "未提供 experiment_id"))
                continue
            if score > 0:
                selected.append(tool)
        # 默认兜底：如果什么都没命中，加入 dataset + eda
        if not selected:
            for tool in self._tools.values():
                if tool.name in ("dataset", "eda") and not (
                    tool.requires_experiment and not has_experiment
                ):
                    selected.append(tool)
        # 限制最大工具数（report 不计入）
        cap = max(1, int(settings.agent_max_tools))
        if len(selected) > cap:
            skipped.extend((t, "超过单次最大工具数限制") for t in selected[cap:])
            selected = selected[:cap]
        # 总是带上 Report（占位 + 提示 LLM 报告工具可用）
        selected.append(self._tools["report"])
        return ToolPlan(selected=selected, skipped=skipped)

    def run(self, plan: ToolPlan, context: ToolContext) -> list:
        """按顺序执行 plan 中选中的工具（同步串行）。"""
        results = []
        for tool in plan.selected:
            logger.info("agent tool %s executing", tool.name)
            results.append(tool.safe_execute(context))
        return results
