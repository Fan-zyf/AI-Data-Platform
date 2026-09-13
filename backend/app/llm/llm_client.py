"""v0.7 LLM 客户端。

- LLMClient：调用 OpenAI-compatible ``/chat/completions``（httpx 直接发请求，
  不依赖 ``openai`` SDK 以减少依赖体积）。
- MockLLMClient：未配置环境变量时使用，从工具结果拼装固定格式的中文分析报告。
- 业务层只依赖 ``LLMClient.chat(messages) -> str`` 接口；切换 mock / 真实 LLM 由
  ``get_default_client()`` 决定，**真实 LLM 出错时自动降级到 mock**，绝不阻塞分析。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Protocol

import httpx

from app.llm.llm_config import LLMConfig, load_llm_config

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """LLM 调用业务异常。"""

    def __init__(self, code: str, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class LLMClient(Protocol):
    """LLM 客户端协议（mock / 真实共用）。"""

    is_mock: bool

    def chat(self, system: str, user: str) -> str:  # pragma: no cover - 接口约定
        ...


# ============================================================
# 真实 LLM 客户端（OpenAI-compatible /chat/completions）
# ============================================================


class HTTPLLMClient:
    """通过 httpx 调用任意 OpenAI 兼容的 /chat/completions 端点。"""

    def __init__(self, config: LLMConfig) -> None:
        if config.is_mock:
            raise LLMError(
                "llm_not_configured",
                "LLM 客户端未启用：环境变量 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL 缺失。",
                503,
            )
        self._config = config
        self._endpoint = f"{config.base_url}/chat/completions"
        self.is_mock = False

    def chat(self, system: str, user: str) -> str:
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
        }
        try:
            with httpx.Client(timeout=self._config.timeout_seconds) as client:
                resp = client.post(self._endpoint, headers=headers, json=payload)
        except httpx.HTTPError as exc:
            raise LLMError("llm_network_error", f"LLM 服务连接失败：{exc}", 502) from exc

        if resp.status_code >= 400:
            snippet = resp.text[:300] if resp.text else ""
            raise LLMError(
                "llm_api_error",
                f"LLM 服务返回 HTTP {resp.status_code}：{snippet}",
                502,
            )
        try:
            data = resp.json()
        except json.JSONDecodeError as exc:
            raise LLMError("llm_invalid_response", "LLM 响应不是合法 JSON。", 502) from exc

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("llm_invalid_response", "LLM 响应缺少 choices[0].message.content。", 502) from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMError("llm_empty_response", "LLM 响应内容为空。", 502)
        return content.strip()


# ============================================================
# Mock 客户端：从工具结果拼装固定格式中文分析报告
# ============================================================


class MockLLMClient:
    """无 LLM 时的兜底客户端。

    特点：
    - 完全 deterministic，便于测试；
    - 不依赖任何外部服务，启动即可用；
    - 输出结构与真实 LLM 一致：``answer`` 段落 + ``insights`` 列表 + ``recommendations`` 列表。
    """

    is_mock = True

    def __init__(self) -> None:
        pass

    def chat(self, system: str, user: str) -> str:  # noqa: D401
        """根据 system 提示词与 user 上下文拼装一份自然语言回答。"""
        tool_results = _parse_user_payload(user) or {}
        question = _extract_question(system, user)
        return _compose_report(question=question, tool_results=tool_results)


# ============================================================
# 工具：解析 + 拼装报告（mock 专用）
# ============================================================


def _extract_question(system: str, user: str) -> str:
    m = re.search(r"用户问题[：:]\s*(.+?)(?:\n|$)", user)
    if m:
        return m.group(1).strip()
    return user.strip().splitlines()[0] if user.strip() else ""


def _parse_user_payload(user: str) -> dict[str, Any] | None:
    """从 user prompt 的 ``<<<TOOL_RESULTS>>>`` 块提取 JSON 字典。

    兼容两种格式：
    1. ``<<<TOOL_RESULTS>>>\\n<JSON>\\n<<<END_TOOL_RESULTS>>>``（prompts.py 当前输出）
    2. ``<<<TOOL_RESULTS>>>\\n```json\\n<JSON>\\n```\\n<<<END_TOOL_RESULTS>>>``（老格式兼容）
    """
    marker = "<<<TOOL_RESULTS>>>"
    if marker not in user:
        return None
    after = user.split(marker, 1)[1]
    end_marker = "<<<END_TOOL_RESULTS>>>"
    if end_marker in after:
        after = after.split(end_marker, 1)[0]
    after = after.strip()
    if not after:
        return None
    # 可选兼容 ```json ... ``` 围栏
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", after)
    candidate = fenced.group(1) if fenced else after
    # 截到第一个完整的 {...} 顶层块（避免尾部残留注释/空行）
    first = re.search(r"\{[\s\S]*\}", candidate)
    if first:
        candidate = first.group(0)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _compose_report(question: str, tool_results: dict[str, Any]) -> str:
    sections: list[str] = []
    answer_lines: list[str] = [f"针对您的问题「{question or '（无）'}」，根据当前已训练模型与数据状态给出以下分析："]
    insights: list[str] = []
    recommendations: list[str] = []

    # ---- 数据概况 ----
    dataset = tool_results.get("dataset") or {}
    if dataset:
        rows = dataset.get("rows") or dataset.get("n_rows")
        cols = dataset.get("n_columns") or dataset.get("columns")
        n_columns = cols if isinstance(cols, int) else (len(cols) if isinstance(cols, list) else None)
        missing_rate = dataset.get("missing_rate")
        quality = dataset.get("quality")
        bits = []
        if rows is not None:
            bits.append(f"{rows} 行")
        if n_columns is not None:
            bits.append(f"{n_columns} 列")
        head = "数据集概况：" + (" / ".join(bits) if bits else "（无字段统计）")
        if missing_rate is not None:
            head += f"，平均缺失率 {missing_rate}"
            try:
                mr = float(missing_rate)
                if mr > 0.05:
                    insights.append(f"数据集平均缺失率 {mr:.2%}，建议优先处理高缺失列。")
            except (TypeError, ValueError):
                pass
        if quality:
            warnings = quality.get("warnings") if isinstance(quality, dict) else None
            if warnings:
                insights.append(f"质量警告 {len(warnings)} 条，例如：" + "、".join(warnings[:3]))
        sections.append(head)

    # ---- EDA ----
    eda = tool_results.get("eda") or {}
    if eda:
        eda_warnings = eda.get("warnings") or []
        if eda_warnings:
            insights.append("EDA 阶段发现 " + "、".join(map(str, eda_warnings[:3])))
        if eda.get("missing_top"):
            top_missing = eda["missing_top"][:3]
            if top_missing:
                names = "、".join(f"{row['column']}({row['missing_rate']:.1%})" for row in top_missing if isinstance(row, dict))
                if names:
                    insights.append(f"缺失最严重的列：{names}")

    # ---- ML ----
    ml = tool_results.get("ml") or {}
    if ml:
        task = ml.get("task_type")
        model = ml.get("best_model") or ml.get("model")
        metrics = ml.get("metrics") or {}
        if task or model:
            label = f"{('回归' if task == 'regression' else '分类' if task else '机器学习')}模型：{model or '未知'}"
            sections.append(label)
        if metrics:
            bullets = []
            for k, v in list(metrics.items())[:6]:
                try:
                    bullets.append(f"{k} = {float(v):.4f}")
                except (TypeError, ValueError):
                    bullets.append(f"{k} = {v}")
            if bullets:
                sections.append("核心指标：" + "，".join(bullets))
        test_metrics = ml.get("test_metrics") or {}
        if isinstance(test_metrics, dict) and test_metrics:
            low_metrics = [k for k, v in test_metrics.items() if isinstance(v, (int, float)) and v < 0.7]
            if low_metrics:
                insights.append(f"测试集指标 {', '.join(low_metrics)} 偏低，可考虑更复杂模型或特征工程。")
            accuracy = test_metrics.get("accuracy")
            if accuracy is not None and accuracy < 0.8:
                recommendations.append("尝试加入交叉特征或对类别特征做目标编码。")

    # ---- SHAP ----
    shap = tool_results.get("shap") or {}
    if shap:
        top = shap.get("top_features") or []
        if top:
            sections.append("SHAP Top 特征：" + "、".join(map(str, top[:5])))
            if len(top) >= 3:
                insights.append(f"全局最重要的特征是 {top[0]}，建议优先关注其分布与缺失。")
        sample = shap.get("sample_explanation") or {}
        if sample:
            contributions = sample.get("contributions") or []
            if contributions:
                max_c = max(contributions, key=lambda c: abs(c.get("shap") or 0))
                if max_c:
                    insights.append(
                        f"样本 #{sample.get('index', 0) + 1} 的预测主要由 {max_c.get('feature')} (SHAP={max_c.get('shap')}) 推动。"
                    )

    if not sections:
        sections.append("未读取到具体数据 / 模型 / SHAP 上下文，请确认 dataset_id 与 experiment_id 有效。")

    # ---- 通用建议 ----
    if not recommendations:
        recommendations.append("结合 SHAP Top 特征与业务规则，进一步构造交叉特征。")
    recommendations.append("在增加数据量的同时评估更复杂的模型（如 XGBoost / LightGBM）。")

    # 拼装成 JSON 字符串（业务层再解析）
    answer = "\n".join(answer_lines + [""] + [f"### {s}" for s in sections])
    return json.dumps(
        {"answer": answer, "insights": insights, "recommendations": recommendations},
        ensure_ascii=False,
    )


# ============================================================
# 工厂
# ============================================================


_client_singleton: LLMClient | None = None


def get_default_client() -> LLMClient:
    """返回默认 LLM 客户端：未配置时为 Mock；配置时为真实 HTTP 客户端。"""
    global _client_singleton
    if _client_singleton is not None:
        return _client_singleton
    config = load_llm_config()
    if config.is_mock:
        logger.info("LLM env not configured → using MockLLMClient")
        _client_singleton = MockLLMClient()
    else:
        logger.info("LLM configured → using HTTPLLMClient (%s)", config.model)
        _client_singleton = HTTPLLMClient(config)
    return _client_singleton


def reset_default_client_for_test() -> None:
    """测试用：清除单例缓存以应用新环境变量。"""
    global _client_singleton
    _client_singleton = None
