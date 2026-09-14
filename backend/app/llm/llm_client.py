"""v0.7.2 LLM 客户端。

- LLMClient：调用 OpenAI-compatible ``/chat/completions``（httpx 直接发请求，
  不依赖 ``openai`` SDK 以减少依赖体积）。
- MockLLMClient：未配置环境变量时使用，从工具结果拼装固定格式的中文分析报告。
- 业务层只依赖 ``LLMClient.chat(messages) -> str`` 接口；切换 mock / 真实 LLM 由
  ``get_default_client()`` 决定，**真实 LLM 出错时自动降级到 mock**，绝不阻塞分析。

v0.7.2：Mock 报告按 A/B/C/D 四节组织，全部使用「可能 / 建议尝试 / 待验证」措辞，
严守 Grounding Rules（不再产出未经验证的具体性能提升数字）。
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


def _tag_unverified(text: str) -> str:
    """给单条建议统一加【待验证】前缀，确保不被误读为已验证结论。"""
    text = (text or "").strip()
    if not text or text.startswith("【待验证】"):
        return text
    return f"【待验证】{text}"


def _is_small_sample(tool_results: dict[str, Any]) -> bool:
    """轻量小样本判定：训练/测试样本或总行数偏低。"""
    ds = tool_results.get("dataset") or {}
    ml = tool_results.get("ml") or {}
    eda = tool_results.get("eda") or {}

    train_rows = ml.get("train_rows")
    test_rows = ml.get("test_rows")
    if isinstance(train_rows, (int, float)) and train_rows < 50:
        return True
    if isinstance(test_rows, (int, float)) and test_rows < 20:
        return True
    for v in (ds.get("rows"), ds.get("n_rows"), eda.get("rows"), eda.get("total_rows")):
        if isinstance(v, (int, float)) and v < 100:
            return True
    cv = ml.get("cv") or {}
    if isinstance(cv, dict):
        folds = cv.get("folds")
        if isinstance(folds, (int, float)) and folds <= 3:
            return True
    return False


def _compose_report(question: str, tool_results: dict[str, Any]) -> str:
    """v0.7.2 Mock 报告：按 A/B/C/D 四节组织，全部使用「可能 / 建议尝试 / 待验证」措辞。"""
    facts: list[str] = []
    insights: list[str] = []
    limitations: list[str] = []
    recommendations: list[str] = []

    dataset = tool_results.get("dataset") or {}
    eda = tool_results.get("eda") or {}
    ml = tool_results.get("ml") or {}
    shap = tool_results.get("shap") or {}

    # ---- A. 事实 ----
    rows = (
        dataset.get("rows")
        or dataset.get("n_rows")
        or eda.get("rows")
        or eda.get("total_rows")
    )
    cols = dataset.get("n_columns") or dataset.get("columns")
    n_columns = cols if isinstance(cols, int) else (len(cols) if isinstance(cols, list) else None)
    if rows is not None or n_columns is not None:
        bits = []
        if rows is not None:
            bits.append(f"{rows} 行")
        if n_columns is not None:
            bits.append(f"{n_columns} 列")
        facts.append("数据集规模：" + " / ".join(bits))

    if ml:
        for k, label in (("train_rows", "训练"), ("test_rows", "测试")):
            v = ml.get(k)
            if isinstance(v, (int, float)):
                facts.append(f"{label}样本数 = {v}")

    if eda:
        missing_rate = eda.get("missing_rate")
        if missing_rate is not None:
            facts.append(f"平均缺失率 = {missing_rate}")
        missing_top = eda.get("missing_top") or []
        if missing_top:
            names = "、".join(
                f"{m.get('column')}({m.get('missing_rate')})" for m in missing_top[:3]
                if isinstance(m, dict)
            )
            if names:
                facts.append(f"缺失最严重的列：{names}")
        quality = eda.get("quality")
        if isinstance(quality, dict):
            warnings = quality.get("warnings") or []
            if warnings:
                facts.append(f"质量警告 {len(warnings)} 条，例如：" + "、".join(map(str, warnings[:3])))
        elif eda.get("warnings"):
            facts.append(f"EDA 报告 {len(eda['warnings'])} 项警告")

    if ml:
        task = ml.get("task_type")
        model = ml.get("best_model") or ml.get("model")
        if task or model:
            label = f"{('回归' if task == 'regression' else '分类' if task else '机器学习')}模型：{model or '未知'}"
            facts.append(label)
        metrics = ml.get("metrics") or {}
        if metrics:
            bullets = []
            for k, v in list(metrics.items())[:6]:
                try:
                    bullets.append(f"{k} = {float(v):.4f}")
                except (TypeError, ValueError):
                    bullets.append(f"{k} = {v}")
            if bullets:
                facts.append("核心指标：" + "，".join(bullets))
        test_metrics = ml.get("test_metrics") or {}
        if isinstance(test_metrics, dict) and test_metrics:
            tm_bullets = []
            for k, v in list(test_metrics.items())[:6]:
                try:
                    tm_bullets.append(f"{k} = {float(v):.4f}")
                except (TypeError, ValueError):
                    tm_bullets.append(f"{k} = {v}")
            if tm_bullets:
                facts.append("测试集指标：" + "，".join(tm_bullets))

    if shap:
        top = shap.get("top_features") or []
        if top:
            facts.append("SHAP Top 特征：" + "、".join(map(str, top[:5])))
        # 严禁自行推断 missing indicator —— 仅在 feature_names 中确实出现才引用
        feat_names = shap.get("feature_names") or []
        if any(
            isinstance(f, str) and ("missing" in f.lower() or "indicator" in f.lower())
            for f in feat_names
        ):
            facts.append("SHAP feature_names 中包含缺失指示类特征。")

    # ---- C. 局限 / 小样本 ----
    if _is_small_sample(tool_results):
        # 与 prompts.SMALL_SAMPLE_NOTE / report_tool 共享措辞
        limitations.append(
            "测试集过小，指标不稳定，"
            "不能据此对模型性能作确定判断；需要扩大样本后再评估。"
        )

    # ---- B. 谨慎解释 ----
    if ml:
        test_metrics = ml.get("test_metrics") or {}
        low_metrics = [
            k for k, v in test_metrics.items()
            if isinstance(v, (int, float)) and v < 0.7
        ]
        if low_metrics:
            insights.append(
                f"测试集中 {', '.join(low_metrics)} 等指标偏低；"
                f"在当前样本规模下可能是数据不足或类别分布不均的影响，需要进一步验证。"
            )
    if eda:
        eda_warnings = eda.get("warnings") or []
        if eda_warnings:
            insights.append(
                "EDA 阶段发现 " + "、".join(map(str, eda_warnings[:3]))
                + "；这些可能影响模型稳定性，但与最终效果的关系需要进一步验证。"
            )
        if eda.get("missing_top"):
            top_missing = eda["missing_top"][:3]
            names = "、".join(
                f"{row['column']}({row.get('missing_rate')})" for row in top_missing
                if isinstance(row, dict)
            )
            if names:
                insights.append(f"缺失最严重的列：{names}；建议优先核查其分布。")
    if shap:
        top = shap.get("top_features") or []
        if top:
            insights.append(
                f"全局最重要的特征是 {top[0]}；是否构成因果影响需要进一步业务验证。"
            )
        sample = shap.get("sample_explanation") or {}
        if sample:
            contributions = sample.get("contributions") or []
            if contributions:
                max_c = max(contributions, key=lambda c: abs(c.get("shap") or 0))
                if max_c:
                    insights.append(
                        f"样本 #{sample.get('index', 0) + 1} 的预测主要由 {max_c.get('feature')} 推动；"
                        f"该推论仅对该样本成立。"
                    )

    # ---- D. 建议（候选方案，全部【待验证】，不附数字） ----
    recommendations.append(
        "建议尝试结合 SHAP Top 特征与业务经验构造交叉特征，但具体增益需要在新数据上重新训练后验证。"
    )
    if ml:
        test_metrics = ml.get("test_metrics") or {}
        low = any(
            isinstance(v, (int, float)) and v < 0.7
            for v in test_metrics.values()
        )
        if low:
            recommendations.append(
                "建议尝试在更大样本（含更平衡的类别分布）上重新训练，"
                "在引入 LightGBM / XGBoost 之前先评估 class_weight 或重采样的实际影响。"
            )
    if eda and eda.get("missing_top"):
        recommendations.append(
            "建议尝试对高缺失列使用中位数/众数填充 + 缺失指示列，"
            "但是否优于当前缺失处理方式，需要在实验中对比验证。"
        )
    if not recommendations:
        recommendations.append(
            "建议尝试扩大样本规模并重训 baseline，在确认数据/指标稳定后再评估更复杂模型。"
        )
    recommendations = [_tag_unverified(r) for r in recommendations]

    # ---- 组装 A/B/C/D ----
    answer_lines: list[str] = [f"针对您的问题「{question or '（无）'}」，基于已调用工具结果整理如下分析："]
    answer_lines.append("")
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

    return json.dumps(
        {
            "answer": "\n".join(answer_lines),
            "insights": insights,
            "recommendations": recommendations,
        },
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
