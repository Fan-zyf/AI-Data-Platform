"""v0.7 AI Data Analyst Agent API。

- ``POST /api/agent/analyze``：自然语言提问 → 自动调用工具 → LLM（或 Mock）汇总。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from app.agent import AgentError, analyze

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])


class AnalyzeRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    dataset_id: str = Field(..., min_length=1, max_length=128)
    experiment_id: str | None = Field(default=None, max_length=128)
    source_version_id: str = Field(default="original", max_length=128)
    max_summary_rows: int | None = Field(default=None, ge=1, le=1000)
    regenerate_shap: bool = Field(default=False)

    @field_validator("dataset_id", "experiment_id", "source_version_id")
    @classmethod
    def _strip(cls, value: str | None) -> str | None:
        if value is None:
            return None
        v = value.strip()
        return v or None


def _err(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"success": False, "error": {"code": code, "message": message}},
    )


@router.post(
    "/analyze",
    summary="AI Data Analyst Agent：自然语言提问 → 自动调用工具 → 生成分析报告",
    description=(
        "由 Analyst Agent 根据用户问题自动选择 EDA / ML / SHAP / Dataset / Report 工具，"
        "将工具结果作为上下文交给 LLM（或 Mock）生成结构化分析报告。"
        "如未配置 ``LLM_API_KEY / LLM_BASE_URL / LLM_MODEL``，自动降级为 Mock 模式。"
    ),
)
def analyze_endpoint(request: AnalyzeRequest):
    try:
        result = analyze(
            question=request.question,
            dataset_id=request.dataset_id,
            experiment_id=request.experiment_id,
            source_version_id=request.source_version_id,
            max_summary_rows=request.max_summary_rows,
            regenerate_shap=request.regenerate_shap,
        )
    except AgentError as exc:
        logger.warning("agent error [%s] %s", exc.code, exc.message)
        return _err(exc.status_code, exc.code, exc.message)
    except Exception as exc:  # noqa: BLE001
        logger.exception("agent unexpected error")
        return _err(500, "internal_error", f"Agent 任务执行失败：{exc}")
    return result
