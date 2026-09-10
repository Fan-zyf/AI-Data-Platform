"""v0.6 模型可解释性（SHAP）请求/响应模型。

请求结构相对简单：只需要可选的 sample_indices 与 max_summary_rows。
响应使用动态 dict（与 v0.5 一致），这里只声明最常用的子结构。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class ExplainRequest(BaseModel):
    """POST /api/ml/experiments/{experiment_id}/explain 的请求体（可空）。"""

    sample_indices: list[int] | None = Field(
        default=None,
        description=(
            "需要在单点解释中返回贡献的样本索引（相对于 SHAP 采样的行）。"
            "缺省时后端自动挑选若干行；负值 / 越界会被后端过滤。"
        ),
    )
    max_summary_rows: int | None = Field(
        default=None,
        description="覆盖默认的 summary 采样行数（1~1000）。",
    )
    regenerate: bool = Field(
        default=False,
        description="强制覆盖已有解释结果（默认 False，重复请求会复用缓存）。",
    )

    @field_validator("max_summary_rows")
    @classmethod
    def _max_summary_rows_in_range(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if value < 1 or value > 1000:
            raise ValueError("max_summary_rows 需在 1~1000 之间")
        return int(value)

    @field_validator("sample_indices")
    @classmethod
    def _sample_indices_bounded(cls, value: list[int] | None) -> list[int] | None:
        if value is None:
            return None
        if len(value) > 20:
            raise ValueError("单次最多请求 20 个样本的解释")
        return [int(v) for v in value]


class SHAPFeatureImportance(BaseModel):
    """全局特征重要性。"""

    feature_names: list[str] = Field(default_factory=list)
    importance: list[float] = Field(default_factory=list)
    n_rows_used: int = 0


class SHAPContribution(BaseModel):
    """单样本的单个特征贡献。"""

    feature: str
    value: Any = None
    shap: float


class SHAPSampleExplanation(BaseModel):
    """单样本的完整解释。"""

    index: int
    prediction: Any
    base_value: float
    contributions: list[SHAPContribution] = Field(default_factory=list)
