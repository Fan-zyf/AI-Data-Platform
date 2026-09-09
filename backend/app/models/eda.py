"""自动 EDA 的 Pydantic 响应模型。

GET /api/datasets/{dataset_id}/eda 的返回结构由此处显式定义，方便：
- Swagger /docs 直接查看 API Schema；
- 前端对响应结构做静态依赖；
- 后续数据清洗 / 特征工程模块在相同语义上扩展。
"""

from typing import Any

from pydantic import BaseModel, Field

from app.models.data import DatasetInfo


class NumericSummary(BaseModel):
    """单个数值变量的描述统计。"""

    column: str = Field(..., description="字段名")
    count: int = Field(..., description="有效数值个数（已剔除缺失与非有限值）")
    missing_count: int = Field(..., description="缺失值数量")
    missing_percentage: float = Field(..., description="缺失比例（0-100）")
    mean: float | None = Field(None, description="均值")
    std: float | None = Field(None, description="标准差")
    min: float | None = Field(None, description="最小值")
    q1: float | None = Field(None, description="下四分位数 Q1")
    median: float | None = Field(None, description="中位数 Q2")
    q3: float | None = Field(None, description="上四分位数 Q3")
    max: float | None = Field(None, description="最大值")
    range: float | None = Field(None, description="极差 = max - min")
    iqr: float | None = Field(None, description="四分位距 = Q3 - Q1")
    skewness: float | None = Field(None, description="偏度（样本数过少或无法计算时为 null）")
    unique_count: int = Field(0, description="去重后的不同数值个数")
    zero_count: int = Field(0, description="等于 0 的数值个数")


class NumericDistribution(BaseModel):
    """单个数值变量的直方图数据（JSON safe，不含图片）。"""

    column: str = Field(..., description="字段名")
    bin_edges: list[float] = Field(default_factory=list, description="分箱边界（长度 = counts + 1）")
    counts: list[int] = Field(default_factory=list, description="每个分箱的样本数")


class TopValue(BaseModel):
    """分类取值的频次条目。"""

    value: str = Field(..., description="类别取值")
    count: int = Field(..., description="出现次数")
    percentage: float = Field(..., description="占非缺失值的比例（0-100）")


class CategoricalSummary(BaseModel):
    """单个分类（含布尔/低基数文本）变量的分布。"""

    column: str = Field(..., description="字段名")
    inferred_type: str = Field(..., description="业务推断类型")
    count: int = Field(..., description="非缺失值个数")
    missing_count: int = Field(..., description="缺失值数量")
    missing_percentage: float = Field(..., description="缺失比例（0-100）")
    unique_count: int = Field(..., description="不同取值个数")
    top_values: list[TopValue] = Field(default_factory=list, description="频次最高的前 N 个取值")


class MissingColumnStat(BaseModel):
    """单字段缺失情况（按缺失率降序）。"""

    column: str = Field(..., description="字段名")
    missing_count: int = Field(..., description="缺失值数量")
    missing_percentage: float = Field(..., description="缺失比例（0-100）")


class MissingAnalysis(BaseModel):
    """缺失值整体分析。"""

    total_missing: int = Field(..., description="缺失单元格总数")
    total_cells: int = Field(..., description="单元格总数 = 行数 x 列数")
    overall_missing_percentage: float = Field(..., description="整体缺失率（0-100）")
    by_column: list[MissingColumnStat] = Field(
        default_factory=list, description="存在缺失的字段明细（缺失率降序）"
    )
    columns_with_missing: list[str] = Field(default_factory=list, description="含缺失字段名")
    columns_without_missing: list[str] = Field(default_factory=list, description="无缺失字段名")


class OutlierSummary(BaseModel):
    """单个数值变量的 IQR 异常值检测结果。"""

    column: str = Field(..., description="字段名")
    detection_method: str = Field("IQR", description="异常值检测方法")
    lower_bound: float | None = Field(None, description="下界 = Q1 - 1.5 * IQR")
    upper_bound: float | None = Field(None, description="上界 = Q3 + 1.5 * IQR")
    outlier_count: int = Field(0, description="异常值个数")
    outlier_percentage: float = Field(0.0, description="异常值占有效数值的比例（0-100）")


class TopCorrelation(BaseModel):
    """一对数值变量的 Pearson 相关系数。"""

    variable_1: str = Field(..., description="变量一")
    variable_2: str = Field(..., description="变量二")
    correlation: float = Field(..., description="Pearson 相关系数")
    abs_correlation: float = Field(..., description="相关系数绝对值")


class CorrelationAnalysis(BaseModel):
    """数值变量 Pearson 相关矩阵（受规模上限保护）。"""

    columns: list[str] = Field(default_factory=list, description="参与分析的数值字段（矩阵行列顺序）")
    matrix: list[list[float | None]] = Field(
        default_factory=list, description="相关矩阵；无法计算的单元为 null"
    )
    top_correlations: list[TopCorrelation] = Field(
        default_factory=list, description="按 |r| 降序的非重复变量对 Top N"
    )


class EDASummary(BaseModel):
    """EDA 规则型摘要（纯 Python 规则计算，不使用 LLM）。"""

    total_rows: int = Field(..., description="总行数")
    total_columns: int = Field(..., description="总列数")
    numeric_columns: int = Field(..., description="数值型字段数量")
    categorical_columns: int = Field(..., description="参与分类分析的字段数量（分类+布尔+低基数文本）")
    datetime_columns: int = Field(..., description="日期时间字段数量")
    text_columns: int = Field(..., description="自由文本字段数量")
    high_missing_columns: int = Field(..., description="高缺失率字段数量")
    columns_with_outliers: int = Field(..., description="检出 IQR 异常值的字段数量")
    strong_correlations: int = Field(..., description="|r| 超过阈值的强相关变量对数")
    rule_based_insights: list[str] = Field(default_factory=list, description="基于真实统计结果的规则型洞察")


class EDAResponse(BaseModel):
    """GET /api/datasets/{dataset_id}/eda 的成功响应结构。"""

    success: bool = True
    dataset_id: str = Field(..., description="数据集会话 ID")
    dataset: DatasetInfo = Field(..., description="数据集基础信息")
    summary: EDASummary = Field(..., description="EDA 摘要与洞察")
    numeric_summaries: list[NumericSummary] = Field(default_factory=list)
    numeric_distributions: list[NumericDistribution] = Field(default_factory=list)
    categorical_summaries: list[CategoricalSummary] = Field(default_factory=list)
    missing_analysis: MissingAnalysis = Field(..., description="缺失值分析")
    outlier_analysis: list[OutlierSummary] = Field(default_factory=list)
    correlation: CorrelationAnalysis = Field(..., description="相关性分析")
