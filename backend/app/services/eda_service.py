"""自动 EDA 服务：基于 pandas / numpy 向量化计算真实统计结果并返回结构化 JSON。

设计约定：
- 不生成 HTML 报告、不生成图片文件、不返回完整 DataFrame，只返回统计汇总与可视化所需数据；
- 本模块为纯规则计算，不使用 LLM，不编造统计结论；
- NaN / Inf / numpy 类型一律归一化为 None 或 Python 原生类型，保证 JSON serialization 安全；
- 单次 EDA 只解析一次数据，随后全部使用向量化操作。
"""

import logging
import math

import numpy as np
import pandas as pd

from app.core.config import settings
from app.models.data import BOOLEAN, CATEGORICAL, DATETIME, NUMERIC, TEXT, DatasetInfo
from app.models.eda import (
    CategoricalSummary,
    CorrelationAnalysis,
    EDAResponse,
    EDASummary,
    MissingAnalysis,
    MissingColumnStat,
    NumericDistribution,
    NumericSummary,
    OutlierSummary,
    TopCorrelation,
    TopValue,
)
from app.services import data_service as ds
from app.services import version_manager as vm

logger = logging.getLogger(__name__)

# 分类展示时类别值的最长截断长度（仅影响展示字段，不影响计算）
_MAX_DISPLAY_VALUE_LENGTH = 200


def _f6(value: float | None) -> float | None:
    """归一化为最多 6 位小数的浮点数；NaN/Inf/异常一律返回 None。"""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, 6) if math.isfinite(number) else None


def _pct(value: float) -> float:
    """百分比统一保留 2 位小数。"""
    return round(float(value), 2)


def _category_label(value) -> str:
    """把类别取值转换为适合展示的字符串。"""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    text = str(value)
    return text[:_MAX_DISPLAY_VALUE_LENGTH]


def _finite_series(series: pd.Series) -> pd.Series:
    """把任意列数值化（coerce）并剔除缺失与非有限值，保留原索引以便对齐。"""
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric[np.isfinite(numeric.to_numpy(dtype="float64"))]


# ============================================================
# 各分析模块
# ============================================================


def _build_numeric_summary(label: str, series: pd.Series, missing_mask: pd.Series) -> NumericSummary:
    """单个数值变量的描述统计。"""
    total = len(series)
    missing_count = int(missing_mask.sum())
    values = _finite_series(series)
    count = int(values.size)
    missing_percentage = _pct(missing_count / total * 100) if total else 0.0

    stats: dict[str, float | None] = {
        "mean": None,
        "std": None,
        "min": None,
        "q1": None,
        "median": None,
        "q3": None,
        "max": None,
        "range": None,
        "iqr": None,
        "skewness": None,
    }

    if count:
        arr = values.to_numpy(dtype="float64")
        q1, median, q3 = np.quantile(arr, [0.25, 0.5, 0.75])
        stats["mean"] = _f6(arr.mean())
        # 与 pandas describe()/Series.std() 保持一致：样本标准差（ddof=1）
        stats["std"] = _f6(float(values.std()))
        stats["min"] = _f6(arr.min())
        stats["q1"] = _f6(q1)
        stats["median"] = _f6(median)
        stats["q3"] = _f6(q3)
        stats["max"] = _f6(arr.max())
        stats["range"] = _f6(float(arr.max()) - float(arr.min()))
        stats["iqr"] = _f6(float(q3) - float(q1))
        if count >= 3:
            try:
                stats["skewness"] = _f6(float(pd.Series(arr).skew()))
            except (ValueError, ZeroDivisionError):  # pragma: no cover - 底层数值边界
                stats["skewness"] = None

    return NumericSummary(
        column=label,
        count=count,
        missing_count=missing_count,
        missing_percentage=missing_percentage,
        unique_count=int(values.nunique(dropna=True)),
        zero_count=int((values == 0).sum()),
        **stats,
    )


def _build_distribution(label: str, series: pd.Series, missing_mask: pd.Series) -> NumericDistribution:
    """单个数值变量的直方图数据（合理自动分箱，处理常量/极少样本/极端值）。"""
    values = _finite_series(series)
    count = int(values.size)

    if count == 0:
        return NumericDistribution(column=label, bin_edges=[], counts=[])

    arr = values.to_numpy(dtype="float64")
    lo, hi = float(arr.min()), float(arr.max())

    # 常量列：为避免 numpy.histogram 对零宽度区间报错，生成对称单箱
    if lo == hi:
        return NumericDistribution(
            column=label,
            bin_edges=[round(lo - 0.5, 6), round(hi + 0.5, 6)],
            counts=[count],
        )

    target_bins = int(
        min(
            settings.eda_max_histogram_bins,
            max(settings.eda_min_histogram_bins, math.ceil(math.sqrt(count))),
        )
    )
    counts, edges = np.histogram(arr, bins=target_bins)
    return NumericDistribution(
        column=label,
        bin_edges=[round(float(edge), 6) for edge in edges],
        counts=[int(value) for value in counts],
    )


def _is_category_eligible(inferred_type: str, non_null_count: int, unique_count: int) -> bool:
    """分类/布尔列直接参与；低基数 text 列也参与类别分析。"""
    if inferred_type in (CATEGORICAL, BOOLEAN):
        return True
    if inferred_type == TEXT and non_null_count > 1:
        ratio = unique_count / non_null_count
        if ratio <= settings.eda_low_cardinality_text_ratio:
            return True
    return False


def _build_categorical_summary(
    label: str, inferred_type: str, series: pd.Series, missing_mask: pd.Series
) -> CategoricalSummary:
    """单个分类变量的分布（只返回 Top N 频次，避免把数千类别全量返回）。"""
    total = len(series)
    missing_count = int(missing_mask.sum())
    non_null = total - missing_count
    non_missing = series.loc[~missing_mask]

    counts = non_missing.value_counts(dropna=True)
    unique_count = int(counts.size)
    top_n = settings.eda_top_values

    top_values = [
        TopValue(
            value=_category_label(value),
            count=int(count),
            percentage=_pct(count / non_null * 100) if non_null else 0.0,
        )
        for value, count in list(counts.head(top_n).items())
    ]

    return CategoricalSummary(
        column=label,
        inferred_type=inferred_type,
        count=non_null,
        missing_count=missing_count,
        missing_percentage=_pct(missing_count / total * 100) if total else 0.0,
        unique_count=unique_count,
        top_values=top_values,
    )


def _build_missing_analysis(
    labeled_masks: list[tuple[str, pd.Series]],
    rows: int,
    columns: int,
) -> MissingAnalysis:
    """整体缺失分析，按缺失率降序返回明细。"""
    by_column: list[MissingColumnStat] = []
    with_missing: list[str] = []
    without_missing: list[str] = []

    for label, mask in labeled_masks:
        missing_count = int(mask.sum())
        percentage = _pct(missing_count / rows * 100) if rows else 0.0
        if missing_count > 0:
            by_column.append(
                MissingColumnStat(column=label, missing_count=missing_count, missing_percentage=percentage)
            )
            with_missing.append(label)
        else:
            without_missing.append(label)

    by_column.sort(key=lambda item: (-item.missing_percentage, item.column))
    total_missing = sum(item.missing_count for item in by_column)
    total_cells = rows * columns

    return MissingAnalysis(
        total_missing=total_missing,
        total_cells=total_cells,
        overall_missing_percentage=_pct(total_missing / total_cells * 100) if total_cells else 0.0,
        by_column=by_column,
        columns_with_missing=with_missing,
        columns_without_missing=without_missing,
    )


def _build_outlier(label: str, series: pd.Series, missing_mask: pd.Series) -> OutlierSummary:
    """IQR 异常值检测（Q1-1.5IQR / Q3+1.5IQR）。

    IQR=0（含常量列）时边界收紧为 Q1/Q3 本身，不会误伤大量正常值。
    """
    values = _finite_series(series)
    count = int(values.size)

    if count == 0:
        return OutlierSummary(column=label, detection_method="IQR")

    arr = values.to_numpy(dtype="float64")
    q1, q3 = np.quantile(arr, [0.25, 0.75])
    iqr = float(q3) - float(q1)
    lower = float(q1) - 1.5 * iqr
    upper = float(q3) + 1.5 * iqr
    outlier_count = int(np.sum((arr < lower) | (arr > upper)))

    return OutlierSummary(
        column=label,
        detection_method="IQR",
        lower_bound=_f6(lower),
        upper_bound=_f6(upper),
        outlier_count=outlier_count,
        outlier_percentage=_pct(outlier_count / count * 100),
    )


def _build_correlation(
    candidates: list[tuple[str, pd.Series]],
) -> tuple[CorrelationAnalysis, list[TopCorrelation]]:
    """Pearson 相关矩阵 + 按 |r| 降序的非重复变量对。

    规模保护：数值字段超过上限时，按（非空数量、方差）排序截断；
    常量字段（唯一值 < 2）不参与，避免产生无意义的 NaN 矩阵。
    """
    empty = CorrelationAnalysis(), []
    if len(candidates) < 2:
        return empty

    # 常量数值字段剔除：只有至少两个不同取值才可能产生有意义的相关
    meaningful = []
    for label, series in candidates:
        if int(series.nunique(dropna=True)) >= 2:
            meaningful.append((label, series))
    if len(meaningful) < 2:
        return empty

    # 规模保护：超过上限时按（非空数量、标准差）降序择优保留
    max_columns = settings.eda_max_correlation_columns
    if len(meaningful) > max_columns:
        meaningful.sort(
            key=lambda item: (
                -int(item[1].size),
                -(item[1].std(ddof=0) if math.isfinite(item[1].std(ddof=0)) else 0.0),
            )
        )
        meaningful = meaningful[:max_columns]

    labels = [label for label, _ in meaningful]
    frame = pd.DataFrame({label: series for label, series in meaningful})
    corr = frame.corr()  # 全为数值列；pairwise 缺失对齐

    matrix = []
    for row_label in labels:
        matrix.append([_f6(corr.at[row_label, col_label]) for col_label in labels])

    pairs: list[TopCorrelation] = []
    for i, label_i in enumerate(labels):
        for j in range(i + 1, len(labels)):
            label_j = labels[j]
            value = corr.at[label_i, label_j]
            rounded = _f6(value)
            if rounded is None:
                continue
            pairs.append(
                TopCorrelation(
                    variable_1=label_i,
                    variable_2=label_j,
                    correlation=rounded,
                    abs_correlation=abs(rounded),
                )
            )
    pairs.sort(key=lambda pair: (-pair.abs_correlation, pair.variable_1, pair.variable_2))

    analysis = CorrelationAnalysis(
        columns=labels,
        matrix=matrix,
        top_correlations=pairs[: settings.eda_top_correlations],
    )
    return analysis, pairs


# ============================================================
# 洞察与组装
# ============================================================


def _build_insights(
    rows: int,
    cols: int,
    numeric_labels: list[str],
    category_labels: list[str],
    missing_analysis: MissingAnalysis,
    high_missing_labels: list[str],
    constant_numeric_labels: list[str],
    outliers: list[OutlierSummary],
    pairs: list[TopCorrelation],
) -> list[str]:
    """生成完全基于真实统计结果的规则型洞察文案。"""
    insights: list[str] = []
    threshold = settings.eda_strong_correlation_threshold
    missing_threshold = settings.eda_high_missing_threshold_percent

    insights.append(f"数据集共 {rows} 行 {cols} 列，其中数值字段 {len(numeric_labels)} 个、参与分类分析的字段 {len(category_labels)} 个。")

    if not numeric_labels:
        insights.append("未检测到数值型字段，数值统计、异常值检测与相关性分析已跳过。")
    if not category_labels:
        insights.append("未检测到适合分类分析的字段（分类 / 布尔 / 低基数文本）。")

    total_missing = missing_analysis.total_missing
    if total_missing > 0:
        insights.append(
            f"数据整体缺失率为 {missing_analysis.overall_missing_percentage:.2f}%，"
            f"共 {total_missing} 个缺失单元格，分布在 {len(missing_analysis.columns_with_missing)} 个字段。"
        )
        if high_missing_labels:
            detail = "、".join(
                f"{item.column}（{item.missing_percentage:.1f}%）"
                for item in missing_analysis.by_column
                if item.column in high_missing_labels
            )
            insights.append(
                f"存在 {len(high_missing_labels)} 个高缺失率字段（缺失率 ≥ {missing_threshold:.0f}%）：{detail}。"
            )
    else:
        insights.append("数据不存在缺失值。")

    if constant_numeric_labels:
        joined = "、".join(constant_numeric_labels[:5])
        more = " 等" if len(constant_numeric_labels) > 5 else ""
        insights.append(f"字段 {joined}{more} 为常量数值字段，信息量较低，已排除在相关性分析之外。")

    nonzero_outliers = [item for item in outliers if item.outlier_count > 0]
    nonzero_outliers.sort(key=lambda item: (-item.outlier_percentage, item.column))
    for item in nonzero_outliers[:5]:
        insights.append(
            f"字段 {item.column} 通过 IQR 检测到 {item.outlier_count} 个异常值（占比 {item.outlier_percentage:.2f}%）。"
        )
    if numeric_labels and not nonzero_outliers:
        insights.append("所有数值字段均未检出 IQR 异常值。")

    if pairs:
        strong_pairs = [pair for pair in pairs if pair.abs_correlation >= threshold]
        if strong_pairs:
            for pair in strong_pairs[:5]:
                relation = "正相关" if pair.correlation > 0 else "负相关"
                insights.append(
                    f"字段 {pair.variable_1} 与 {pair.variable_2} 存在较强的{relation}"
                    f"（Pearson r = {pair.correlation:.3f}）。"
                )
        elif len(numeric_labels) >= 2:
            insights.append(f"未发现 |r| ≥ {threshold} 的强相关数值对。")
    elif len(numeric_labels) >= 2:
        insights.append("数值字段不足以计算相关矩阵，相关性分析已跳过。")

    return insights


def compute_eda(frame: pd.DataFrame, dataset: DatasetInfo, dataset_id: str) -> EDAResponse:
    """基于 DataFrame 执行全部 EDA 计算（纯函数，便于复用与测试）。"""
    rows, cols = len(frame), len(frame.columns)
    profiles, masks = ds.analyze_columns(frame)
    # labeled = (字段显示名, 对应数据列 Series, 画像, 缺失掩码)
    labeled = [
        (str(df_col), frame[df_col], profile, mask)
        for df_col, profile, mask in zip(frame.columns, profiles, masks)
    ]

    types_by_label = {item[0]: item[2].inferred_type for item in labeled}
    type_counts: dict[str, int] = {}
    for inferred in types_by_label.values():
        type_counts[inferred] = type_counts.get(inferred, 0) + 1

    # ---- 数值分析（描述统计 + 分布 + 异常值 + 相关性候选）----
    numeric_summaries: list[NumericSummary] = []
    numeric_distributions: list[NumericDistribution] = []
    outlier_analysis: list[OutlierSummary] = []
    correlation_candidates: list[tuple[str, pd.Series]] = []
    constant_numeric_labels: list[str] = []

    for label, series, profile, mask in labeled:
        if profile.inferred_type != NUMERIC:
            continue
        numeric_summaries.append(_build_numeric_summary(label, series, mask))
        numeric_distributions.append(_build_distribution(label, series, mask))
        outlier_analysis.append(_build_outlier(label, series, mask))
        if profile.unique_count <= 1 and profile.non_null_count > 0:
            constant_numeric_labels.append(label)
        else:
            values = _finite_series(series)
            if int(values.size) >= 1:
                correlation_candidates.append((label, values))

    numeric_labels = [item.column for item in numeric_summaries]
    correlation, all_pairs = _build_correlation(correlation_candidates)

    # ---- 分类分析 ----
    categorical_summaries: list[CategoricalSummary] = []
    category_labels: list[str] = []
    for label, series, profile, mask in labeled:
        if not _is_category_eligible(profile.inferred_type, profile.non_null_count, profile.unique_count):
            continue
        category_labels.append(label)
        categorical_summaries.append(
            _build_categorical_summary(label, profile.inferred_type, series, mask)
        )

    # ---- 缺失分析 ----
    labeled_masks: list[tuple[str, pd.Series]] = [(item[0], item[3]) for item in labeled]
    missing_analysis = _build_missing_analysis(labeled_masks, rows, cols)

    high_missing_threshold = settings.eda_high_missing_threshold_percent
    high_missing_labels = [
        item.column
        for item in missing_analysis.by_column
        if item.missing_percentage >= high_missing_threshold
    ]

    summary = EDASummary(
        total_rows=rows,
        total_columns=cols,
        numeric_columns=type_counts.get(NUMERIC, 0),
        categorical_columns=len(category_labels),
        datetime_columns=type_counts.get(DATETIME, 0),
        text_columns=type_counts.get(TEXT, 0),
        high_missing_columns=len(high_missing_labels),
        columns_with_outliers=sum(1 for item in outlier_analysis if item.outlier_count > 0),
        strong_correlations=sum(1 for pair in all_pairs if pair.abs_correlation >= settings.eda_strong_correlation_threshold),
        rule_based_insights=_build_insights(
            rows=rows,
            cols=cols,
            numeric_labels=numeric_labels,
            category_labels=category_labels,
            missing_analysis=missing_analysis,
            high_missing_labels=high_missing_labels,
            constant_numeric_labels=constant_numeric_labels,
            outliers=outlier_analysis,
            pairs=all_pairs,
        ),
    )

    return EDAResponse(
        success=True,
        dataset_id=dataset_id,
        dataset=dataset,
        summary=summary,
        numeric_summaries=numeric_summaries,
        numeric_distributions=numeric_distributions,
        categorical_summaries=categorical_summaries,
        missing_analysis=missing_analysis,
        outlier_analysis=outlier_analysis,
        correlation=correlation,
    )


def run_eda(dataset_id: str, version_id: str | None = None) -> EDAResponse:
    """按 dataset_id（可选 version_id，缺省为 original）加载数据并执行自动 EDA。

    v0.4 起数据统一经由 version_manager 加载：
    - version_id 为空 / 'original' → 读取原始 source 文件（与 v0.3 行为一致）；
    - version_id 为 UUID → 读取对应派生版本 versions/<uuid>/data.csv。
    """
    frame, info = vm.load_dataset_version(dataset_id, version_id)
    dataset = DatasetInfo(
        file_name=info["file_name"],
        file_type=info["file_type"],
        file_size=info["file_size"],
        rows=info["rows"],
        columns=info["columns"],
    )
    return compute_eda(frame, dataset, dataset_id)
