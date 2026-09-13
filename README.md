# AI Data Intelligence Platform

智能数据分析与预测平台（AI Data Intelligence Platform）

一个可长期扩展的 AI 数据智能分析平台。

**当前进度（v0.7.0）**
- ✅ 基础架构：React + Vite 前端 / FastAPI 后端 / Git
- ✅ CSV、Excel(.xlsx/.xls) 数据上传与解析（只读分析，不修改原始数据）
- ✅ 基础数据画像：字段类型推断、缺失统计、唯一值、前 20 行预览、质量警告
- ✅ 临时数据集会话（Dataset Session）：上传即建会话、独立目录、自动过期、可手动删除
- ✅ 自动 EDA：数值统计 / 直方图、分类分布、缺失值分析、IQR 异常值、Pearson 相关、规则型洞察摘要，且支持 `?version_id=` 对任意数据版本执行
- ✅ 前端 ECharts 可视化：直方图、分类条形图、缺失热图、异常值箱线、相关性热力图
- ✅ 数据处理与版本管理（v0.4）：Transformation Plan → Preview → Apply → 新数据版本
  - 结构化 JSON 操作（9 种，严格白名单，绝不执行任意代码 / eval / exec）
  - Preview 仅内存执行不落盘；Apply 全部成功才写入新 UUID 版本（原子），失败不留半成品
  - 原始数据永不修改；派生版本存于 `runtime/datasets/<dataset_id>/versions/<uuid>/`
  - 数据版本历史 / 版本对比 / 删除派生版本 / 对指定版本运行 EDA
- ✅ 机器学习建模与预测（v0.5）：
  - 防泄漏训练管线：train/test split → 训练集内 CV → 候选模型对比 → 测试集最终评估 → 实验持久化
  - 分类（Dummy / LogisticRegression / RandomForest）与回归（Dummy / Ridge / RandomForest）候选模型
  - 自动特征推断 / 显式特征选择 / ID-like 高基数排除 / 测试样本泄漏警告
  - 实验管理：列表 / 详情 / 删除 / 版本删除保护（实验引用版本时 409）
  - 批量预测：基于持久化 Pipeline 对新记录打分，支持分类概率
  - 前端 ML 面板：训练配置、CV 对比表、混淆矩阵、ROC、回归散点 / 残差、预测结果表
- ✅ 模型可解释性（v0.6）：
  - SHAP 解释（Tree / Linear / Kernel explainer 按模型类型自动路由）
  - 全局特征重要性 + SHAP Summary 散点 + 单点预测解释
  - 解释结果持久化到 `runtime/datasets/<id>/ml/experiments/<exp>/shap_result.json`，与 `metadata.json#explainability` 同步
  - 跨会话定位 experiment_id，dataset-scoped 入口校验归属
  - 重复请求默认命中缓存，`regenerate=true` 强制重算
  - 实验删除级联清理 SHAP 文件（沿用 v0.5 `shutil.rmtree`）
  - 前端 SHAP 面板：生成 / 缓存 / 强制重算 + 三个 ECharts 图表
- ✅ **AI Data Analyst Agent（v0.7）**：
  - 5 个 Tool：`dataset` / `eda` / `ml` / `shap` / `report`，对 v0.4~v0.6 服务做轻量包装（不写新数据）
  - ToolRegistry 关键词路由 + experiment 必备校验 + `agent_max_tools` 上限 + dataset/eda 兜底
  - LLM 客户端：未配置环境变量时自动用 Mock（从工具结果拼装中文分析报告）；配置 `LLM_API_KEY`/`LLM_BASE_URL`/`LLM_MODEL` 后切换 OpenAI-compatible HTTP 客户端
  - 真实 LLM 失败自动降级 Mock，不阻塞分析
  - 统一 Prompt：`<<<TOOL_RESULTS>>>` ... `<<<END_TOOL_RESULTS>>>` 包裹结构化 JSON，Mock 解析与真实 LLM 共享同一模板
  - API：`POST /api/agent/analyze`，请求 `{question, dataset_id, experiment_id?}`，响应 `{success, answer, insights, recommendations, tools_used, tool_trace, plan, llm, raw_compose, app_version}`；错误码：`dataset_not_found` / `experiment_not_found` / `agent_invalid_request`
  - 前端 Agent 面板：自然语言提问 + dataset_id / experiment_id 输入 + 快捷问题按钮 + 工具 trace 列表 + 回答 / 洞察 / 建议
  - 后端测试 25 个 mock + Agent 用例 + 134 个总测试全绿
  - 浏览器 UI E2E：upload → train → SHAP → 输入问题 → Agent 回答截图
- ⬜ 后续规划：技能 / MCP 扩展、LLM 多模态、Agent 规划多轮

## 技术栈

| 层 | 技术 |
| --- | --- |
| 前端 | React + Vite + ECharts |
| 后端 | Python 3.14 + FastAPI |
| 数据处理 | pandas、numpy、openpyxl、xlrd |
| 机器学习 | scikit-learn 1.9.0、joblib |
| 模型解释 | SHAP 0.52.0（Tree / Linear / Kernel） |
| 后续扩展 | XGBoost、LLM、Skill、MCP |
| 版本管理 | Git |

> v0.6 新增 Python 依赖：**shap==0.52.0**（含 numba / llvmlite / slicer 间接依赖，Python 3.14 已配 prebuilt wheel）。
> v0.5 新增 Python 依赖：**scikit-learn==1.9.0**、**joblib**（Pipeline 持久化）。
> v0.4 新增 Python 依赖：无（数据清洗/特征工程全部用已有 pandas/numpy 实现）。

## 目录结构

```
AI-Data-Platform/
├── backend/
│   ├── app/
│   │   ├── main.py            # 应用入口（注册 /api 路由，版本 0.4.0）
│   │   ├── api/
│   │   │   ├── health.py      # GET /api/health
│   │   │   ├── data.py        # POST /api/data/upload
│   │   │   ├── datasets.py    # GET /api/datasets/{id}/eda?version_id=、DELETE /api/datasets/{id}
│   │   │   ├── processing.py  # v0.4：preview / apply / versions / compare / delete-version
│   │   │   ├── ml.py          # v0.5：train / experiments / predict
│   │   │   └── explainability.py  # v0.6：POST /api/ml/experiments/{exp_id}/explain（dataset-scoped 同义入口）
│   │   ├── core/config.py     # 配置（端口、CORS、上传限制、ML_*、ML_SHAP_* 等）
│   │   ├── services/
│   │   │   ├── data_service.py         # 文件解析 / 数据画像 / 质量分析
│   │   │   ├── dataset_manager.py      # Dataset Session 的创建 / 读取 / 过期清理 / 删除
│   │   │   ├── eda_service.py          # v0.4：run_eda(dataset_id, version_id=None) 版本感知
│   │   │   ├── processing_service.py   # v0.4：预览 / 应用 9 种变换操作
│   │   │   ├── version_manager.py      # v0.4：UUID 版本目录 / 加载 / 列表 / 对比 / 删除 + v0.5 ML 引用保护
│   │   │   ├── ml_service.py           # v0.5：训练管线 / CV / 测试评估 / 预测
│   │   │   ├── experiment_manager.py   # v0.5：实验目录 / 持久化 / 列表 / 详情 / 删除 / 引用检查 + v0.6 跨会话定位 / SHAP 写回
│   │   │   ├── preprocessing_factory.py # v0.5：为 sklearn Pipeline 生成 ColumnTransformer
│   │   │   └── shap_service.py         # v0.6：Tree/Linear/Kernel explainer 路由 + 全局/summary/单点解释
│   │   └── models/
│   │       ├── data.py        # 上传响应的 Pydantic 模型
│   │       ├── eda.py         # 自动 EDA 响应的 Pydantic 模型
│   │       ├── processing.py  # v0.4：TransformationPlan / Operation / 版本相关响应模型
│   │       ├── ml.py          # v0.5：训练 / 预测 / 实验的 Pydantic 模型
│   │       └── explainability.py  # v0.6：ExplainRequest / SHAPFeatureImportance / SHAPContribution / SHAPSampleExplanation
│   ├── runtime/               # 运行时临时数据（会话目录 + 派生版本 + ML 实验 + SHAP，自动清理，勿提交）
│   ├── tests/                 # pytest 自动化测试（109 个用例：v0.4 65 + v0.5 32 + v0.6 12）
│   └── requirements.txt
├── data/
│   └── sample/demo.csv        # 示例数据（含数值/分类/文本/日期/布尔列、缺失与重复）
├── frontend/
│   ├── src/
│   │   ├── App.jsx            # 首页（系统状态 + 数据上传 + EDA + 数据处理入口）
│   │   ├── upload/UploadPanel.jsx   # 上传与结果展示（连接 DataProcessingPanel）
│   │   ├── eda/               # EDA 可视化面板（ECharts）
│   │   │   ├── EDASection.jsx      # 版本感知的 EDA 容器（forwardRef + analyze(versionId)）
│   │   │   ├── HistogramPanel.jsx / CategoricalPanel.jsx / MissingChart.jsx
│   │   │   ├── OutlierPanel.jsx / CorrelationHeatmap.jsx / NumericTable.jsx / EChart.jsx
│   │   │   └── chartTheme.js
│   │   ├── processing/        # v0.4 数据处理面板（位于上传/EDA 之后）
│   │   │   ├── DataProcessingPanel.jsx   # 总控制器：版本/步骤/预览/应用/历史/对比
│   │   │   ├── opDefinitions.js          # 前后端共享的操作 schema 与 step<->JSON 转换
│   │   │   ├── TransformationBuilder.jsx # 步骤卡片构建器（加/排/删/参数校验）
│   │   │   ├── PreviewTabs（内置）        # 处理前/后数据与指标对比
│   │   │   ├── VersionHistory.jsx        # 版本历史（切换源/EDA/删除）
│   │   │   ├── VersionCompare.jsx        # 版本 A/B 对比
│   │   │   └── DataTable.jsx             # 通用数据表（null 渲染/行数截断）
│   │   └── ml/                # v0.5 机器学习面板
│   │       ├── MLPanel.jsx             # 训练配置 / 实验历史 / 批量预测 / 解释入口
│   │       └── ExperimentReport.jsx    # 实验报告：CV 对比 / CM / ROC / 回归 / 残差
│   │   └── explainability/     # v0.6 模型可解释性面板
│   │       ├── ExplainabilityPanel.jsx     # 主容器：生成 / 缓存 / 强制重算 + 错误提示
│   │       ├── FeatureImportanceChart.jsx  # 全局 mean|SHAP| 横向条形图
│   │       ├── SHAPSummaryChart.jsx        # Beeswarm 风格散点（颜色映射原始值）
│   │       └── SampleExplanation.jsx       # 单点预测解释卡片（贡献条形图）
│   ├── index.html
│   └── vite.config.js         # /api 代理到后端
├── .env.example
├── .gitignore
└── README.md
```

### 运行时文件布局（v0.4 版本化 + v0.5 ML 实验 + v0.6 SHAP）

```
backend/runtime/datasets/<dataset_id>/
├── source.<ext>            # 原始文件（永远保留，绝不修改）
├── metadata.json           # Dataset Session 元信息
├── versions/
│   └── <version_id>/       # version_id = 服务端生成的合法 UUID
│       ├── data.csv        # 该版本的清洗/特征数据
│       └── metadata.json   # 版本元信息（父版本 / operations / quality / created_at …）
└── ml/
    └── experiments/
        └── <experiment_id>/    # experiment_id = 服务端生成的合法 UUID
            ├── model.joblib        # 完整 Best Pipeline（预处理 + 模型）
            ├── metadata.json       # 实验元信息（含 explainability 子段：status / result_path / explainer_type / created_at …）
            ├── evaluation.json     # CV 全模型对比 + 最终测试评估（CM / ROC / 散点 / 残差）
            └── shap_result.json    # v0.6 SHAP 解释结果（global / summary / samples / class_label_used / base_value …）
```

版本模型为单向 **parent / child 链**：`original → v1 → v2 …`。
- original 是只读基线，不落盘为版本目录，删除接口明确禁止；
- version_id 一律校验为合法 UUID，路径由服务端拼接，防止路径穿越；
- 派生版本每次独立目录保存，绝不覆盖其他版本。
- **v0.5 新增** 实验被某派生版本引用时，删除该版本返回 409 `version_in_use_by_ml_experiment`；删除实验会同步移除 `model.joblib` 与元信息。

## 快速开始

### 1. 启动后端（FastAPI，端口 8000）

```powershell
cd backend

# 首次：创建虚拟环境并安装依赖
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# 启动开发服务器
.venv\Scripts\python -m uvicorn app.main:app --reload --port 8000
```

- 健康检查：<http://127.0.0.1:8000/api/health>
- 接口文档：<http://127.0.0.1:8000/docs>（Swagger UI）

### 2. 启动前端（React + Vite，端口 5173）

```powershell
cd frontend
npm install      # 首次
npm run dev
```

浏览器访问 <http://localhost:5173>。页面会自动请求 `/api/health` 并显示后端连接状态；下方的“数据上传”区域支持点击或拖拽上传 `.csv / .xlsx / .xls` 文件。上传成功后将自动展示三段式分析：

1. **数据画像（v0.2.0）**：数据集概览、字段信息表、质量警告与前 20 行预览；
2. **自动 EDA（v0.3.0）**：基于临时 Dataset Session，展示数值统计与直方图、分类分布、缺失值分析、IQR 异常值分析与 Pearson 相关性热力图等图表；
3. **数据清洗与特征工程（v0.4.0）**：选择处理基准版本（Original 或任意派生版本）→ Transformation Builder 组合步骤 → Preview（Before/After）→ Apply 生成新 UUID 版本 → 版本历史 / 对比 / 对指定版本再次 EDA。

## API 说明

### GET /api/health

```json
{ "status": "ok", "message": "AI Data Platform backend is running" }
```

### POST /api/data/upload

multipart 表单字段 `file`。仅支持 `.csv / .xlsx / .xls`，最大 20MB（可通过环境变量 `DATA_MAX_UPLOAD_MB` 调整）。

上传成功后会创建一个临时 **Dataset Session**（源文件与画像结果写入 `backend/runtime/datasets/`，默认有效期 `DATASET_TTL_HOURS` 小时），后续 EDA / 数据处理接口基于该会话 ID 读取数据。

成功返回结构：

```json
{
  "success": true,
  "dataset_id": "9f6c…-uuid",
  "created_at": "2026-09-09T12:00:00Z",
  "expires_at": "2026-09-09T20:00:00Z",
  "dataset": { "file_name": "demo.csv", "file_type": "csv", "file_size": 2345, "rows": 23, "columns": 7 },
  "quality": { "total_missing": 7, "columns_with_missing": 3, "duplicate_rows": 1, "empty_columns": 0, "constant_columns": 0 },
  "column_profiles": [
    { "column_name": "age", "dtype": "float64", "inferred_type": "numeric", "non_null_count": 22, "missing_count": 1, "missing_percentage": 4.35, "unique_count": 20 }
  ],
  "preview": [ { "id": 1, "city": "北京", "score": 88.5, "joined": "2024-01-15", "is_active": "true", "notes": "..." } ],
  "warnings": ["发现 3 个包含缺失值的字段（共 7 个缺失单元格）", "发现 1 行重复数据"]
}
```

`inferred_type` 依据数据内容推断，取值包括：`numeric`、`categorical`、`datetime`、`boolean`、`text`。

失败返回结构（HTTP 400/413/422/500）：

```json
{ "success": false, "error": { "code": "empty_file", "message": "上传的文件内容为空，请检查文件后重试。" } }
```

常见错误码：`empty_file`、`unsupported_file_type`、`file_too_large`、`csv_encoding_error`、`csv_parse_error`、`excel_parse_error`、`no_rows`、`internal_error`。

### GET /api/datasets/{dataset_id}/eda

对已上传的数据集会话执行自动 EDA，返回结构化 JSON（响应结构由 `models/eda.py` 显式定义）。

**v0.4 起支持指定版本**：`GET /api/datasets/{dataset_id}/eda?version_id=<uuid>`。不传 `version_id` 或传 `original` 时，行为与 v0.3 完全一致（对原始版本分析）；传入派生版本 UUID 时对清洗后数据执行 EDA，用于对比清洗前后的统计。

```json
{
  "success": true,
  "dataset_id": "…",
  "dataset": { "file_name": "demo.csv", "rows": 23, "columns": 7 },
  "summary": {
    "total_rows": 23, "total_columns": 7,
    "numeric_columns": 2, "categorical_columns": 3,
    "datetime_columns": 1, "text_columns": 1,
    "columns_with_outliers": 1, "strong_correlations": 1,
    "rule_based_insights": ["…强相关洞察 / 缺失提示…"]
  },
  "numeric_summaries": [{ "column": "age", "count": 22, "missing_count": 1, "mean": 33.2, "median": 31.0, "iqr": 15.0, "skewness": 0.8, "outlier_count": 1 }],
  "numeric_distributions": [{ "column": "age", "bin_edges": [0, 10, 20], "counts": [2, 5, 8] }],
  "categorical_summaries": [{ "column": "city", "unique_count": 5, "top_values": [{ "value": "北京", "count": 8, "percentage": 34.8 }] }],
  "missing_analysis": { "total_missing": 7, "total_cells": 161, "overall_missing_percentage": 4.3, "by_column": [] },
  "outlier_analysis": [{ "column": "age", "detection_method": "IQR", "outlier_count": 2, "lower_bound": -2.5, "upper_bound": 57.5 }],
  "correlation": { "columns": ["age", "score"], "matrix": [[1.0, 0.83], [0.83, 1.0]], "top_correlations": [] }
}
```

主要能力：数值列统计与直方图分箱、分类/布尔/低基数文本 TopN、整体与逐列缺失明细、IQR 异常值、数值列 Pearson 相关与 Top 对、纯规则型洞察（未接入 LLM）。

> 设计约定：EDA 只做**只读分析**；分布图以 JSON 分箱数据下发，前端用 ECharts 渲染，后端不生成图片。

### DELETE /api/datasets/{dataset_id}

删除临时 Dataset Session（源文件与全部派生版本一并清理）：

```json
{ "success": true, "message": "Dataset session deleted" }
```

### POST /api/datasets/{dataset_id}/processing/preview（v0.4）

**只在内存执行** Transformation Plan，返回 Before/After 指标与前 20 行预览，**不写盘、不产生新版本**。

请求体（`TransformationPlan`）：

```json
{
  "source_version_id": null,
  "operations": [
    { "type": "drop_duplicates" },
    { "type": "fill_missing", "columns": ["score"], "strategy": "median" }
  ]
}
```

- `source_version_id`：缺省/`null`/`"original"` 表示原始版本；也可传派生版本 UUID（支持对清洗后的数据继续处理）。
- `operations`：至少 1 步，严格**按数组顺序执行**。

成功返回：

```json
{
  "success": true,
  "dataset_id": "…", "source_version_id": "original",
  "rows_before": 23, "rows_after": 22,
  "columns_before": 7, "columns_after": 7,
  "missing_before": 7, "missing_after": 3,
  "duplicates_before": 1, "duplicates_after": 0,
  "generated_columns": [], "removed_columns": [],
  "warnings": [],
  "preview_before": [ { "id": 1, "score": 88.5 } ],
  "preview_after":  [ { "id": 1, "score": 88.5 } ]
}
```

### POST /api/datasets/{dataset_id}/processing/apply（v0.4）

与 preview 相同请求体，但会**原子执行并持久化**：只有全部步骤都成功后，才会把结果写入新的 UUID 版本目录；任何一步失败即中止并返回错误，**绝不留下半成品版本**，原始数据永不修改。

成功返回：

```json
{
  "success": true,
  "dataset_id": "…",
  "version_id": "06ea2452-dd49-4053-8135-3224bc5d35fb",
  "parent_version_id": "original",
  "created_at": "2026-09-09T20:32:40+08:00",
  "rows": 22, "columns": 7,
  "operations_applied": [
    { "type": "drop_duplicates", "effect": { "removed_rows": 1 } },
    { "type": "fill_missing", "columns": ["score"], "strategy": "median", "effect": { "filled_cells": 4 } }
  ],
  "quality": { "total_missing": 3, "duplicate_rows": 0 },
  "warnings": [],
  "preview": [ { "id": 1, "score": 88.5 } ]
}
```

### 版本管理 API（v0.4）

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/datasets/{id}/versions` | 版本列表：`{ original: VersionListItem, versions: [派生版本…] }`，派生按创建时间升序；每项含行/列/缺失/重复与操作摘要 |
| GET | `/api/datasets/{id}/versions/{version_id}` | 版本详情：元信息 + 质量 + 列画像 + 前 20 行预览 + 完整操作历史（`original` 同样可用） |
| POST | `/api/datasets/{id}/versions/compare` | 版本对比，body `{ "version_a": "original", "version_b": "<uuid>" }` |
| DELETE | `/api/datasets/{id}/versions/{version_id}` | 删除派生版本；original 禁止删除；存在子版本时返回 409 |

compare 返回（A→B 方向）：`rows/columns/missing/duplicates` 的 A/B 两侧值，以及 `generated_columns`（B 独有列）、`removed_columns`（A 独有列）。

```json
{
  "success": true, "dataset_id": "…",
  "version_a": "original", "version_b": "06ea2452-…",
  "rows_a": 23, "columns_a": 7, "missing_a": 7, "duplicates_a": 1,
  "rows_b": 22, "columns_b": 7, "missing_b": 3, "duplicates_b": 0,
  "generated_columns": [], "removed_columns": []
}
```

错误码：`invalid_version_id`（400）、`version_not_found`（404）、`version_has_children`（409）、`original_version_protected`（400）、`version_delete_failed`（500）、`version_storage_error`（500）。

### Transformation Plan 支持的 9 种操作（v0.4）

所有操作都是**结构化 JSON 白名单**；后端按 `type` 走固定逻辑，列名只作为数据索引使用，绝不拼装为代码 / eval / exec。校验失败快速返回结构化 422。

| type | 说明 | 关键参数 | 典型 effect |
| --- | --- | --- | --- |
| `drop_duplicates` | 删除完全重复行（保留首次） | — | `removed_rows` |
| `fill_missing` | 缺失值处理 | `columns[]`、`strategy` | `filled_cells` |
| `drop_columns` | 删除字段（禁止删除全部列） | `columns[]` | `removed_columns` |
| `convert_type` | 类型安全转换 | `column`、`target_type` | `converted_type` |
| `remove_outliers` | IQR 异常值处理 | `column`、`method="IQR"`、`action` | `clipped_cells`/`removed_rows` |
| `text_transform` | 文本确定性变换 | `columns[]`、`action` | `transformed_cells` |
| `date_features` | 日期派生特征（命名冲突即报错） | `column`、`features[]` | `generated_columns` |
| `one_hot_encode` | 分类 One-Hot 编码 | `columns[]`、`allow_high_cardinality` | `generated_columns` |
| `scale_numeric` | 数值缩放 | `columns[]`、`scale_method` | `scaled_columns` |

参数约束：

- `fill_missing.strategy`：`mean` / `median` / `mode` / `constant` / `drop_rows` / `drop_columns`；`constant` 必须提供 `fill_value`。数值列可用 mean/median/constant，分类/文本列建议 mode/constant。
- `drop_columns`：校验会拒绝“删除全部列”；删除后至少保留 1 列。
- `remove_outliers`：`method` 仅 `IQR`；`action` = `clip`（裁剪到 `[Q1-1.5IQR, Q3+1.5IQR]`）或 `remove_rows`。
- `text_transform.action`：`strip` / `lowercase` / `uppercase`（确定性变换，非 LLM）。
- `date_features.features`：`year` / `month` / `day` / `day_of_week` / `quarter`；生成列形如 `<column>_year`，若与已有列冲突返回错误，不静默覆盖。
- `one_hot_encode`：取值基数 > 50 时返回 warning 并要求 `allow_high_cardinality: true` 显式确认（通过独立可读 warning + 参数双重表达）。
- `scale_numeric.scale_method`：`standardization`（z-score）或 `min_max`；列方差为 0（常量列）时跳过并给出 warning，避免除零。
- 针对不同目标类型（numeric/datetime/boolean）的转换失败（缺列、无法解析）均返回带列名与原因的结构化错误。
- > 说明：缩放/标准化只做“当前批次数据的统计变换”。若未来用于模型训练，应在 sklearn Pipeline 中对训练/验证集**分别 fit**，避免数据泄露；该内容留待 v0.5 建模阶段实现。

### 版本化设计约定（v0.4）

- **不可变原始数据**：上传的 `source.csv/xlsx` 永远只读；所有处理只读加载 → 派生新版本。测试中以 `source` 文件字节哈希校验不可变性。
- **原子 Apply**：版本先写入 `versions/.tmp-<uuid>`，成功后 `os.replace` 原子改名为正式目录；失败清理临时目录。
- **派生链**：每个新版本的 `parent_version_id` 记录处理基准；列表/详情都可追溯完整操作历史。
- **删除保护**：original 永不可删；有子版本的版本返回 409 `version_has_children`，必须先删除其子版本。

### 机器学习 API（v0.5）

所有机器学习路由前缀 `/api/datasets/{dataset_id}/ml/...`，统一响应 `{ success, error? }` 形式。
训练管线严格遵循「**先 train/test split，再训练集内 CV 择最优，最后仅在测试集上评估一次**」的防泄漏准则。

#### POST /api/datasets/{dataset_id}/ml/train

请求体（`MLTrainRequest`）：

```json
{
  "source_version_id": null,
  "target_column": "churn",
  "task_type": "auto",
  "feature_columns": null,
  "exclude_columns": [],
  "candidate_models": null,
  "test_size": 0.2,
  "cv_folds": 5,
  "random_state": 42
}
```

- `source_version_id`：`null` 或 `"original"` 表示原始版本；也可传派生版本 UUID。
- `task_type`：`auto` / `classification` / `regression`。
- `feature_columns`：`null` = 自动选择；数组 = 仅使用这些列；不允许与 target / exclude 重叠。
- `candidate_models`：`null` = 按任务类型使用默认候选；可选 `dummy / logistic_regression / ridge / random_forest`。
- `test_size`：`(0, 1)`；`cv_folds`：`[2, 20]`；`random_state`：整数。

成功返回（关键字段）：

```json
{
  "success": true,
  "experiment_id": "58aa0249-…",
  "dataset_id": "…",
  "source_version_id": "original",
  "target_column": "churn",
  "task_type": "classification",
  "features": ["age", "income", "tenure", "region", "plan_type", "is_active"],
  "excluded_features": [],
  "train_rows": 115, "test_rows": 29,
  "best_model": "random_forest",
  "primary_metric": "f1_macro",
  "primary_score": 0.7521,
  "cv": {
    "metric": "f1_macro",
    "results": [
      { "model": "random_forest", "is_best": true, "accuracy": 0.7739, "balanced_accuracy": 0.6833,
        "precision_macro": 0.7819, "recall_macro": 0.6833, "f1_macro": 0.6970,
        "fit_seconds": 0.12, "predict_seconds": 0.01 },
      { "model": "logistic_regression", "f1_macro": 0.6502, … },
      { "model": "dummy", "f1_macro": 0.5022, … }
    ]
  },
  "test": {
    "accuracy": 0.8276, "balanced_accuracy": 0.7262,
    "precision_macro": 0.8167, "recall_macro": 0.7262, "f1_macro": 0.7621,
    "roc_auc": 0.83, "auc_method": "ovr_macro",
    "confusion_matrix": { "labels": ["no", "yes"], "matrix": [[...]] },
    "roc_curve": { "fpr": [0, 0.04, …, 1], "tpr": [0, 0.1, …, 1], "pos_label": "yes" }
  },
  "test_score_summary": { "metric": "f1_macro", "score": 0.7521, "higher_is_better": true },
  "model_errors": [],
  "warnings": ["目标字段 churn 缺失 6 行已自动剔除", "候选 dummy 优于其他模型时请检查数据质量"],
  "created_at": "2026-09-10T02:18:00Z"
}
```

回归任务返回 `test` 中 `rmse / mae / r2 / scatter / residuals / predictions` 字段：
- `scatter`：`{ x_axis: "actual", y_axis: "predicted", points: [{x, y}, ...] }`（最多 `ML_ROC_MAX_POINTS` 默认 500 点）。
- `residuals`：直方图分箱（`bin_edges / counts`）。
- `roc_auc` / `roc_curve` / `confusion_matrix` 在回归任务下为 `null`。

错误码：`dataset_not_found`（404）、`source_version_not_found`（404）、`empty_dataset`（400）、`target_missing`（400）、`target_not_found`（400）、`target_constant`（400）、`no_features`（400）、`invalid_feature_columns`（422）、`unsupported_feature_type`（422）、`cv_folds_out_of_range`（422）、`test_size_out_of_range`（422）、`all_models_failed`（422）、`model_load_failed`（500）。

#### GET /api/datasets/{dataset_id}/ml/experiments

```json
{
  "success": true,
  "dataset_id": "…",
  "count": 2,
  "experiments": [
    {
      "experiment_id": "…", "source_version_id": "original",
      "created_at": "2026-09-10T02:18:00Z",
      "target_column": "churn", "task_type": "classification",
      "best_model": "random_forest",
      "primary_metric": "f1_macro", "primary_score": 0.7521,
      "primary_higher_is_better": true,
      "train_rows": 115, "test_rows": 29,
      "features": ["age", "income", "tenure", "region", "plan_type", "is_active"],
      "warnings": [...]
    }
  ]
}
```

#### GET /api/datasets/{dataset_id}/ml/experiments/{experiment_id}

返回与 `/train` 成功响应相同结构的完整实验详情（含 `cv` / `test` 全部字段）。

#### DELETE /api/datasets/{dataset_id}/ml/experiments/{experiment_id}

删除实验目录（`model.joblib + metadata + evaluation`），返回 `{ success: true }`。错误码：`invalid_experiment_id`（400）、`experiment_not_found`（404）、`experiment_delete_failed`（500）。

#### POST /api/datasets/{dataset_id}/ml/experiments/{experiment_id}/predict

```json
{ "records": [ {"age": 30, "income": 11500, "tenure": 10, "region": "West", "plan_type": "Basic", "is_active": "True"}, … ] }
```

- 每条记录必须是 JSON 对象；多余字段被忽略并写入 `warnings`；
- 缺少训练时的特征字段返回 422 `missing_feature_columns`；
- 最多 `ML_MAX_PREDICTION_RECORDS`（默认 1000）。

成功返回：

```json
{
  "success": true,
  "experiment_id": "…",
  "task_type": "classification",
  "count": 2,
  "predictions": [ { "index": 0, "prediction": "yes" }, { "index": 1, "prediction": "no" } ],
  "probabilities": [ { "no": 0.27, "yes": 0.73 }, { "no": 0.57, "yes": 0.43" } ],
  "class_labels": ["no", "yes"],
  "warnings": ["记录 0 含训练时不存在的字段 \"extra\"，已忽略"]
}
```

回归任务时 `probabilities / class_labels = null`，`predictions[].prediction` 为浮点数。

### 机器学习设计约定（v0.5）

- **不可变原始数据 + Pipeline 整体持久化**：模型推理使用训练时保存的完整 `ColumnTransformer + Estimator` 流水线，预测时无需重新拟合任何预处理。
- **防泄漏数据切分**：先 `train_test_split`（分类用分层），再在训练集内做 KFold 交叉验证，最后**仅在测试集上做唯一一次**最终评估。
- **候选模型**：分类默认 `[dummy, logistic_regression, random_forest]`，回归默认 `[dummy, ridge, random_forest]`；选择范围受前端 checkbox 控制。
- **自动特征推断**：
  - `numeric` 列直接进入数值列；
  - `boolean` 列作为类别列；
  - `categorical` 唯一值 > `ML_MAX_UNIQUE_CLASSES`（默认 50）时直接拒绝任务；
  - 整型 dtype 且 unique_ratio ≥ `ML_ID_LIKE_UNIQUE_RATIO`（默认 0.95）的列被识别为 ID，自动排除并写入 `warnings`；
  - `datetime / text` 类型不参与建模；
  - 用户可通过 `feature_columns` 显式指定（必须与自动选择不冲突），或 `exclude_columns` 额外排除。
- **常量列 / 单值列警告**：在选定特征中遇到 `unique_count == 1` 的列，给出警告并保留（无信息量但允许用户显式决策）。
- **缺失行处理**：目标字段缺失的行直接 drop，记录 `target_missing_dropped`。
- **超限保护**：`ml_max_rows` / `ml_max_features` / `ml_max_encoded_features` / `ml_min_total_rows`；超限返回 422。
- **CV 安全**：当分类任务的最小类样本数 < `cv_folds` 时自动降低 `effective_cv_folds`，避免空折。
- **持久化原子性**：实验先写 `.tmp-<uuid>` 目录，全部成功再 `os.replace`；失败清理临时目录，绝不留半成品。
- **实验删除**：单实验删除立即释放 `model.joblib` + 元信息；与版本同处 Dataset Session，会话删除时一并清理。
- **版本删除保护**：若某派生版本仍被任意实验引用，`DELETE version` 返回 409 `version_in_use_by_ml_experiment`，避免实验指向已删除数据。
- **多任务适配**：分类产出混淆矩阵 + ROC；回归产出 actual/predicted 散点 + 残差直方图 + RMSE/MAE/R²。
- **数值稳定**：sklearn 1.9.0；`roc_curve` 显式传 `pos_label=class_labels[1]`，确保字符串类目标可用。

### 模型可解释性 API（v0.6）

针对已训练完成的 experiment（任意 dataset_id / source_version_id / 模型类型）生成 SHAP 解释。
解释结果同时落盘到 ``runtime/datasets/<dataset_id>/ml/experiments/<experiment_id>/shap_result.json``，
并把 ``status: "computed"`` 写回 experiment 的 ``metadata.json#explainability`` 字段。

提供两个等价入口：

- ``POST /api/ml/experiments/{experiment_id}/explain``（用户指定路径；服务端扫描 ``runtime/datasets`` 定位所属 dataset）
- ``POST /api/datasets/{dataset_id}/ml/experiments/{experiment_id}/explain``（与 v0.5 风格保持一致的 dataset-scoped 入口，并校验归属）

请求体 ``ExplainRequest``（全部可选）：

```json
{
  "sample_indices": [0, 5, 12],
  "max_summary_rows": 200,
  "regenerate": false
}
```

- ``sample_indices``：单点解释的样本索引（相对于 SHAP 采样行，0 ≤ idx < n_rows_used，最多 20 个）。
- ``max_summary_rows``：summary 阶段从源版本采样的行数（1~1000，默认 ``ML_SHAP_SAMPLE_SIZE``）。
- ``regenerate``：默认 False 命中缓存；为 True 时强制重算并覆盖 ``shap_result.json``。

成功返回（关键字段）：

```json
{
  "success": true,
  "experiment_id": "ec0b6826-…",
  "dataset_id": "…",
  "source_version_id": "original",
  "task_type": "classification",
  "best_model": "random_forest",
  "model_class": "RandomForestClassifier",
  "explainer_type": "tree",
  "class_label_used": "yes",
  "class_labels": ["no", "yes"],
  "base_value": 0.5234,
  "n_features_post_preprocessing": 5,
  "feature_names": ["numeric__x1", "numeric__x2", "categorical__grp_Alpha", "categorical__grp_Beta", "categorical__grp_Gamma"],
  "global": {
    "feature_names": [...],
    "importance": [0.123, 0.087, ...],
    "n_rows_used": 120,
    "top_feature_names": [...],
    "top_importance": [...]
  },
  "summary": {
    "feature_names": [...],
    "sample_indices": [3, 7, 14, ...],
    "values": [[0.34, 0.0, 1.0, 0.0, 0.0], ...],
    "raw_values": [[0.34, 8.0, "Beta"], ...],
    "raw_feature_names": ["x1", "x2", "grp"],
    "shap_values": [[0.03, -0.02, 0.05, -0.01, 0.0], ...]
  },
  "samples": [
    {
      "index": 0,
      "sample_source_index": 3,
      "prediction": { "kind": "classification", "class_label": "yes", "probability": 0.73 },
      "base_value_delta": 0.21,
      "contributions": [
        { "feature": "numeric__x1", "value": 0.34, "shap": 0.05 },
        ...
      ],
      "top_contributions": [...]
    }
  ],
  "warnings": [],
  "created_at": "2026-09-10T02:18:00Z",
  "app_version": "0.6.0",  /* v0.7 agent 的 app_version = "0.7.0" */
  "cached": false
}
```

关键约定：
- ``feature_names / summary.values / summary.shap_values / samples[].contributions[].feature`` **都基于 post-preprocessing 矩阵**（含 One-Hot 展开后特征名），三者维数严格一致；
- ``summary.raw_values`` 与 ``summary.raw_feature_names`` 同时给出原始数据值（仅供人读，便于理解原始取值与 SHAP 的对应关系）；
- 对分类任务统一以 ``class_labels[1]``（即「正类」）的 SHAP 为基准，``base_value`` 同样取 ``expected_value[1]``，二分类与多分类走同一条前端路径；
- ``samples`` 默认 3 条（首/中/末），可通过请求体 ``sample_indices`` 自定义；
- 全部 SHAP 数值经过 ``round(…, 6)`` 并保证 JSON safe（NaN / Inf 转为 null）。

错误码：
- ``invalid_experiment_id``（400）
- ``experiment_not_found``（404）
- ``experiment_model_missing``（500）
- ``experiment_dataset_mismatch``（400，dataset-scoped 入口专属）
- ``source_version_unavailable``（422）
- ``experiment_invalid``（500，特征缺失等）
- ``empty_dataset``（422）
- ``shap_computation_failed``（500，SHAP 内部异常）
- ``shap_persist_error``（500，落盘失败）

### 模型可解释性设计约定（v0.6）

- **复用 v0.5 已落盘 Pipeline**：不重新训练 / 不重新 fit 任何预处理；直接 ``joblib.load(model.joblib)`` 并切出 ``pipeline[:-1]`` 作为 preprocessor。
- **三档 Explainer 自动路由**：
  - ``RandomForestClassifier / RandomForestRegressor`` → ``shap.TreeExplainer``（精确，O(树深度)）；
  - ``LogisticRegression / Ridge`` → ``shap.LinearExplainer``（interventional 路径，O(1)）；
  - ``DummyClassifier / DummyRegressor / 其它`` → ``shap.KernelExplainer``（兜底，背景摘要 ``kmeans(ML_SHAP_BACKGROUND_SIZE)``）。
- **特征名优先用 v0.5 已存的 ``feature_names_out``**：避免 One-Hot 之后再次推断；只有当元信息缺失时才回退到 ``ColumnTransformer.get_feature_names_out()``，再不行则回退为 ``feature_i``。
- **分类 SHAP 基准固定为正类**：与 v0.5 的 ROC 曲线（``pos_label=class_labels[1]``）保持一致，前端无需为类别数写分支。
- **3D / 2D / list-of-2D 兼容**：服务统一用 ``_select_class_shap`` 归一化为 ``(n_samples, n_features)``，对 TreeExplainer 新旧格式与 KernelExplainer 的 list 输出一视同仁。
- **采样与缓存**：相同 ``random_state`` 下重新计算可复现；``shap_result.json`` 与 ``metadata.json#explainability`` 同步写回，**重复请求默认命中缓存**，``regenerate=true`` 强制重算。
- **级联删除**：实验删除走 ``shutil.rmtree(experiment_dir)``，``shap_result.json`` 跟随整个目录被删除，**无需任何额外代码**。
- **跨会话定位** ``find_experiment_location``：服务端通过扫描 ``runtime/datasets/*/ml/experiments/<uuid>`` 找到 experiment 所属 dataset，因此前端可以走不带 dataset_id 的简洁路径。
- **磁盘路径遵守约束**：所有 SHAP 产物仅落在 ``D:\ai\AI-Data-Platform\backend\runtime\``，绝不写入 C 盘。

## 自动化测试

```powershell
cd backend
.venv\Scripts\python.exe -m pytest tests -v
```

共 **109 个用例全部通过**（旧有 65 + v0.5 新增 32 + v0.6 新增 12）：

- `tests/test_upload.py`（10）：CSV 成功 / Excel(.xlsx) 成功 / 非法扩展名 / 空文件 / 损坏 CSV / 损坏 Excel / 超限文件 / 健康检查；
- `tests/test_datasets_eda.py`（17）：Dataset Session 上传建会话、落盘、删除 / EDA 结构与数值统计 / 缺失 / IQR 异常 / Pearson 相关 / 分类 TopN / 非法、不存在与过期 dataset_id / 边界数据；
- `tests/test_processing.py`（38，v0.4 新增）：drop_duplicates / 各类 fill_missing（mean/median/mode/constant/drop_rows/drop_columns）/ drop_columns 全部删除防护 / convert_type / remove_outliers（clip 与 remove_rows）/ text_transform / date_features（含命名冲突）/ one_hot_encode（含高基数确认与拒绝）/ scale_numeric（标准化、min-max、常量列跳过）/ **preview 不写盘 / apply 原子性（失败不落盘、成功生成 2 个版本）/ original 字节哈希不变 / 非法 version_id 与 404 / 子版本 409 / 版本感知 EDA 等**；
- `tests/test_ml.py`（32，v0.5 新增）：健康检查 + 训练（auto 推断 classification / regression / 显式 task_type 校验 / 目标缺失 drop / 常量目标拒绝 / 特征全集显式 / exclude / 高基数 categorical 拒绝 / 唯一值过少拒绝 / ID-like 整型高基数自动排除 / float 连续特征保留 / 候选模型自定义 / 测试集含训练集目标值（已切分干净）/ test_size 越界 / cv_folds 越界 / 全部候选失败清理 / 多类分类报告 / 显式 task + auto 推断一致）+ 列表 / 详情 / 删除 / 预测（多记录 + 类别概率 / 缺特征 422 / 多余字段警告 / 回归无概率 / 实验不存在 / 非法 experiment_id）+ 派生版本引用保护 409 / 删除实验后再删版本成功 / 整型/类别混合数据集 / 数据集不存在 404 / 上传后整 session 生命周期。
- `tests/test_explainability.py`（12，v0.6 新增）：experiment 不存在 404 / 非法 experiment_id 400 / 模型文件缺失 500 / **classification TreeExplainer** / **regression TreeExplainer** / JSON schema 完整性 / 落盘路径与 metadata 写回 / 删除实验级联清理 / **Dummy KernelExplainer 兜底**（SHAP ≈ 0）/ dataset-scoped 入口 + 跨 dataset 校验 / 缓存命中与 regenerate 行为 / ExplainRequest 字段校验。

## 手动测试

```powershell
# PowerShell（简单）
Invoke-RestMethod http://127.0.0.1:8000/api/health

# curl 上传示例文件（响应中的 dataset_id 供后续使用）
curl.exe -F "file=@data/sample/demo.csv" http://127.0.0.1:8000/api/data/upload

# 对上传的会话执行自动 EDA（替换为上面返回的 dataset_id）
curl.exe http://127.0.0.1:8000/api/datasets/<dataset_id>/eda

# Preview 处理（drop_duplicates + fill_missing median）
curl.exe -X POST http://127.0.0.1:8000/api/datasets/<dataset_id>/processing/preview `
  -H "Content-Type: application/json" `
  -d '{\"source_version_id\": null, \"operations\": [ {\"type\": \"drop_duplicates\"}, {\"type\": \"fill_missing\", \"columns\": [\"score\"], \"strategy\": \"median\"} ]}'

# Apply 生成新版本（返回 version_id）
curl.exe -X POST http://127.0.0.1:8000/api/datasets/<dataset_id>/processing/apply `
  -H "Content-Type: application/json" `
  -d '{\"operations\": [ {\"type\": \"drop_duplicates\"}, {\"type\": \"fill_missing\", \"columns\": [\"score\"], \"strategy\": \"median\"} ]}'

# 版本列表 / 对派生版本执行 EDA / 版本对比 / 删除派生版本
curl.exe http://127.0.0.1:8000/api/datasets/<dataset_id>/versions
curl.exe "http://127.0.0.1:8000/api/datasets/<dataset_id>/eda?version_id=<uuid>"
curl.exe -X POST http://127.0.0.1:8000/api/datasets/<dataset_id>/versions/compare `
  -H "Content-Type: application/json" -d '{\"version_a\": \"original\", \"version_b\": \"<uuid>\"}'
curl.exe -X DELETE http://127.0.0.1:8000/api/datasets/<dataset_id>/versions/<uuid>

# v0.5 机器学习：训练（auto 推断，目标=churn）
curl.exe -X POST http://127.0.0.1:8000/api/datasets/<dataset_id>/ml/train `
  -H "Content-Type: application/json" `
  -d '{\"target_column\": \"churn\", \"task_type\": \"auto\"}'

# v0.5 实验列表 / 详情 / 预测 / 删除
curl.exe http://127.0.0.1:8000/api/datasets/<dataset_id>/ml/experiments
curl.exe http://127.0.0.1:8000/api/datasets/<dataset_id>/ml/experiments/<experiment_id>
curl.exe -X POST http://127.0.0.1:8000/api/datasets/<dataset_id>/ml/experiments/<experiment_id>/predict `
  -H "Content-Type: application/json" `
  -d '{\"records\": [{\"age\":30,\"income\":11500,\"tenure\":10,\"region\":\"West\",\"plan_type\":\"Basic\",\"is_active\":true}]}'
curl.exe -X DELETE http://127.0.0.1:8000/api/datasets/<dataset_id>/ml/experiments/<experiment_id>

# v0.6 模型可解释性：生成 SHAP 解释（简化路径：仅 experiment_id）
curl.exe -X POST http://127.0.0.1:8000/api/ml/experiments/<experiment_id>/explain `
  -H "Content-Type: application/json" `
  -d '{\"max_summary_rows\": 200, \"sample_indices\": [0, 5, 12]}'

# v0.6 强制重算（覆盖缓存）
curl.exe -X POST http://127.0.0.1:8000/api/ml/experiments/<experiment_id>/explain `
  -H "Content-Type: application/json" -d '{\"regenerate\": true}'

# v0.6 dataset-scoped 入口（额外校验归属）
curl.exe -X POST http://127.0.0.1:8000/api/datasets/<dataset_id>/ml/experiments/<experiment_id>/explain
```

或直接用浏览器打开 <http://localhost:5173> 拖拽上传，体验「**上传 → EDA → 数据清洗 → 版本化 → 指定版本 EDA / 版本对比 → 机器学习训练（CV 对比 / 混淆矩阵 / ROC / 散点残差）→ 实验历史 → 批量预测 → 模型可解释性（SHAP 全局 / Summary / 单点）→ 清理**」的完整界面流程。

## v0.5 机器学习 · 44 点逐项总结

#### 后端 API（5）
1. `POST /api/datasets/{id}/ml/train`：训练并保存实验。
2. `GET  /api/datasets/{id}/ml/experiments`：实验列表（按 `created_at` 倒序）。
3. `GET  /api/datasets/{id}/ml/experiments/{exp_id}`：实验详情（`cv` / `test` 完整）。
4. `DELETE /api/datasets/{id}/ml/experiments/{exp_id}`：删除实验 + `model.joblib`。
5. `POST /api/datasets/{id}/ml/experiments/{exp_id}/predict`：批量预测（分类含概率）。

#### 任务与候选模型（5）
6. `task_type` 支持 `auto` / `classification` / `regression`。
7. 分类默认候选：`dummy` / `logistic_regression` / `random_forest`。
8. 回归默认候选：`dummy` / `ridge` / `random_forest`。
9. 前端 checkbox 可自定义候选集合；空数组 = 后端按任务用默认集。
10. 主指标：分类用 `f1_macro`，回归用 `rmse`（越低越好）。

#### 特征工程与防护（8）
11. 列画像 → 自动特征选择（数值/类别/布尔/排除）。
12. 整型 dtype 且 `unique_ratio ≥ 0.95` 的列视为 ID 自动排除。
13. `categorical` 唯一值 > `ML_MAX_UNIQUE_CLASSES`（默认 50）直接拒绝任务。
14. 目标字段缺失行自动 drop 并写 `warnings`。
15. 常量目标（`unique_count == 1`）拒绝训练。
16. 显式 `feature_columns` / `exclude_columns` 支持，与自动选择校验不冲突。
17. 高级参数 `test_size / cv_folds / random_state` 全部入参 + 范围校验。
18. `ML_MAX_ROWS / ML_MAX_FEATURES / ML_MIN_TOTAL_ROWS` 等配置项做硬上限保护。

#### 训练管线（8）
19. 防泄漏：先 `train_test_split`（分类分层），再训练集内 KFold CV，最后**仅**在测试集做唯一评估。
20. `effective_cv_folds` 自动下调以满足最小类样本数。
21. 分类：`StratifiedKFold`；回归：`KFold`；每次循环重建 `cv.split(...)` 避免生成器耗尽。
22. 分类指标：`accuracy / balanced_accuracy / precision(macro) / recall(macro) / f1(macro)`。
23. 回归指标：`rmse / mae / r2 + 散点 + 残差直方图`。
24. `Pipeline(ColumnTransformer → Estimator)` 整体 `joblib.dump`。
25. 实验目录：`<session>/ml/experiments/<exp_id>/{model.joblib, metadata.json, evaluation.json}`。
26. 持久化原子：`.tmp-<uuid>` → `os.replace`；失败清理临时目录。

#### 测试评估（4）
27. 分类额外产出 `roc_auc` / `confusion_matrix` / `roc_curve`。
28. `roc_curve` 显式 `pos_label=class_labels[1]`，兼容字符串目标。
29. 散点最多 `ML_SCATTER_MAX_POINTS`（默认 500）点采样。
30. 所有候选模型失败 → 422 `all_models_failed` 并清理已创建版本。

#### 存储与引用（4）
31. 实验元信息含 `source_version_id` / 目标 / 特征集 / 警告 / `random_state`。
32. 删除派生版本前 `find_experiments_using_version()` 扫描引用。
33. 若仍被实验引用 → 409 `version_in_use_by_ml_experiment`。
34. 删除实验后即可正常删除被引用版本（解除保护）。

#### 预测（5）
35. 多记录（≤ `ML_MAX_PREDICTION_RECORDS` 默认 1000）。
36. 缺特征字段 422 `missing_feature_columns`（列出缺失列）。
37. 多余字段 → `warnings` 提示，**不报错**。
38. 分类返回 `predictions` + `probabilities`（每个类别概率）。
39. 回归返回 `predictions`（浮点数），`probabilities` 为 `null`。

#### 前端 ML 面板（5）
40. `MLPanel.jsx` 训练配置（数据版本 / 目标 / 任务类型 / 特征 / 候选 / 高级参数）。
41. `ExperimentReport.jsx` 报告：CV 对比表（最佳高亮）+ 混淆矩阵热力图 + ROC 曲线 + 回归散点 + 残差直方图 + 特征信息。
42. 实验历史侧栏：行（最佳/警告/任务标识）+ 删除带 `confirm` 对话框。
43. 预测编辑器：JSON 文本 + 「从源版本载入示例」+ 预测结果表（含概率条）。
44. 端到端真实浏览器 E2E 验证：上传 → 训练 → 预测 → 删除 → 清理全部通过。

#### 自动化测试（与上述对照）
- `tests/test_ml.py` 32 用例：覆盖上述 API 契约、特征工程、CV 切分、Pipeline 持久化、预测、版本引用保护等关键路径；与 v0.4 合计 97 个测试全部通过。

## v0.6 模型可解释性（SHAP）· 44 点逐项总结

#### 后端 API（4）
1. `POST /api/ml/experiments/{exp_id}/explain`：用户指定的简洁路径，跨会话定位 experiment。
2. `POST /api/datasets/{dataset_id}/ml/experiments/{exp_id}/explain`：与 v0.5 风格一致的 dataset-scoped 同义入口，额外校验归属。
3. 统一响应结构：`{success, error?}`；错误码含 `invalid_experiment_id / experiment_not_found / experiment_model_missing / experiment_dataset_mismatch / source_version_unavailable / experiment_invalid / empty_dataset / shap_computation_failed / shap_persist_error`。
4. `ExplainRequest` Pydantic v2 模型：`sample_indices`（≤20）/ `max_summary_rows`（1~1000）/ `regenerate`（bool）。

#### Explainer 路由（5）
5. `RandomForestClassifier` → `shap.TreeExplainer`。
6. `RandomForestRegressor` → `shap.TreeExplainer`。
7. `LogisticRegression` → `shap.LinearExplainer`（interventional）。
8. `Ridge` → `shap.LinearExplainer`。
9. `DummyClassifier` / `DummyRegressor` / 其它 → `shap.KernelExplainer`（kmeans 摘要背景）。

#### Pipeline 复用与特征名（6）
10. 复用 v0.5 已落盘的 `model.joblib`（`Pipeline(preprocessing → model)`），不重新训练 / 不重 fit 预处理。
11. 通过 `pipeline[:-1]` 切出 preprocessor 子 Pipeline。
12. 特征名优先用 v0.5 `metadata["feature_names_out"]`，保持 One-Hot 展开后名称（如 `numeric__x1` / `categorical__grp_Alpha`）。
13. 元信息缺失时回退到 `ColumnTransformer.get_feature_names_out()`，再不行回退为 `feature_i`。
14. `feature_names / summary.values / summary.shap_values / samples[].contributions[].feature` 全部 post-preprocessing 同维对齐。
15. `summary.raw_values + summary.raw_feature_names` 同时返回原始数据值，便于人读。

#### 分类 / 回归归一化（4）
16. 分类任务统一以 `class_labels[1]`（正类）作为 SHAP 基准，与 v0.5 ROC（`pos_label=class_labels[1]`）保持一致。
17. 二分类与多分类走同一条前端路径（无需为类别数写分支）。
18. 回归任务 `class_label_used=null` / `prediction.kind=regression` / `prediction.value` 为浮点。
19. `_select_class_shap` 统一处理 2D / 3D / list-of-2D 三种 SHAP 输出格式。

#### 采样与缓存（6）
20. 训练阶段保存的 `random_state` 用于 SHAP 行采样，保证可复现。
21. `max_summary_rows` 由 `ExplainRequest` 决定，默认 `ML_SHAP_SAMPLE_SIZE=200`。
22. `summary.sample_indices` 记录每行对应源版本的位置。
23. 重复请求默认命中 `shap_result.json` 缓存（`cached=true`）。
24. `regenerate=true` 强制重算并覆盖缓存。
25. `cache` 命中时仍写回最新 `created_at`，保持前端显示稳定。

#### 落盘与级联（5）
26. 落盘到 `runtime/datasets/<id>/ml/experiments/<exp>/shap_result.json`（D 盘 runtime）。
27. 落盘原子性：先写 `.tmp` 再 `os.replace`；失败清理临时文件。
28. `metadata.json#explainability` 同步写回 `status / result_path / explainer_type / model_class / task_type / class_label_used / n_rows_used / warnings / created_at`。
29. `experiment_manager.delete_experiment` 用 `shutil.rmtree(experiment_dir)` 删除整个目录，**shap_result.json 跟随实验目录被级联删除**。
30. `find_experiments_using_version` 链路不变；SHAP 不会改变 v0.5 版本引用关系。

#### 跨会话定位与校验（4）
31. `find_experiment_location(experiment_id)` 扫描 `runtime/datasets/*/ml/experiments/<uuid>/`，返回 `(dataset_id, exp_dir)`。
32. 只接受合法 UUID 格式的目录名，避免误判临时文件。
33. dataset-scoped 入口对比 `found_dataset_id == dataset_id`，不一致返回 400 `experiment_dataset_mismatch`。
34. 找不到 experiment → 404 `experiment_not_found`（同时在缓存命中检查之前）。

#### 错误防护（4）
35. 模型文件缺失 → 500 `experiment_model_missing`。
36. 源版本不可用（已被删除或会话过期） → 422 `source_version_unavailable`。
37. 训练特征在源版本中缺失 → 500 `experiment_invalid`。
38. SHAP 内部异常 → 500 `shap_computation_failed`（含原始 traceback 给后端日志）。

#### 前端（5）
39. `ExplainabilityPanel.jsx`：实验下拉 / 生成 / 缓存加载 / 强制重算 / 错误提示 / 配置项（采样行数 + 样本索引）。
40. `FeatureImportanceChart.jsx`：横向条形图（按 `mean |SHAP|` 降序，分类正类标识）。
41. `SHAPSummaryChart.jsx`：Beeswarm 散点（x=SHAP，y=feature，颜色按原始值映射）。
42. `SampleExplanation.jsx`：默认 3 条样本卡片，每条含预测、base_value_delta、top-K 贡献条形图。
43. 切换实验自动重置 SHAP 状态（避免串数据）。
44. ECharts 与 `chartTheme` 主题保持一致；样式变量使用现有 `--color-*` / `rgba(125,157,255,*)` 调色板。

#### 自动化测试（与上述对照）
- `tests/test_explainability.py` 12 用例：覆盖 API 404/400/500、分类/回归 TreeExplainer、KernelExplainer 兜底、JSON schema 完整性、落盘路径、metadata 写回、级联删除、缓存命中、regenerate 行为、ExplainRequest 字段校验、dataset-scoped 校验；与 v0.5 合计 109 个测试全部通过。

## 数据处理流程（v0.4）

1. 路由层（`api/data.py`）：读取 multipart 文件（分块读取并限制 20MB）→ 只保留文件名
2. 校验与解析层（`services/data_service.py`）：空文件 / 扩展名 / 大小 → pandas 解析（CSV 自动尝试 UTF-8/GB18030 编码，Excel 按格式选择 openpyxl/xlrd）
3. 画像层：逐列画像（类型推断、缺失、唯一值）→ 汇总质量概览 → 生成 warnings → JSON 安全的前 20 行预览
4. 会话层（`services/dataset_manager.py`）：创建临时 Dataset Session（UUID），将源文件与 metadata 落盘到 `runtime/datasets/<dataset_id>/`
5. EDA 层（`services/eda_service.py`）：`GET /eda?version_id=` 时通过 `version_manager.load_dataset_version` 统一加载（original 读 source，派生版本读 versions/<uuid>/data.csv），计算全部统计并以 JSON 返回
6. 处理层（`services/processing_service.py`）：解析校验 Transformation Plan → 逐操作在内存执行向量化变换（缺列/类型失败立即中止并返回清晰错误）→ preview 只返回结果；apply 在全部成功后由 `version_manager.persist_new_version` 原子写入新 UUID 版本
7. 版本管理（`services/version_manager.py`）：列表 / 详情 / 对比 / 删除；original 只读保护、UUID 校验防路径穿越、子版本删除 409
8. 响应模型（`models/data.py` / `models/eda.py` / `models/processing.py`）保证 JSON 结构稳定；任何异常都以结构化 `{ success, error }` 返回，不泄露 traceback
9. 清理：会话默认 2 小时过期，惰性清理过期目录；`DELETE /api/datasets/{id}` 连同派生版本一并清理

**设计约定：原始数据只读不可变；所有清洗/特征工程都产生新派生版本；Preview 绝不落盘；Apply 全成或全不成（无半成品）；图表由前端 ECharts 渲染，后端不生成图片。**

## 环境变量

复制 `.env.example` 为 `.env`（项目根目录）后按需修改：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `BACKEND_HOST` | `0.0.0.0` | 后端监听地址 |
| `BACKEND_PORT` | `8000` | 后端端口 |
| `BACKEND_CORS_ORIGINS` | `http://localhost:5173` | 允许跨域来源 |
| `DATA_MAX_UPLOAD_MB` | `20` | 上传大小上限（MB） |
| `DATASET_SESSION_TTL_MINUTES` | `120` | Dataset Session 有效期（分钟） |
| `DATASET_RUNTIME_DIR` | `backend/runtime/datasets` | 会话数据存放目录 |
| `VITE_PROXY_TARGET` | `http://127.0.0.1:8000` | Vite `/api` 代理目标 |
| `ML_MAX_ROWS` | `50000` | 单次训练最大行数 |
| `ML_MAX_FEATURES` | `200` | 训练允许的最大特征数 |
| `ML_MAX_ENCODED_FEATURES` | `500` | one-hot 后总特征数上限 |
| `ML_MIN_TOTAL_ROWS` | `20` | 训练最少行数 |
| `ML_MAX_UNIQUE_CLASSES` | `50` | 目标/类别列唯一值上限（超过拒绝） |
| `ML_MAX_PREDICTION_RECORDS` | `1000` | 单次预测最大记录数 |
| `ML_DEFAULT_TEST_SIZE` | `0.2` | 默认 train/test 切分比例 |
| `ML_DEFAULT_CV_FOLDS` | `5` | 默认交叉验证折数 |
| `ML_RANDOM_STATE` | `42` | 默认随机种子 |
| `ML_ROC_MAX_POINTS` | `500` | ROC 曲线最大点数 |
| `ML_SCATTER_MAX_POINTS` | `500` | 回归散点最大点数 |
| `ML_ID_LIKE_UNIQUE_RATIO` | `0.95` | 整型列 unique_ratio 阈值，高于此值视为 ID 自动排除 |
| `ML_SHAP_SAMPLE_SIZE` | `200` | SHAP 解释默认从源版本采样的行数 |
| `ML_SHAP_BACKGROUND_SIZE` | `50` | KernelExplainer 背景摘要（kmeans 中心数） |
| `ML_SHAP_TOP_FEATURES` | `30` | 全局特征重要性 topK 保留数量 |
| `ML_SHAP_TIMEOUT_SECONDS` | `120` | SHAP 计算耗时告警阈值 |

> 提示：代理目标建议使用 `127.0.0.1`，避免 `localhost` 被解析为 IPv6(`::1`) 而后端仅监听 IPv4 导致 502。

## 路线图

- [x] 项目基础架构与健康检查（v0.1.0）
- [x] 数据上传、解析、基础画像、质量概览、数据预览（v0.2.0）
- [x] Dataset Session 与会话管理（v0.3.0）
- [x] EDA 探索性数据分析（数值/分类/缺失/异常/相关性，v0.3.0）
- [x] 数据清洗与特征工程：9 种结构化变换 + 规则校验（v0.4.0）
- [x] 数据版本管理：UUID 派生版本 / 历史 / 对比 / 删除 / 指定版本 EDA（v0.4.0）
- [x] 机器学习建模：防泄漏训练管线 / CV 对比 / 实验管理 / 批量预测（v0.5.0）
- [x] 模型可解释性：SHAP（Tree/Linear/Kernel 路由）/ 全局 / Summary / 单点（v0.6.0）
- [x] **AI Data Analyst Agent：5 个 Tool + 关键词路由 + LLM/Mock 自动切换 + 真实 LLM 失败降级（v0.7.0）**
- [ ] Skill / MCP 扩展
- [ ] Agent 多轮规划 / 多模态
