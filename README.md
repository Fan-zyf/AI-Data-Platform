# AI Data Intelligence Platform

智能数据分析与预测平台（AI Data Intelligence Platform）

一个可长期扩展的 AI 数据智能分析平台。

**当前进度（v0.4.0）**
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
- ⬜ 后续规划：机器学习建模、模型解释（SHAP）、LLM 分析、Skill / MCP 扩展

## 技术栈

| 层 | 技术 |
| --- | --- |
| 前端 | React + Vite + ECharts |
| 后端 | Python 3.14 + FastAPI |
| 数据处理 | pandas、numpy、openpyxl、xlrd |
| 后续扩展 | scikit-learn、XGBoost、SHAP、LLM、Skill、MCP |
| 版本管理 | Git |

> v0.4 新增 Python 依赖：**无**（数据清洗/特征工程全部用已有 pandas/numpy 实现）。

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
│   │   │   └── processing.py  # v0.4：preview / apply / versions / compare / delete-version
│   │   ├── core/config.py     # 配置（端口、CORS、上传限制等）
│   │   ├── services/
│   │   │   ├── data_service.py         # 文件解析 / 数据画像 / 质量分析
│   │   │   ├── dataset_manager.py      # Dataset Session 的创建 / 读取 / 过期清理 / 删除
│   │   │   ├── eda_service.py          # v0.4：run_eda(dataset_id, version_id=None) 版本感知
│   │   │   ├── processing_service.py   # v0.4：预览 / 应用 9 种变换操作
│   │   │   └── version_manager.py      # v0.4：UUID 版本目录 / 加载 / 列表 / 对比 / 删除
│   │   └── models/
│   │       ├── data.py        # 上传响应的 Pydantic 模型
│   │       ├── eda.py         # 自动 EDA 响应的 Pydantic 模型
│   │       └── processing.py  # v0.4：TransformationPlan / Operation / 版本相关响应模型
│   ├── runtime/               # 运行时临时数据（会话目录 + 派生版本，自动清理，勿提交）
│   ├── tests/                 # pytest 自动化测试（65 个用例）
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
│   │   └── processing/        # v0.4 数据处理面板（位于上传/EDA 之后）
│   │       ├── DataProcessingPanel.jsx   # 总控制器：版本/步骤/预览/应用/历史/对比
│   │       ├── opDefinitions.js          # 前后端共享的操作 schema 与 step<->JSON 转换
│   │       ├── TransformationBuilder.jsx # 步骤卡片构建器（加/排/删/参数校验）
│   │       ├── PreviewTabs（内置）        # 处理前/后数据与指标对比
│   │       ├── VersionHistory.jsx        # 版本历史（切换源/EDA/删除）
│   │       ├── VersionCompare.jsx        # 版本 A/B 对比
│   │       └── DataTable.jsx             # 通用数据表（null 渲染/行数截断）
│   ├── index.html
│   └── vite.config.js         # /api 代理到后端
├── .env.example
├── .gitignore
└── README.md
```

### 运行时文件布局（v0.4 版本化）

```
backend/runtime/datasets/<dataset_id>/
├── source.<ext>            # 原始文件（永远保留，绝不修改）
├── metadata.json           # Dataset Session 元信息（v0.3 结构保持不变）
└── versions/
    └── <version_id>/       # version_id = 服务端生成的合法 UUID
        ├── data.csv        # 该版本的清洗/特征数据
        └── metadata.json   # 版本元信息（父版本 / operations / quality / created_at …）
```

版本模型为单向 **parent / child 链**：`original → v1 → v2 …`。
- original 是只读基线，不落盘为版本目录，删除接口明确禁止；
- version_id 一律校验为合法 UUID，路径由服务端拼接，防止路径穿越；
- 派生版本每次独立目录保存，绝不覆盖其他版本。

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

## 自动化测试

```powershell
cd backend
.venv\Scripts\python.exe -m pytest tests -v
```

共 **65 个用例全部通过**（旧有 27 + v0.4 新增 38）：

- `tests/test_upload.py`（10）：CSV 成功 / Excel(.xlsx) 成功 / 非法扩展名 / 空文件 / 损坏 CSV / 损坏 Excel / 超限文件 / 健康检查；
- `tests/test_datasets_eda.py`（17）：Dataset Session 上传建会话、落盘、删除 / EDA 结构与数值统计 / 缺失 / IQR 异常 / Pearson 相关 / 分类 TopN / 非法、不存在与过期 dataset_id / 边界数据；
- `tests/test_processing.py`（38，v0.4 新增）：drop_duplicates / 各类 fill_missing（mean/median/mode/constant/drop_rows/drop_columns）/ drop_columns 全部删除防护 / convert_type / remove_outliers（clip 与 remove_rows）/ text_transform / date_features（含命名冲突）/ one_hot_encode（含高基数确认与拒绝）/ scale_numeric（标准化、min-max、常量列跳过）/ **preview 不写盘 / apply 原子性（失败不落盘、成功生成 2 个版本）/ original 字节哈希不变 / 非法 version_id 与 404 / 子版本 409 / 版本感知 EDA 等**。

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
```

或直接用浏览器打开 <http://localhost:5173> 拖拽上传，体验「上传 → EDA → 数据清洗 → 版本化 → 指定版本 EDA / 版本对比」的完整界面流程。

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

> 提示：代理目标建议使用 `127.0.0.1`，避免 `localhost` 被解析为 IPv6(`::1`) 而后端仅监听 IPv4 导致 502。

## 路线图

- [x] 项目基础架构与健康检查（v0.1.0）
- [x] 数据上传、解析、基础画像、质量概览、数据预览（v0.2.0）
- [x] Dataset Session 与会话管理（v0.3.0）
- [x] EDA 探索性数据分析（数值/分类/缺失/异常/相关性，v0.3.0）
- [x] 数据清洗与特征工程：9 种结构化变换 + 规则校验（v0.4.0）
- [x] 数据版本管理：UUID 派生版本 / 历史 / 对比 / 删除 / 指定版本 EDA（v0.4.0）
- [ ] 机器学习建模
- [ ] 模型解释（SHAP）
- [ ] LLM 智能分析
- [ ] Skill / MCP 扩展
