# AI Data Intelligence Platform

智能数据分析与预测平台（AI Data Intelligence Platform）

一个可长期扩展的 AI 数据智能分析平台。

**当前进度（v0.2.0）**
- ✅ 基础架构：React + Vite 前端 / FastAPI 后端 / Git
- ✅ CSV、Excel(.xlsx/.xls) 数据上传与解析（只读分析，不修改原始数据）
- ✅ 基础数据画像：字段类型推断、缺失统计、唯一值、前 20 行预览、质量警告
- ⬜ 后续规划：数据质量检测、EDA、特征工程、机器学习建模、模型解释（SHAP）、LLM 分析、Skill / MCP 扩展

## 技术栈

| 层 | 技术 |
| --- | --- |
| 前端 | React + Vite |
| 后端 | Python 3.14 + FastAPI |
| 数据处理 | pandas、numpy、openpyxl、xlrd |
| 后续扩展 | scikit-learn、XGBoost、SHAP、LLM、Skill、MCP |
| 版本管理 | Git |

## 目录结构

```
AI-Data-Platform/
├── backend/
│   ├── app/
│   │   ├── main.py            # 应用入口（注册 /api 路由）
│   │   ├── api/
│   │   │   ├── health.py      # GET /api/health
│   │   │   └── data.py        # POST /api/data/upload
│   │   ├── core/config.py     # 配置（端口、CORS、上传限制等）
│   │   ├── services/
│   │   │   └── data_service.py# 文件解析 / 数据画像 / 质量分析
│   │   └── models/
│   │       └── data.py        # 上传响应的 Pydantic 模型
│   ├── tests/                 # pytest 自动化测试
│   └── requirements.txt
├── data/
│   └── sample/demo.csv        # 示例数据（含数值/分类/文本/日期/布尔列、缺失与重复）
├── frontend/
│   ├── src/
│   │   ├── App.jsx            # 首页（系统状态 + 数据上传区域）
│   │   ├── upload/UploadPanel.jsx  # 上传与结果展示组件
│   │   └── ...
│   ├── index.html
│   └── vite.config.js         # /api 代理到后端
├── .env.example
├── .gitignore
└── README.md
```

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

浏览器访问 <http://localhost:5173>。页面会自动请求 `/api/health` 并显示后端连接状态；下方的“数据上传”区域支持点击或拖拽上传 `.csv / .xlsx / .xls` 文件，上传后展示数据集概览、字段信息表、质量警告与前 20 行预览。

## API 说明

### GET /api/health

```json
{ "status": "ok", "message": "AI Data Platform backend is running" }
```

### POST /api/data/upload

multipart 表单字段 `file`。仅支持 `.csv / .xlsx / .xls`，最大 20MB（可通过环境变量 `DATA_MAX_UPLOAD_MB` 调整）。

成功返回结构：

```json
{
  "success": true,
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

## 自动化测试

```powershell
cd backend
.venv\Scripts\python.exe -m pytest tests -v
```

覆盖：CSV 成功 / Excel(.xlsx) 成功 / 非法扩展名 / 空文件 / 损坏 CSV / 损坏 Excel / 超限文件 / 健康检查。

## 手动测试上传

```powershell
# PowerShell（简单）
Invoke-RestMethod http://127.0.0.1:8000/api/health

# curl 上传示例文件
curl.exe -F "file=@data/sample/demo.csv" http://127.0.0.1:8000/api/data/upload

# 或直接用浏览器打开 http://localhost:5173 拖拽上传
```

## 数据上传处理流程

1. 路由层（`api/data.py`）：读取 multipart 文件（分块读取并限制 20MB）→ 只保留文件名
2. 校验层（`services/data_service.py`）：空文件 / 扩展名 / 大小 → pandas 解析（CSV 自动尝试 UTF-8/GB18030 编码，Excel 按格式选择 openpyxl/xlrd）
3. 分析层：逐列画像（类型推断、缺失、唯一值）→ 汇总质量概览 → 生成用户可读 warnings → 生成 JSON 安全的前 20 行预览
4. 响应模型（`models/data.py`）保证 JSON 结构稳定；任何异常都以结构化 `{ success, error }` 返回，不泄露 traceback

**设计约定：所有流程均为只读分析，不修改、清洗或覆盖用户原始数据。**

## 环境变量

复制 `.env.example` 为 `.env`（项目根目录）后按需修改：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `BACKEND_HOST` | `0.0.0.0` | 后端监听地址 |
| `BACKEND_PORT` | `8000` | 后端端口 |
| `BACKEND_CORS_ORIGINS` | `http://localhost:5173` | 允许跨域来源 |
| `DATA_MAX_UPLOAD_MB` | `20` | 上传大小上限（MB） |
| `VITE_PROXY_TARGET` | `http://127.0.0.1:8000` | Vite `/api` 代理目标 |

> 提示：代理目标建议使用 `127.0.0.1`，避免 `localhost` 被解析为 IPv6(`::1`) 而后端仅监听 IPv4 导致 502。

## 路线图

- [x] 项目基础架构与健康检查（v0.1.0）
- [x] 数据上传、解析、基础画像、质量概览、数据预览（v0.2.0）
- [ ] 数据质量检测
- [ ] EDA 探索性数据分析
- [ ] 特征工程
- [ ] 机器学习建模
- [ ] 模型解释（SHAP）
- [ ] LLM 智能分析
- [ ] Skill / MCP 扩展
