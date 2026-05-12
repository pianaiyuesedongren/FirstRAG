# Cheer - 多智能体助手

基于 **LangGraph** 构建的多智能体对话系统。用户提出问题后，由 Supervisor Agent 自动路由到对应的专业子 Agent 处理，并返回结果。

## 功能

| 功能 | 说明 |
|------|------|
| 🗺️ **路线规划** | 调用高德地图 MCP 服务，生成旅游路线规划 |
| 😂 **讲笑话** | 由大语言模型生成幽默笑话 |
| 🪄 **对对联** | 结合 RAG 向量检索（ChromaDB + DashScope Embedding），从 10000 条对联数据中检索参考样本，生成工整的下联 |
| 💬 **随便聊聊** | 兜底回复，处理其他类型问题 |

## 架构

```
用户输入
   │
   ▼
Supervisor Agent ──→ Travel Agent (高德地图 MCP)
   │                  Joke Agent (LLM 直接生成)
   │                  Couplet Agent (RAG + ChromaDB 检索增强)
   │                  Other Agent (兜底)
   │
   ▼
   返回结果
```

- **LangGraph** 驱动多智能体工作流，支持状态管理与条件路由
- **Supervisor Node** 基于 LLM 对用户问题进行分类
- 每个 Agent 执行完毕后再次回到 Supervisor，支持多轮处理

## 技术栈

- **LangGraph / LangChain** — 智能体编排与状态管理
- **ChatOpenAI (DashScope)** — 底层大语言模型（qwen3.5-flash）
- **Gradio** — Web 交互界面
- **ChromaDB + DashScope Embedding** — 对联 RAG 向量库
- **高德地图 MCP Server** — 路线规划工具

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> 如项目未提供 `requirements.txt`，核心依赖包括：`langchain`, `langchain-chroma`, `langchain-openai`, `langgraph`, `dashscope`, `gradio`, `langchain-mcp-adapters`

### 2. 配置 API 密钥

编辑 `config/Keys.json`：

```json
{
  "BAILIAN_API_KEY": "your-dashscope-api-key",
  "AMAP_MAPS_API_KEY": "your-amap-api-key"
}
```

- `BAILIAN_API_KEY`：阿里云百炼（DashScope）的 API Key，用于调用大模型和嵌入模型
- `AMAP_MAPS_API_KEY`：高德地图 API Key，用于路线规划功能

### 3. 运行

```bash
cd Cheer
python app.py
```

访问 `http://127.0.0.1:7860` 即可打开 Web 界面。

### 首次运行说明

- 首次使用对联功能时，系统会自动读取 `resource/coupletData_top10000.csv` 中的 10000 条对联数据，调用 DashScope 嵌入模型构建 Chroma 向量索引
- 默认首次只写入 100 条作为引导（可通过环境变量 `COUPLET_BOOTSTRAP_LIMIT` 调整），全量建库可手动改为 10000
- 向量库会持久化到 `chroma_couplet_db_*/` 目录

## 项目结构

```
Cheer/
├── app.py                    # Gradio Web 界面入口
├── Test_in_local.py          # LangGraph 工作流定义（所有 Agent 节点）
├── couplet_load.py           # 对联 RAG 模块（嵌入、检索、向量库管理）
├── config/
│   ├── Keys.json             # API 密钥配置
│   ├── load_key.py           # 密钥读取工具
│   └── __init__.py
└── resource/
    └── coupletData_top10000.csv  # 对联样本数据（10000 条）
```

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `COUPLET_BOOTSTRAP_LIMIT` | 首次建库写入的最大条数 | `100` |
| `COUPLET_EMBEDDING_MODEL` | 嵌入模型名称 | `tongyi-embedding-vision-plus-2026-03-06` |

## 注意事项

- 本项目使用阿里云百炼（DashScope）的 API，首次运行需要配置有效的 API Key
- Chroma 向量数据库为本地运行生成的文件，未上传至 Git 仓库，首次运行时会自动创建
- 对联检索依赖 DashScope 多模态嵌入模型，请确保账户有对应模型的调用权限
