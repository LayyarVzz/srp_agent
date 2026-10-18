# RAG MCP 模块

RAG MCP 是 `srp_agent` 中独立的知识检索服务。它通过 FastMCP 的 `stdio` 传输注册
`search_knowledge` 工具，在线使用同一份预构建 Chunk 集合完成 BM25 与 FAISS 向量双路召回，
再通过 Reciprocal Rank Fusion（RRF）合并结果。

当前版本支持从**单个 TXT 或 Markdown 文件**离线构建知识库。它不支持目录扫描、自动多文件同步、
增量更新或在线重建索引。

> 本文中的所有命令都应从仓库根目录执行。下文以 `<repo>` 表示仓库根目录；
> 需要配置绝对路径时，请将其替换为本机实际仓库位置。PowerShell 与 Bash 示例分别适用于 Windows 和 Linux/macOS。

## 架构与运行边界

离线构建链路：

```text
单个 TXT/Markdown
  -> KnowledgeBuilder
  -> DocumentChunk[]
  -> OpenAI-compatible Embedding（分批）
  -> float32 + L2 normalize
  -> FAISS IndexFlatIP
  -> index.faiss + chunks.json + manifest.json
  -> A2 加载器校验后发布
```

在线检索链路：

```text
FastMCP search_knowledge(query, top_k)
  -> RAGPipeline
  -> HybridRetriever
       |- BM25Retriever
       `- VectorRetriever -> embed_query -> FAISS IndexFlatIP
  -> RRF 融合
  -> SearchKnowledgeResponse
```

在线服务只加载预构建的三件套，不读取或重新切分原始 Markdown/TXT，也不执行文档
Embedding。索引在服务模块导入时加载一次；索引重建或替换后必须重启 MCP 服务。

Vector 查询失败（包括 Embedding 超时）时，`HybridRetriever` 会记录通道失败并使用正常的
BM25 结果继续融合。两路同时失败时会明确报错。索引缺失、损坏或配置不匹配属于启动错误，
不会静默退化成旧 Demo BM25。

## 关键文件

```text
services/rag_mcp/
├── README.md                    # 本文档
├── config.py                    # RAG 本地配置与默认值
├── schemas.py                   # MCP 响应模型
├── server.py                    # FastMCP stdio 入口
├── tools/search.py              # 生产装配与 search_knowledge
├── rag/
│   ├── builder.py               # KnowledgeBuilder
│   ├── splitter.py              # TXT/Markdown 切分
│   ├── models.py                # DocumentChunk、RetrievedChunk
│   ├── index.py                 # Manifest、三件套加载与一致性校验
│   ├── retriever.py             # BM25、Vector、Hybrid Retriever
│   ├── fusion.py                # RAG 内部 RRF
│   └── pipeline.py              # Retriever 到 MCP Schema 的映射
└── knowledge/index/             # 本地生成索引；Git 忽略

scripts/build_rag_faiss_index.py # 离线构建 CLI
scripts/test_rag_mcp_client.py    # 确定性 FastMCP stdio 客户端烟测
tests/rag_mcp/                    # RAG MCP 确定性测试
tests/fixtures/rag_mcp/
└── uat_company_policy.md        # 默认 UAT 知识源，可由 Git 跟踪
```

## Python 环境与依赖

项目要求 Python 3.12 或更高版本，依赖由根目录 `pyproject.toml` 和 `uv.lock` 管理。RAG
核心依赖包括 `fastmcp`、`faiss-cpu`、`langchain-openai`、`rank-bm25`、`numpy`、
`pydantic-settings`；测试和检查使用 `pytest` 与 `ruff`。

安装项目及默认开发依赖：

```powershell
# Windows PowerShell
uv sync
.venv\Scripts\python.exe --version
```

```bash
# Linux/macOS
uv sync
.venv/bin/python --version
```

推荐预先安装 `uv` 并将其加入 `PATH`。当前开发环境使用 Python 3.12；`uv sync` 以锁文件为准，
不要为单独运行 RAG 手工升级某个依赖。若 Windows 的全局 `uv` 不在 `PATH`，但现有虚拟环境中
包含 `uv.exe`，可以用下列只读检查确认锁文件和环境状态：

```powershell
.venv\Scripts\uv.exe lock --check
.venv\Scripts\python.exe --version
```

`uv lock --check` 只检查 `pyproject.toml` 与 `uv.lock` 的一致性，不会安装、升级或删除依赖。
没有现成虚拟环境的新机器仍需先安装 `uv`，再执行前面的 `uv sync`。已有环境也可以直接使用
本文各命令所示的 `.venv\Scripts\python.exe`。需要严格核对环境时可另行执行 `uv sync --check`；
该命令会把额外安装但未锁定的工具也报告为环境不一致。

## Embedding 配置

根目录 `settings.py` 从进程环境变量或根目录 `.env` 读取以下配置：

| 环境变量 | 必需 | 说明 |
|---|---:|---|
| `EMBEDDING_ENABLED` | 在线必需 | MCP 服务必须设为 `true` |
| `EMBEDDING_MODEL` | 是 | 必须与 `manifest.json` 的 `embedding_model` 一致 |
| `EMBEDDING_DIMS` | 是 | 正整数，必须与模型输出及 Manifest 一致 |
| `EMBEDDING_BASE_URL` | 兼容端点需要 | OpenAI-compatible API 根地址 |
| `EMBEDDING_API_KEY` | 服务需要 | API 密钥；不要提交到 Git 或写进本文档 |

当前构建器生成的 Manifest provider 为 `openai_compatible`。已经验收的 UAT 索引使用：

- 模型：`text-embedding-v4`
- 维度：`1024`
- Provider：`openai_compatible`

PowerShell 临时环境示例：

```powershell
$env:EMBEDDING_ENABLED = "true"
$env:EMBEDDING_MODEL = "text-embedding-v4"
$env:EMBEDDING_DIMS = "1024"
$env:EMBEDDING_BASE_URL = "https://your-openai-compatible-endpoint.example/v1"
$env:EMBEDDING_API_KEY = "<从安全凭据存储读取，不要提交>"

$env:RAG_QUERY_EMBEDDING_TIMEOUT = "10"
$env:RAG_QUERY_EMBEDDING_MAX_RETRIES = "0"
```

Bash 示例：

```bash
export EMBEDDING_ENABLED=true
export EMBEDDING_MODEL=text-embedding-v4
export EMBEDDING_DIMS=1024
export EMBEDDING_BASE_URL=https://your-openai-compatible-endpoint.example/v1
export EMBEDDING_API_KEY='<从安全凭据存储读取，不要提交>'

export RAG_QUERY_EMBEDDING_TIMEOUT=10
export RAG_QUERY_EMBEDDING_MAX_RETRIES=0
```

离线构建脚本会在构建边界内启用 Embedding，但仍需正确配置模型、维度、Base URL 和密钥。
在线 MCP 服务必须显式设置 `EMBEDDING_ENABLED=true`，否则会在初始化时失败。

### RAG 本地配置

| 环境变量 | 默认值 | 说明 |
|---|---:|---|
| `RAG_INDEX_DIR` | `services/rag_mcp/knowledge/index/uat_company_policy` | 三件套目录，可覆盖 |
| `RAG_CANDIDATE_MULTIPLIER` | `3` | 每路候选数相对最终 `top_k` 的倍率 |
| `RAG_MAX_CANDIDATES` | `50` | 候选上限，也是工具允许的最大 `top_k` |
| `RAG_RRF_K` | `60` | RRF 常数，必须为正整数 |
| `RAG_VECTOR_MIN_SCORE` | `0.0` | Vector 原始内积分过滤阈值 |
| `RAG_QUERY_EMBEDDING_TIMEOUT` | `10` 秒 | 单次在线 Embedding 请求超时，允许范围 `(0, 60]` |
| `RAG_QUERY_EMBEDDING_MAX_RETRIES` | `0` | 在线重试数，允许范围 `[0, 3]` |

`RAG_QUERY_EMBEDDING_TIMEOUT=10` 只约束单次在线 Embedding 请求，不是整个 MCP 工具调用的
硬性总截止时间。

## UAT 知识源与默认索引

当前默认知识源是：

```text
tests/fixtures/rag_mcp/uat_company_policy.md
```

它是虚构的《星河科技有限公司员工与IT服务管理制度（测试版）》，用于开发和验收，不应替换成
真实企业敏感资料。生产在线服务的默认索引目录是：

```text
services/rag_mcp/knowledge/index/uat_company_policy/
```

目录必须包含：

```text
index.faiss
chunks.json
manifest.json
```

整个 `services/rag_mcp/knowledge/index/` 被 `.gitignore` 忽略，因此三件套不随 Git 提交。
新环境克隆仓库后必须先执行离线构建，不能假定索引已经存在。

## 离线构建 FAISS 三件套

构建器当前只接受一个 `.txt`、`.md` 或 `.markdown` 文件。UAT 构建命令必须显式传入
`--output-dir`，因为构建 CLI 自身的通用默认目录与在线服务的 UAT 默认子目录不同。

PowerShell：

```powershell
.venv\Scripts\python.exe -u -m scripts.build_rag_faiss_index `
  --input tests/fixtures/rag_mcp/uat_company_policy.md `
  --source-id uat_company_policy `
  --source-title "星河科技有限公司员工与IT服务管理制度（测试版）" `
  --output-dir services/rag_mcp/knowledge/index/uat_company_policy `
  --embedding-batch-size 10 `
  --embedding-timeout 20 `
  --embedding-max-retries 0
```

Linux/macOS：

```bash
.venv/bin/python -u -m scripts.build_rag_faiss_index \
  --input tests/fixtures/rag_mcp/uat_company_policy.md \
  --source-id uat_company_policy \
  --source-title '星河科技有限公司员工与IT服务管理制度（测试版）' \
  --output-dir services/rag_mcp/knowledge/index/uat_company_policy \
  --embedding-batch-size 10 \
  --embedding-timeout 20 \
  --embedding-max-retries 0
```

CLI 参数可用以下命令核对：

```powershell
.venv\Scripts\python.exe -m scripts.build_rag_faiss_index --help
```

构建流程按批调用 `embed_documents`，保持 Chunk 和向量顺序；随后执行 float32 转换、有限值和
零向量检查、L2 归一化以及 `IndexFlatIP` 建立。构建器先在临时目录生成并使用线上加载器校验，
再发布到目标目录，发布后再次校验。

`manifest.json` Schema v1 包含索引类型、metric、归一化声明、Embedding provider/model/维度、
Chunk 数量及 `index.faiss`、`chunks.json` 的 SHA-256。三个文件的逐文件替换不是文件系统事务；
异常时实现会尽力回滚，但进程或机器在发布中途终止仍可能留下暂时不一致的目录。

## 启动 MCP Server

确认三件套存在、Embedding 配置与 Manifest 一致后，从仓库根目录启动：

```powershell
# 不设置 RAG_INDEX_DIR 时使用默认 UAT 目录
Remove-Item Env:RAG_INDEX_DIR -ErrorAction SilentlyContinue
$env:EMBEDDING_ENABLED = "true"
$env:RAG_QUERY_EMBEDDING_TIMEOUT = "10"
$env:RAG_QUERY_EMBEDDING_MAX_RETRIES = "0"
.venv\Scripts\python.exe -u -m services.rag_mcp.server
```

```bash
unset RAG_INDEX_DIR
export EMBEDDING_ENABLED=true
export RAG_QUERY_EMBEDDING_TIMEOUT=10
export RAG_QUERY_EMBEDDING_MAX_RETRIES=0
.venv/bin/python -u -m services.rag_mcp.server
```

这是 `stdio` MCP 服务，不是交互式终端程序。启动后等待输入是正常现象；不要在终端里手工输入
JSON 或普通文本。应由 MCP Host/Client 以子进程方式启动，并保持 stdout 仅用于 MCP 协议。
配置环境变量必须在子进程启动前完成，因为工具模块导入时就会装配并验证索引。

## MCP 工具与响应

工具签名保持为：

```python
search_knowledge(query: str, top_k: int = 5) -> SearchKnowledgeResponse
```

- `query` 必须是字符串。空字符串返回空 `chunks`。
- `top_k` 必须是正整数；布尔值不作为整数接受。
- `top_k` 不得超过 `RAG_MAX_CANDIDATES`，默认上限为 50。

请求示例：

```json
{
  "query": "VPN访问授权有效期是多久？",
  "top_k": 5
}
```

响应结构：

```json
{
  "query": "VPN访问授权有效期是多久？",
  "chunks": [
    {
      "chunk_id": "uat_company_policy:12",
      "content": "...",
      "source": {
        "id": "uat_company_policy",
        "title": "星河科技有限公司员工与IT服务管理制度（测试版）",
        "url": null
      },
      "metadata": {
        "chunk_index": 12,
        "heading_path": ["...", "第二章：远程办公与VPN访问", "申请、审批与有效期"]
      },
      "score": 1.0,
      "score_details": {
        "bm25": 19.386,
        "vector": 0.8338,
        "rrf": 0.03279,
        "bm25_rank": 1,
        "vector_rank": 1
      }
    }
  ]
}
```

分数含义：

- `bm25`：BM25 原始分数。
- `vector`：L2 归一化查询向量与索引向量的内积相似度。
- `rrf`：未归一化的 RRF 排名分。
- `score`：归一化 RRF 排名展示值。
- `bm25_rank`、`vector_rank`：各通道的1-based排名。

单通道降级时，失败通道对应的分数及名次保持 `null`，不会伪造成双通道命中。

## 客户端验证

### 确定性 FastMCP 客户端测试

以下测试使用临时索引和 Fake Embedding，不访问外部服务，但会通过 FastMCP `Client` 验证工具
注册、参数约束、Hybrid 结果和降级契约：

```powershell
.venv\Scripts\python.exe -m pytest tests/rag_mcp/test_mcp_hybrid.py -q
```

```bash
.venv/bin/python -m pytest tests/rag_mcp/test_mcp_hybrid.py -q
```

### 真实 stdio Host

真实客户端应以仓库根目录作为子进程工作目录，并在启动环境中提供前述 Embedding 配置：

```text
Windows command: <repo>\.venv\Scripts\python.exe
Windows args:    -u -m services.rag_mcp.server

Linux command:   <repo>/.venv/bin/python
Linux args:      -u -m services.rag_mcp.server
```

`<repo>` 是占位符；在 MCP Host 配置中，应将其替换为本机仓库根目录的绝对路径。

连接后先执行 `list_tools`，应只看到 `search_knowledge`；随后以请求示例调用该工具。不要先导入
`services.rag_mcp.server` 再设置 `RAG_INDEX_DIR` 或 Embedding 环境变量。

### FastMCP stdio 客户端烟测

Windows PowerShell：

```powershell
.venv\Scripts\python.exe -m scripts.test_rag_mcp_client
```

Linux/macOS：

```bash
.venv/bin/python -m scripts.test_rag_mcp_client
```

该脚本已经完成生产装配兼容修复：它启动真实 FastMCP stdio 子进程，在临时目录中生成合法的
FAISS 三件套，并在服务模块导入前注入 Fake Query Embedding。它不会访问真实 Embedding API，
也不会读取或覆盖默认 UAT 索引。烟测会验证 `list_tools` 中包含 `search_knowledge`、BM25 与
Vector 两路均实际参与 RRF 融合、`chunk_id`、`score`、`score_details` 和两路 rank 完整，并
验证 `top_k` 的严格类型校验。测试结束时会关闭子进程并清理临时目录；Vector 降级结果不会被
当作正常双路 Hybrid 验证通过。

## 测试与静态检查

完整 RAG MCP 测试：

```powershell
.venv\Scripts\python.exe -m pytest tests/rag_mcp -q
```

```bash
.venv/bin/python -m pytest tests/rag_mcp -q
```

离线构建烟测、Pipeline 兼容回归：

```powershell
.venv\Scripts\python.exe -m scripts.test_rag_faiss_index_build
.venv\Scripts\python.exe -m scripts.test_rag_pipeline
```

```bash
.venv/bin/python -m scripts.test_rag_faiss_index_build
.venv/bin/python -m scripts.test_rag_pipeline
```

Ruff 与补丁空白检查：

```powershell
.venv\Scripts\python.exe -m ruff check --no-cache services/rag_mcp scripts/build_rag_faiss_index.py scripts/test_rag_mcp_client.py tests/rag_mcp
git diff --check
```

## 常见问题

### 启动时报索引文件缺失

新环境不会从 Git 获得三件套。先执行离线构建，并确认 `--output-dir` 与服务使用的
`RAG_INDEX_DIR` 一致。默认应为 `services/rag_mcp/knowledge/index/uat_company_policy`。

### 启动时报 Embedding 配置或维度不匹配

检查 `EMBEDDING_ENABLED`、`EMBEDDING_MODEL` 和 `EMBEDDING_DIMS`。加载器会比较当前配置与
Manifest 的 provider、model 和 dimensions；不一致时拒绝启动。不要通过编辑 Manifest 绕过校验，
应使用正确配置重新构建完整三件套。

### 启动时报 SHA-256 不一致或 FAISS 无法读取

三件套可能来自不同构建批次、复制不完整或已经损坏。不要只替换其中一个文件；使用同一构建命令
重新生成并发布完整目录。

### 查询只返回 BM25 分数

检查日志中的 Vector 通道错误类型。在线 Embedding 超时或其他异常会触发诚实的 BM25 降级，
此时 `score_details.vector` 和 `vector_rank` 应为 `null`。默认单次请求超时为10秒、自动重试为0，
但整个 MCP 调用没有独立的硬性总超时。

### 更新知识文件后结果没有变化

在线服务不会监听或重新切分源文件。需要重新运行离线构建，确认三件套通过校验，然后重启 MCP
服务以加载新索引。

### 服务启动后终端看起来“卡住”

`stdio` Server 正在等待 MCP Host 的协议消息，这是正常行为。不要把它当作命令行问答程序使用。

## 已知限制与后续方向

当前版本具有以下明确边界：

- 只支持单个 TXT/Markdown 输入，不支持多文件目录导入、自动同步或增量索引。
- RRF 是排序融合算法，不负责判断答案真实性。
- 归一化 `score` 不是概率，也不是经过校准的语义相关度或回答置信度。
- 当前没有 Rerank。
- 当前没有相邻 Chunk 自动上下文扩展。
- 当前 Splitter 使用较小的固定 Chunk 上限；同一自然段中的场景与直接答案可能位于相邻 Chunk。
- 某些语义改写查询中，Vector 能正确召回答案 Chunk，但该 Chunk 仍可能落在最终 Top-5 之外。
- 无答案查询仍可能返回词法或语义上相近但不能直接回答问题的 Chunk。
- BM25 单通道降级保障可用性，但不等价于正常的双路 Hybrid 命中。
- 在线 Embedding 超时只约束单次请求，当前 MCP 工具没有独立的整体调用截止时间。

后续可按独立任务评估同标题相邻 Chunk 扩展、Splitter 调整、质量评测集、Rerank、多文件导入和
索引版本化。不得为单个测试问题硬编码关键词、答案或 `chunk_id`。
