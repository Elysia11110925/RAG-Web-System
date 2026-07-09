# RAG-QA · 知识库问答系统

> 北京科技大学 · 计算机与人工智能实践 · 第三周项目

## ✨ 核心功能

- 📁 **文档上传与管理** — 支持 PDF/TXT 上传，自动解析、切块、向量化
- 🔍 **语义检索** — 基于 ChromaDB 的 Top-K 相似度检索
- 🤖 **RAG 问答** — DeepSeek 大模型基于检索结果生成准确回答
- 💬 **对话式界面** — ChatGPT 风格聊天 UI，暗橙黑主题
- ⚡ **流式输出** — SSE 逐字推送，打字机效果
- 📖 **来源追溯** — 每个回答标注引用的文档和片段

---

## 🌐 在线演示

```
http://127.0.0.1:5002
```

---

## 🔧 环境配置

| 组件 | 版本/型号 |
|------|----------|
| Python | 3.14.3 |
| PyTorch | 2.11.0+cu128 |
| Flask | 3.1.3 |
| ChromaDB | 1.5.9 |
| LangChain | 1.3.12 |
| sentence-transformers | 5.6.0 |
| Embedding Model | all-MiniLM-L6-v2 (384维) |
| LLM | DeepSeek (deepseek-chat) |
| GPU | NVIDIA GeForce RTX 5060 Laptop (8GB) |

---

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/Elysia11110925/RAG-QA.git
cd RAG-QA
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 API Key

注册 [DeepSeek](https://platform.deepseek.com) 获取 API Key，然后设置环境变量：

```powershell
# Windows (永久)
setx DEEPSEEK_API_KEY "sk-your-key"

# Windows (临时)
set DEEPSEEK_API_KEY=sk-your-key
```

### 4. 启动服务

```bash
python app.py
```

浏览器打开 `http://127.0.0.1:5002`

---

## 📐 系统架构

```
┌─────────────────────┐                        ┌──────────────────────┐
│   前端 (HTML/CSS/JS)  │ ◄────── SSE ──────────│   Flask 后端 (app.py) │
│                      │                        │                      │
│  - 文档上传/管理      │ ── POST /api/upload ─► │  /api/upload         │
│  - 聊天对话界面       │ ◄─ GET /api/documents─ │  /api/documents      │
│  - 流式打字机效果     │ ── DELETE /api/docs──► │  /api/documents/<id> │
│  - 参考来源标注       │ ── POST /api/chat ──► │  /api/chat (SSE)     │
└─────────────────────┘                        └──────┬───────────────┘
                                                      │
                                         ┌────────────┼────────────┐
                                         │            │            │
                                    pdfplumber   LangChain   ChromaDB
                                    (PDF解析)    (文本切块)   (向量存储)
                                         │            │            │
                                         └────────────┼────────────┘
                                                      │
                                              sentence-transformers
                                              (all-MiniLM-L6-v2)
                                                      │
                                              DeepSeek API
                                              (deepseek-chat)
```

---

## 📡 API 接口

### 健康检查

```
GET /api/health
```

响应：
```json
{
  "status": "ok",
  "device": "cuda",
  "embedding_model": "all-MiniLM-L6-v2",
  "documents": 1,
  "total_chunks": 19,
  "llm": "deepseek-chat"
}
```

### 上传文档

```
POST /api/upload
Content-Type: multipart/form-data

file: document.pdf
```

响应：
```json
{
  "success": true,
  "doc_id": "786f3bbc",
  "name": "document.pdf",
  "chunks": 12,
  "total_chunks": 12
}
```

### 文档列表

```
GET /api/documents
```

### 删除文档

```
DELETE /api/documents/<doc_id>
```

### RAG 问答 (SSE 流式)

```
POST /api/chat
Content-Type: application/json

{"question": "RAG是什么？"}
```

SSE 事件类型：
- `token` — 答案片段
- `sources` — 引用来源
- `done` — 回答完成

---

## 📁 项目结构

```
RAG-QA/
├── app.py                    # Flask 后端
├── templates/
│   └── index.html            # 前端聊天界面
├── screenshots/              # 系统截图
├── requirements.txt          # Python 依赖
├── 实验记录.md               # 实验记录
├── uploads/                  # 上传文档存储
└── chroma_db/                # 向量数据库
```

---

## 🔧 已知问题与解决

| # | 问题 | 原因/解决 |
|---|------|----------|
| 1 | ChromaDB API 不兼容 | 使用内置 SentenceTransformerEmbeddingFunction |
| 2 | 中文文件名上传崩溃 | Werkzeug secure_filename 过滤中文 → UUID 重命名 |
| 3 | 嵌入模型首次加载慢 | 80MB 从 HF 下载 → 启动时预加载 |

---

## 📚 参考资料

- [ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch) — 上游项目
- [ChromaDB](https://docs.trychroma.com/) — 向量数据库
- [LangChain](https://python.langchain.com/) — LLM 编排框架
- [DeepSeek API](https://platform.deepseek.com/api-docs/) — 大模型 API
- [sentence-transformers](https://sbert.net/) — 嵌入模型

---

> **作者**: Elysia11110925
> **日期**: 2026年7月
> **课程**: 北京科技大学 · 计算机与人工智能实践 · 第三周
