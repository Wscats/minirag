# 🤖 MiniRAG - 最小化 RAG 引擎

用最少的代码实现完整的 RAG（检索增强生成）引擎，**零依赖**，纯 JavaScript，详细中文注释，适合学习和理解 RAG 原理。

## 核心特性

- **零依赖**：不需要 npm install，纯 Node.js 原生模块
- **CodeBuddy CLI 驱动**：使用 CodeBuddy CLI 作为 LLM Provider，无需 API Key
- **纯 JS 向量化**：用 TF-IDF 实现文本向量化，不依赖外部 Embedding API
- **详细注释**：每个函数、每个步骤都有清晰的中文注释和原理解释

## 文件结构

```
minirag/
├── minirag.mjs     # 核心引擎 (~300行) — LLM客户端、分块、TF-IDF、向量存储、RAG编排
├── knowledge.mjs   # 示例知识库 — 5篇关于RAG技术的文档
├── main.mjs        # 演示入口 — 完整的 索引→查询 流程
└── README.md       # 本文件
```

## 快速开始

```bash
# 前提：已安装 CodeBuddy CLI
codebuddy --version

# 运行（使用默认问题）
node main.mjs

# 自定义问题
node main.mjs "什么是向量数据库？"
node main.mjs "TF-IDF 的原理是什么？"
```

## RAG 核心流程

```
┌─ 索引阶段 ────────────────────────────────────┐
│  文档 → 分块(splitText) → TF-IDF向量化 → 存储   │
└───────────────────────────────────────────────┘

┌─ 查询阶段 ──────────────────────────────────────────────────┐
│  问题 → TF-IDF向量化 → 余弦相似度检索 → 构建提示词 → LLM回答  │
└─────────────────────────────────────────────────────────────┘
```

## 核心组件（全在 minirag.mjs 中）

| 组件 | 函数 | 作用 |
|------|------|------|
| LLM 客户端 | `createLLMClient()` | 通过子进程调用 CodeBuddy CLI 进行对话 |
| 文本分块 | `splitText()` | 按段落+句子分割文本，支持重叠 |
| 中文分词 | `tokenize()` | bigram 切分 + 英文单词提取 |
| 向量化 | `createTfidf()` | TF-IDF 词频-逆文档频率向量化 |
| 相似度 | `cosineSimilarity()` | 余弦相似度计算 |
| 向量存储 | `createVectorStore()` | 内存存储 + 暴力搜索 Top-K |
| RAG 引擎 | `createRAG()` | 编排以上所有组件的完整流程 |

## TF-IDF vs Embedding API

| 特性 | TF-IDF（本项目） | Embedding API |
|------|------------------|---------------|
| 依赖 | 零依赖 | 需要网络 API |
| 原理 | 词频统计 | 神经网络语义理解 |
| 效果 | 关键词匹配好 | 语义理解更深 |
| 适用 | 学习/小规模 | 生产环境 |

## 学习路线

建议按以下顺序阅读代码：

1. **`tokenize()`** — 理解分词是 NLP 的第一步
2. **`createTfidf()`** — 理解 TF-IDF 如何将文本变成数字
3. **`cosineSimilarity()`** — 理解向量相似度的数学原理
4. **`splitText()`** — 理解分块对 RAG 效果的影响
5. **`createVectorStore()`** — 理解向量存储和检索
6. **`createLLMClient()`** — 理解 LLM 调用方式
7. **`createRAG()`** — 理解整个 RAG 流程的编排

## License

MIT
