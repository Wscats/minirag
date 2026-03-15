/**
 * ============================================
 *   MiniRAG - 最小化 RAG 引擎
 * ============================================
 *
 * RAG (Retrieval-Augmented Generation) = 检索增强生成
 *
 * 核心思想：
 *   不让 LLM 凭空回答，而是先从知识库中检索相关文档，
 *   再把文档作为上下文喂给 LLM，让它基于真实资料回答。
 *
 * 完整流程：
 *   ┌─ 索引阶段 ─────────────────────────────┐
 *   │  文档 → 分块 → 向量化(TF-IDF) → 存入向量库 │
 *   └──────────────────────────────────────────┘
 *   ┌─ 查询阶段 ─────────────────────────────────────────┐
 *   │  问题 → 向量化 → 检索相似块 → 构建提示词 → LLM 生成回答 │
 *   └────────────────────────────────────────────────────┘
 *
 * 技术栈：
 *   - 零依赖，纯 JavaScript
 *   - CodeBuddy CLI 作为 LLM Provider（无需 API Key）
 *   - TF-IDF 实现文本向量化（替代 Embedding API）
 */

import { execFileSync } from "node:child_process";

// ============================================
// 第一部分：CodeBuddy CLI 客户端
// ============================================

/**
 * 创建 CodeBuddy CLI 客户端
 *
 * 原理：通过子进程调用 CodeBuddy CLI，以 JSON 格式获取回复
 *
 * 命令格式：
 *   codebuddy -p                    # 非交互式打印模式
 *     --output-format json          # JSON 格式输出
 *     --max-turns 1                 # 单轮对话（不让 LLM 调用工具循环）
 *     --tools ""                    # 禁用所有工具（纯文本对话）
 *     --system-prompt "..."         # 系统提示词
 *     "用户问题"                     # 用户输入
 *
 * JSON 输出结构（数组）：
 *   [
 *     { type: "message", role: "user", ... },        // 用户消息
 *     { type: "message", role: "assistant", ... },   // 助手回复
 *     { type: "result", result: "回复文本", ... }     // 最终结果
 *   ]
 */
function createLLMClient() {
  return {
    /**
     * 调用 CodeBuddy CLI 进行对话
     *
     * @param {string} prompt - 用户提问内容
     * @param {string} systemPrompt - 系统提示词（指导 LLM 行为）
     * @returns {string} LLM 的回复文本
     */
    chat(prompt, systemPrompt = "") {
      // 构建 CLI 参数
      const args = [
        "-p",                    // print 模式：输出结果后退出
        "--output-format", "json", // JSON 格式便于程序解析
        "--max-turns", "1",      // 单轮对话，避免多轮工具调用
        "--tools", "",           // 禁用工具，纯文本对话
      ];

      // 如果有系统提示词，添加 --system-prompt 参数
      if (systemPrompt) {
        args.push("--system-prompt", systemPrompt);
      }

      // 最后一个参数是用户的提问
      args.push(prompt);

      // 同步调用 CodeBuddy CLI
      // 使用 execFileSync 阻塞等待结果，简化异步逻辑
      const output = execFileSync("codebuddy", args, {
        encoding: "utf8",        // 返回字符串而非 Buffer
        maxBuffer: 10 * 1024 * 1024, // 10MB 缓冲区，防止大回复截断
        timeout: 120_000,        // 120 秒超时
      });

      // 解析 JSON 输出，提取 result 项
      const items = JSON.parse(output);
      const result = items.find((item) => item.type === "result");

      if (!result || result.is_error) {
        throw new Error(`CodeBuddy CLI 调用失败: ${JSON.stringify(result)}`);
      }

      return result.result;
    },
  };
}

// ============================================
// 第二部分：文本分块器
// ============================================

/**
 * 将长文本切分为小块（Chunking）
 *
 * 为什么要分块？
 *   1. LLM 有上下文长度限制，不能把整篇文章都塞进去
 *   2. 分块后可以精准检索最相关的段落，而非返回整篇文档
 *   3. 小块文本的向量表示更精确，检索效果更好
 *
 * 分块策略：
 *   - 优先按段落（\n\n）分割，保持语义完整性
 *   - 如果单段超过 chunkSize，则按句号分割
 *   - 相邻块之间有 overlap 重叠，避免边界处上下文丢失
 *
 * @param {string} text - 原始文本
 * @param {number} chunkSize - 每块最大字符数（默认 300）
 * @param {number} overlap - 相邻块重叠字符数（默认 50）
 * @returns {string[]} 文本块数组
 */
function splitText(text, chunkSize = 300, overlap = 50) {
  // 第一步：按双换行拆成段落
  const paragraphs = text.split(/\n\n+/).filter((p) => p.trim());

  const chunks = [];
  let currentChunk = "";

  for (const para of paragraphs) {
    // 如果当前块加上新段落没超过限制，就合并
    if (currentChunk.length + para.length <= chunkSize) {
      currentChunk += (currentChunk ? "\n\n" : "") + para;
    } else {
      // 当前块已满，保存它
      if (currentChunk) chunks.push(currentChunk);

      // 如果单个段落就超过限制，需要进一步切割
      if (para.length > chunkSize) {
        // 按句号、问号、感叹号分割成句子
        const sentences = para.split(/(?<=[。！？.!?])/);
        currentChunk = "";
        for (const sent of sentences) {
          if (currentChunk.length + sent.length <= chunkSize) {
            currentChunk += sent;
          } else {
            if (currentChunk) chunks.push(currentChunk);
            currentChunk = sent;
          }
        }
      } else {
        // 新段落作为新块的开始
        // 使用重叠：从上一块的末尾取 overlap 个字符作为前缀
        const prev = chunks[chunks.length - 1] || "";
        const overlapText = prev.slice(-overlap);
        currentChunk = overlapText + (overlapText ? "\n\n" : "") + para;
      }
    }
  }

  // 别忘了最后一块
  if (currentChunk.trim()) chunks.push(currentChunk);

  return chunks;
}

// ============================================
// 第三部分：TF-IDF 文本向量化
// ============================================

/**
 * TF-IDF (Term Frequency - Inverse Document Frequency)
 *
 * 核心思想：
 *   一个词在当前文档中出现越多（TF 高），同时在其他文档中出现越少（IDF 高），
 *   这个词对当前文档就越重要。
 *
 * 举例：
 *   "的" 在所有文档中都出现 → IDF 低 → 不重要
 *   "向量数据库" 只在特定文档中出现 → IDF 高 → 很重要
 *
 * 对比 Embedding API：
 *   - Embedding：用神经网络理解语义，效果更好但需要 API
 *   - TF-IDF：纯统计方法，零依赖，适合学习和理解 RAG 流程
 */

/**
 * 中文分词（简易版）
 *
 * 策略：
 *   1. 提取连续的中文字符序列
 *   2. 提取连续的英文/数字序列
 *   3. 对中文使用 bigram（二元组）切分
 *      例如 "向量数据库" → ["向量", "量数", "数据", "据库"]
 *      bigram 能捕捉到词语的部分信息，比单字切分效果好
 *
 * @param {string} text - 输入文本
 * @returns {string[]} 分词结果
 */
function tokenize(text) {
  const tokens = [];
  // 转小写，统一处理
  const lower = text.toLowerCase();

  // 提取英文/数字单词
  const engWords = lower.match(/[a-z0-9]+/g) || [];
  tokens.push(...engWords);

  // 提取中文字符序列，然后用 bigram 切分
  const cnSegments = lower.match(/[\u4e00-\u9fff]+/g) || [];
  for (const seg of cnSegments) {
    // bigram：每两个相邻字符组成一个词
    // "向量数据" → ["向量", "量数", "数据"]
    for (let i = 0; i < seg.length - 1; i++) {
      tokens.push(seg[i] + seg[i + 1]);
    }
    // 也保留单字，确保短文本也能匹配
    if (seg.length === 1) tokens.push(seg);
  }

  return tokens;
}

/**
 * 创建 TF-IDF 向量化器
 *
 * 工作流程：
 *   1. fit() - 学习所有文档的词汇表，计算 IDF
 *   2. transform() - 将文本转换为 TF-IDF 向量
 *
 * @returns {object} 向量化器对象
 */
function createTfidf() {
  let vocabulary = new Map(); // 词汇表：词 → 索引
  let idfValues = [];         // 每个词的 IDF 值

  return {
    /**
     * 学习阶段：从文档集合中构建词汇表和 IDF
     *
     * IDF(词) = log(文档总数 / 包含该词的文档数)
     *   - 常见词（如 "的"）：很多文档都包含 → IDF 低
     *   - 稀有词（如 "向量数据库"）：少数文档包含 → IDF 高
     *
     * @param {string[]} documents - 文档文本数组
     */
    fit(documents) {
      const docCount = documents.length;
      // df: 每个词出现在多少个文档中
      const df = new Map();

      for (const doc of documents) {
        // 对每个文档，统计不重复的词（一个文档中出现多次只算一次）
        const uniqueTokens = new Set(tokenize(doc));
        for (const token of uniqueTokens) {
          df.set(token, (df.get(token) || 0) + 1);
        }
      }

      // 构建词汇表，并计算 IDF
      vocabulary = new Map();
      idfValues = [];
      let idx = 0;

      for (const [token, count] of df) {
        vocabulary.set(token, idx);
        // IDF 公式：log(总文档数 / 包含该词的文档数) + 1
        // +1 是平滑处理，避免 log(1) = 0 导致完全匹配的词权重为零
        idfValues.push(Math.log(docCount / count) + 1);
        idx++;
      }
    },

    /**
     * 转换阶段：将文本转为 TF-IDF 向量
     *
     * TF(词, 文档) = 词在文档中出现的次数 / 文档的总词数
     * TF-IDF(词, 文档) = TF × IDF
     *
     * @param {string} text - 输入文本
     * @returns {number[]} TF-IDF 向量（长度等于词汇表大小）
     */
    transform(text) {
      const tokens = tokenize(text);
      const totalTokens = tokens.length || 1; // 避免除零

      // 统计每个词的出现次数
      const tf = new Map();
      for (const token of tokens) {
        tf.set(token, (tf.get(token) || 0) + 1);
      }

      // 构建 TF-IDF 向量
      const vector = new Array(vocabulary.size).fill(0);
      for (const [token, count] of tf) {
        const idx = vocabulary.get(token);
        if (idx !== undefined) {
          // TF-IDF = (词频 / 总词数) × IDF
          vector[idx] = (count / totalTokens) * idfValues[idx];
        }
      }

      return vector;
    },

    /** 获取词汇表大小（向量维度） */
    get vocabSize() {
      return vocabulary.size;
    },
  };
}

// ============================================
// 第四部分：相似度计算
// ============================================

/**
 * 余弦相似度（Cosine Similarity）
 *
 * 衡量两个向量方向的相似程度，忽略长度差异：
 *
 *   cos(A, B) = (A · B) / (|A| × |B|)
 *
 * 结果范围：
 *   1  → 方向完全相同（最相似）
 *   0  → 完全正交（不相关）
 *   -1 → 方向完全相反
 *
 * 为什么用余弦而不是欧几里得距离？
 *   余弦只关注"方向"不关注"长度"，
 *   所以一篇长文档和一个短问题，只要话题相同就会有高相似度。
 *
 * @param {number[]} a - 向量 A
 * @param {number[]} b - 向量 B
 * @returns {number} 相似度分数
 */
function cosineSimilarity(a, b) {
  let dot = 0;   // 点积：A · B
  let normA = 0; // |A|²
  let normB = 0; // |B|²

  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  // 如果任一向量为零向量，相似度为 0
  const denominator = Math.sqrt(normA) * Math.sqrt(normB);
  return denominator === 0 ? 0 : dot / denominator;
}

// ============================================
// 第五部分：向量存储
// ============================================

/**
 * 创建内存向量存储（Vector Store）
 *
 * 向量存储是 RAG 的核心组件：
 *   - 存储：保存文本块及其对应的向量
 *   - 检索：给定查询向量，找到最相似的文本块
 *
 * 这里使用最简单的暴力搜索（Brute Force）：
 *   遍历所有向量，逐一计算相似度，返回 Top-K
 *
 * 生产环境中通常使用专门的向量数据库（如 Pinecone、Milvus），
 * 它们使用 ANN（近似最近邻）算法实现高效搜索。
 *
 * @returns {object} 向量存储对象
 */
function createVectorStore() {
  // 存储条目：{ text, vector, metadata }
  const entries = [];

  return {
    /**
     * 添加一个条目
     * @param {string} text - 文本内容
     * @param {number[]} vector - 文本的向量表示
     * @param {object} metadata - 附加元数据（如文档来源）
     */
    add(text, vector, metadata = {}) {
      entries.push({ text, vector, metadata });
    },

    /**
     * 检索最相似的 K 个文本块
     *
     * 算法：
     *   1. 计算查询向量与每个存储向量的余弦相似度
     *   2. 按相似度降序排列
     *   3. 返回前 K 个结果
     *
     * @param {number[]} queryVector - 查询文本的向量
     * @param {number} topK - 返回结果数量
     * @returns {Array<{text, score, metadata}>} 检索结果
     */
    search(queryVector, topK = 3) {
      return entries
        .map((entry) => ({
          text: entry.text,
          score: cosineSimilarity(queryVector, entry.vector),
          metadata: entry.metadata,
        }))
        .sort((a, b) => b.score - a.score) // 按相似度降序
        .slice(0, topK);                   // 取前 K 个
    },

    /** 已存储的条目总数 */
    get size() {
      return entries.length;
    },
  };
}

// ============================================
// 第六部分：RAG 引擎（核心编排层）
// ============================================

/**
 * 创建 RAG 引擎
 *
 * RAG 引擎是整个系统的编排层，协调各组件完成：
 *   1. 文档索引（addDocument）：文档 → 分块 → TF-IDF 向量化 → 存储
 *   2. 问题回答（query）：问题 → 向量化 → 检索 → 构建提示词 → LLM 生成
 *
 * @param {object} options - 配置选项
 * @param {number} options.chunkSize - 文本块大小（默认 300 字符）
 * @param {number} options.chunkOverlap - 块间重叠大小（默认 50 字符）
 * @param {number} options.topK - 检索返回的文档块数（默认 3）
 * @returns {object} RAG 引擎对象
 */
function createRAG(options = {}) {
  const { chunkSize = 300, chunkOverlap = 50, topK = 3 } = options;

  // 初始化各组件
  const llm = createLLMClient();    // LLM 客户端（CodeBuddy CLI）
  const store = createVectorStore(); // 向量存储
  const tfidf = createTfidf();       // TF-IDF 向量化器

  // 收集所有文本块，用于最后统一训练 TF-IDF
  let allChunks = [];
  // 标记是否已完成索引构建
  let indexed = false;

  return {
    /**
     * 添加文档到知识库
     *
     * 流程：文档 → 分块 → 暂存（等待 build() 统一向量化）
     *
     * @param {string} text - 文档文本
     * @param {object} metadata - 文档元数据（来源、标题等）
     */
    addDocument(text, metadata = {}) {
      const chunks = splitText(text, chunkSize, chunkOverlap);
      for (const chunk of chunks) {
        allChunks.push({ text: chunk, metadata });
      }
      console.log(`  📄 已添加文档，分为 ${chunks.length} 个块`);
    },

    /**
     * 构建索引
     *
     * 为什么要单独 build？
     *   TF-IDF 需要先看到所有文档才能计算 IDF（逆文档频率），
     *   所以必须在所有文档添加完毕后，统一训练向量化器。
     *
     * 流程：
     *   1. 用所有文本块训练 TF-IDF（计算词汇表和 IDF）
     *   2. 将每个文本块转为向量
     *   3. 存入向量存储
     */
    build() {
      console.log(`\n🔨 构建索引（共 ${allChunks.length} 个文本块）...`);

      // 第一步：训练 TF-IDF —— 学习词汇表和 IDF 值
      const texts = allChunks.map((c) => c.text);
      tfidf.fit(texts);
      console.log(`  📊 词汇表大小: ${tfidf.vocabSize} 个词`);

      // 第二步：将每个文本块向量化并存入向量库
      for (const chunk of allChunks) {
        const vector = tfidf.transform(chunk.text);
        store.add(chunk.text, vector, chunk.metadata);
      }

      indexed = true;
      console.log(`  ✅ 索引构建完成，共 ${store.size} 个向量\n`);
    },

    /**
     * 查询 —— RAG 的核心流程
     *
     * 完整流程：
     *   1. 将用户问题转为 TF-IDF 向量
     *   2. 在向量库中搜索最相似的 K 个文本块
     *   3. 将检索到的文本块拼成上下文
     *   4. 构建提示词：系统指令 + 上下文 + 用户问题
     *   5. 调用 CodeBuddy CLI 生成回答
     *
     * @param {string} question - 用户的问题
     * @returns {object} { answer, sources } 回答和引用来源
     */
    query(question) {
      if (!indexed) {
        throw new Error("请先调用 build() 构建索引");
      }

      console.log(`🔍 检索中: "${question}"`);

      // === 第 1 步：将问题向量化 ===
      const queryVector = tfidf.transform(question);

      // === 第 2 步：在向量库中搜索最相似的文本块 ===
      const results = store.search(queryVector, topK);

      console.log(`  📋 找到 ${results.length} 个相关文本块:`);
      for (const r of results) {
        console.log(`    - [相似度 ${r.score.toFixed(3)}] ${r.text.slice(0, 50)}...`);
      }

      // === 第 3 步：构建上下文 ===
      // 将检索到的文本块拼接成一个大字符串，作为 LLM 的参考资料
      const context = results.map((r, i) => `[文档${i + 1}] ${r.text}`).join("\n\n");

      // === 第 4 步：构建提示词 ===
      // 系统提示词：告诉 LLM 它的角色和行为规则
      const systemPrompt = [
        "你是一个知识问答助手。请严格根据下面提供的参考文档来回答用户问题。",
        "如果参考文档中没有相关信息，请如实说明。",
        "回答要简洁准确，并标注信息来源。",
      ].join("\n");

      // 用户消息：参考文档 + 实际问题
      const userMessage = [
        "以下是检索到的参考文档：",
        "---",
        context,
        "---",
        "",
        `问题：${question}`,
      ].join("\n");

      // === 第 5 步：调用 LLM 生成回答 ===
      console.log("\n🤖 CodeBuddy 生成回答中...\n");
      const answer = llm.chat(userMessage, systemPrompt);

      return {
        answer,
        sources: results.map((r) => ({
          text: r.text.slice(0, 100) + "...",
          score: r.score,
          metadata: r.metadata,
        })),
      };
    },
  };
}

// ============================================
// 导出所有组件
// ============================================

export {
  createLLMClient, // CodeBuddy CLI 客户端
  splitText,       // 文本分块器
  tokenize,        // 中文分词
  createTfidf,     // TF-IDF 向量化器
  cosineSimilarity, // 余弦相似度
  createVectorStore, // 向量存储
  createRAG,       // RAG 引擎（主入口）
};
