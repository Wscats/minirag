/**
 * ============================================
 *   MiniRAG 演示 - 完整的 RAG 流程演示
 * ============================================
 *
 * 演示 RAG 的完整工作流程：
 *   1. 加载知识库文档
 *   2. 文档分块 + TF-IDF 索引构建
 *   3. 用户提问 → 检索 → LLM 生成回答
 *
 * 运行方式（无需 API Key，使用 CodeBuddy CLI）：
 *   node main.mjs                    # 使用预设问题
 *   node main.mjs "你想问的问题"       # 自定义问题
 *
 * 前提条件：
 *   已安装 CodeBuddy CLI（codebuddy 命令可用）
 */

import { createRAG } from "./minirag.mjs";
import { documents } from "./knowledge.mjs";

// ============================================
// 主流程
// ============================================

function main() {
  console.log("╔══════════════════════════════════════╗");
  console.log("║     🤖 MiniRAG - 最小化 RAG 引擎     ║");
  console.log("║   使用 CodeBuddy CLI 作为 LLM 提供者   ║");
  console.log("╚══════════════════════════════════════╝\n");

  // ---- 第 1 步：创建 RAG 引擎 ----
  console.log("⚙️  初始化 RAG 引擎...");
  const rag = createRAG({
    chunkSize: 300,   // 每块最多 300 字符
    chunkOverlap: 50, // 相邻块重叠 50 字符
    topK: 3,          // 检索返回前 3 个最相关的块
  });

  // ---- 第 2 步：加载知识库文档 ----
  console.log("\n📚 加载知识库文档...");
  for (const doc of documents) {
    console.log(`  加载: ${doc.title}`);
    rag.addDocument(doc.content, { title: doc.title });
  }

  // ---- 第 3 步：构建索引 ----
  rag.build();

  // ---- 第 4 步：提问并获取回答 ----
  // 从命令行参数读取问题，或使用默认问题
  const question = process.argv[2] || "RAG 系统如何解决大语言模型的幻觉问题？";

  console.log("━".repeat(50));
  const { answer, sources } = rag.query(question);

  // ---- 显示结果 ----
  console.log("━".repeat(50));
  console.log("💬 回答：\n");
  console.log(answer);

  console.log("\n" + "━".repeat(50));
  console.log("📎 引用来源：");
  for (const src of sources) {
    const title = src.metadata.title || "未知";
    console.log(`  [${title}] 相似度: ${src.score.toFixed(3)} | ${src.text}`);
  }
  console.log();
}

// 运行
try {
  main();
} catch (err) {
  console.error("\n❌ 运行出错:", err.message);

  if (err.message.includes("ENOENT") || err.message.includes("codebuddy")) {
    console.error("\n💡 提示：请确保已安装 CodeBuddy CLI");
    console.error("   运行 codebuddy --version 检查是否可用");
  }

  process.exit(1);
}
