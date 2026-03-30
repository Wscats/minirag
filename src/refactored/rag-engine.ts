/**
 * Mini RAG Engine - TypeScript refactored version
 * Based on wscats-projects-refactor-spec.md
 */

export interface Document {
  id: string;
  content: string;
  metadata?: Record<string, unknown>;
  embedding?: number[];
}

export interface RetrievalResult {
  document: Document;
  score: number;
}

export interface RAGConfig {
  chunkSize?: number;
  chunkOverlap?: number;
  topK?: number;
}

export class TextSplitter {
  constructor(
    private chunkSize: number = 512,
    private overlap: number = 50
  ) {}

  split(text: string): string[] {
    const chunks: string[] = [];
    let start = 0;
    while (start < text.length) {
      const end = Math.min(start + this.chunkSize, text.length);
      chunks.push(text.slice(start, end));
      start += this.chunkSize - this.overlap;
    }
    return chunks;
  }
}

export class VectorStore {
  private documents: Document[] = [];

  async add(docs: Document[]): Promise<void> {
    this.documents.push(...docs);
  }

  async search(queryEmbedding: number[], topK: number = 5): Promise<RetrievalResult[]> {
    return this.documents
      .filter(doc => doc.embedding && doc.embedding.length > 0)
      .map(doc => ({
        document: doc,
        score: this.cosineSimilarity(queryEmbedding, doc.embedding!),
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length) throw new Error('Vector dimensions must match');
    const dot = a.reduce((sum, ai, i) => sum + ai * b[i], 0);
    const normA = Math.sqrt(a.reduce((sum, ai) => sum + ai * ai, 0));
    const normB = Math.sqrt(b.reduce((sum, bi) => sum + bi * bi, 0));
    return normA && normB ? dot / (normA * normB) : 0;
  }

  get size(): number {
    return this.documents.length;
  }
}

export class RAGEngine {
  private vectorStore = new VectorStore();
  private splitter: TextSplitter;

  constructor(private config: RAGConfig = {}) {
    this.splitter = new TextSplitter(config.chunkSize, config.chunkOverlap);
  }

  async ingest(text: string, metadata?: Record<string, unknown>): Promise<void> {
    const chunks = this.splitter.split(text);
    const docs: Document[] = chunks.map((chunk, i) => ({
      id: `${Date.now()}-${i}`,
      content: chunk,
      metadata,
      embedding: this.simpleEmbed(chunk),
    }));
    await this.vectorStore.add(docs);
  }

  async query(question: string): Promise<string> {
    const queryEmbedding = this.simpleEmbed(question);
    const results = await this.vectorStore.search(queryEmbedding, this.config.topK || 3);
    const context = results.map(r => r.document.content).join('\n\n');
    return `Context:\n${context}\n\nQuestion: ${question}`;
  }

  private simpleEmbed(text: string): number[] {
    // Simple TF-based embedding for zero-dependency version
    const words = text.toLowerCase().split(/\W+/).filter(Boolean);
    const freq = new Map<string, number>();
    words.forEach(w => freq.set(w, (freq.get(w) || 0) + 1));
    return Array.from(freq.values()).map(v => v / words.length);
  }
}
