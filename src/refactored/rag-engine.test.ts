import { TextSplitter, VectorStore, RAGEngine } from './rag-engine';

describe('TextSplitter', () => {
  test('splits text into chunks', () => {
    const splitter = new TextSplitter(10, 0);
    const chunks = splitter.split('Hello World This Is A Test');
    expect(chunks[0].length).toBeLessThanOrEqual(10);
    expect(chunks.length).toBeGreaterThan(1);
  });

  test('handles overlap', () => {
    const splitter = new TextSplitter(10, 5);
    const chunks = splitter.split('ABCDEFGHIJKLMNOPQRST');
    expect(chunks.length).toBeGreaterThan(2);
  });

  test('handles empty string', () => {
    const splitter = new TextSplitter(10, 0);
    expect(splitter.split('')).toEqual(['']);
  });
});

describe('VectorStore', () => {
  test('adds and searches documents', async () => {
    const store = new VectorStore();
    await store.add([
      { id: '1', content: 'doc1', embedding: [1, 0, 0] },
      { id: '2', content: 'doc2', embedding: [0, 1, 0] },
      { id: '3', content: 'doc3', embedding: [0.9, 0.1, 0] },
    ]);
    expect(store.size).toBe(3);

    const results = await store.search([1, 0, 0], 2);
    expect(results).toHaveLength(2);
    expect(results[0].document.id).toBe('1');
    expect(results[0].score).toBeCloseTo(1.0);
  });

  test('returns empty for no documents', async () => {
    const store = new VectorStore();
    const results = await store.search([1, 0, 0]);
    expect(results).toHaveLength(0);
  });
});

describe('RAGEngine', () => {
  test('ingests and queries', async () => {
    const engine = new RAGEngine({ topK: 2 });
    await engine.ingest('TypeScript is a typed superset of JavaScript');
    const answer = await engine.query('What is TypeScript?');
    expect(answer).toContain('TypeScript');
  });
});
