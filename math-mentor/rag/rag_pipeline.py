import os, json
from typing import List, Tuple

KNOWLEDGE_DIR = "knowledge_base"
INDEX_FILE = "rag/index_store.json"

def chunk_text(text, chunk_size=400, overlap=80):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def load_knowledge_base():
    docs = []
    if not os.path.exists(KNOWLEDGE_DIR):
        return docs
    for filename in os.listdir(KNOWLEDGE_DIR):
        if filename.endswith(".txt"):
            with open(os.path.join(KNOWLEDGE_DIR, filename), "r", encoding="utf-8") as f:
                content = f.read()
            topic = filename.replace(".txt","").replace("_"," ").title()
            for i, chunk in enumerate(chunk_text(content)):
                docs.append({"source": filename, "topic": topic,
                              "chunk_id": i, "text": chunk})
    return docs

def build_index(client=None):
    docs = load_knowledge_base()
    os.makedirs("rag", exist_ok=True)
    with open(INDEX_FILE, "w") as f:
        json.dump({"docs": docs}, f)
    print(f"Index built: {len(docs)} chunks")
    return {"docs": docs}

def load_index():
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "r") as f:
            return json.load(f)
    return {"docs": []}

def keyword_score(query: str, text: str) -> float:
    query_words = set(query.lower().split())
    text_words = set(text.lower().split())
    if not query_words:
        return 0.0
    overlap = query_words & text_words
    return len(overlap) / len(query_words)

def retrieve(query: str, client=None, top_k: int = 5):
    index_data = load_index()
    if not index_data["docs"]:
        index_data = build_index()
    scored = []
    for doc in index_data["docs"]:
        score = keyword_score(query, doc["text"])
        if score > 0:
            scored.append((doc, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]

def format_retrieved_context(results):
    if not results:
        return "No relevant context found."
    parts = ["=== RETRIEVED KNOWLEDGE BASE CONTEXT ===\n"]
    for i, (doc, score) in enumerate(results, 1):
        parts.append(
            f"[Source {i}: {doc['source']} | Relevance: {score:.2f}]\n{doc['text']}\n"
        )
    return "\n".join(parts)

def get_index_stats():
    index_data = load_index()
    docs = index_data.get("docs", [])
    topics = {}
    for d in docs:
        t = d.get("topic","unknown")
        topics[t] = topics.get(t, 0) + 1
    return {"total_chunks": len(docs), "topics": topics, "index_built": len(docs) > 0}