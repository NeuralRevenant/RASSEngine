# **RASS Engine: Natural Language Search over EHR / Medical Documents**

🚀 A **Retrieval-Augmented Semantic Search (RASS)** system designed to support **natural language querying** on large-scale medical and EHR documents.  
🔍 Built for **fast, intelligent, and accurate retrieval** with semantic understanding, contextual responses, and caching support.

---

## 📽️ Demo Videos

- **[🖥️ Usage Demo]**: See RASS in action querying EHRs using natural language. (Link to be added)
- **[🛠️ Dev Setup]**: Learn how to install, configure, and run the system. (Link to be added)

---

## 🧠 Architecture

```mermaid
flowchart TD
  subgraph User["🧑 User
Microservices Layer"]
    UQ["Query via REST or WebSocket"]
    UF["Upload FHIR / TXT Files"]
  end

  subgraph UploadService["📂 File Upload Service"]
    F1["Accept File (FHIR/Plain-Text)"]
    F2["Store File to Disk"]
    F3["Call FHIR Parser & Indexer"]
  end

  subgraph FastAPI["🧠 RASS Engine (FastAPI)"]
    A1["Intent Classification (Zero-shot)"]
    A2["Cache Lookup (Redis LFU)"]
    A3["Retriever (Hybrid/Keyword/Semantic)"]
    A4["Answer Generation (BlueHive AI)"]
    A5["Cache Response + Save Chat"]
    A6["Call FHIR Parser & Indexer"]
  end

  subgraph Embedding["🧬 Embedding Model (via Ollama)"]
    B1["Env-driven model (e.g. mxbai-embed-large, jina-embed, etc.)"]
  end

  subgraph OpenSearch["🔎 OpenSearch"]
    C1["HNSW Vector Search"]
    C2["Text Fields (FHIR / Narrative)"]
  end

  subgraph Redis["⚡ Redis"]
    D1["LFU Caching of (Embedding, Answer)"]
  end

  subgraph BlueHive["🤖 BlueHive / LLM"]
    E1["LLM-based Contextual Answer Generation"]
  end

  subgraph FHIRParser["🧩 FHIR Parser & Indexer"]
    P1["Parse + Chunk + Embed"]
    P2["Store in OpenSearch"]
  end

  %% Query Flow
  UQ --> A1 --> A2
  A2 -->|Cache Hit| A5
  A2 -->|Cache Miss| A3
  A3 --> C1 & C2
  A3 --> B1
  A3 --> A4 --> E1
  A4 --> A5 --> D1

  %% Upload Flow
  UF --> F1 --> F2 --> F3 --> P1 --> B1
  P1 --> P2 --> C2 & C1

  %% Ingestion Triggered from RASS
  A6 --> P1
```

---

## 🔑 Key Features

- ✅ **Natural language interface** using REST & WebSocket endpoints.
- 🧠 **Zero-shot classifier** (via HuggingFace model) determines:
  - `SEMANTIC` (vector search),
  - `KEYWORD` (text search),
  - or `HYBRID` (combined).
- 🧬 **Embedding model configurable** via `.env` (Ollama API integration):
  ```env
  OLLAMA_EMBED_MODEL=mxbai-embed-large:latest
  ```
- 📦 **Dedicated FHIR & text ingestion flow**:
  - From Upload Service or RASS Engine.
  - Automatically parsed, chunked, embedded, and stored in OpenSearch.
- 🔁 **Caching**:
  - LFU-style query cache in Redis with embedding-based similarity.
- 🔎 **Retrieval engine**:
  - OpenSearch HNSW for ANN.
  - Text-based multi-match queries for full-text relevance.
- 📘 **Citation-enforced LLM generation** using BlueHive or OpenAI GPT-4o.
- 🔧 **.env-controlled architecture** – zero hardcoding.

---

## ⚙️ Setup & Running

### ✅ Prerequisites

- Python 3.8+
- Local services (with appropriate ports):
  - Redis
  - OpenSearch
  - Ollama (any embedding model)
- PostgreSQL + Prisma ORM

---

### 📦 Install

```bash
git clone https://github.com/NeuralRevenant/RASSEngine
cd RASSEngine
pip install -r requirements.txt
```

---

### 🛠️ Configure `.env`

Create `.env` (or copy `.env.example`) and define:

```env
OLLAMA_EMBED_MODEL=mxbai-embed-large:latest
OPENAI_API_KEY=...
BLUEHIVEAI_URL=http://localhost:8001/generate
OPENSEARCH_HOST=localhost
OPENSEARCH_PORT=9200
REDIS_HOST=localhost
EMB_DIR=notes
POSTGRES_DSN=postgresql://...
...
```

All runtime behavior, model selection, and service ports are environment-driven.

---

### 🚀 Run the RASS Engine

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

This will also trigger automatic ingestion from `EMB_DIR`.

---

### 📤 Run the Upload Service

```bash
uvicorn upload_service:app --host 0.0.0.0 --port 8001
```

This service handles file uploads (`.json` FHIR bundles or `.txt` medical notes), stores to disk, and calls the FHIR parser/indexer.

---

## 💬 Query API Usage

### `POST /ask`

```json
{
  "query": "What is Ghrelin?",
  "user_id": "abc123",
  "chat_id": "xyz789"
}
```

**Sample Response**:

```json
{
  "query": "What is Ghrelin?",
  "answer": "Ghrelin is a hormone that regulates appetite... (Document ABC, Document XYZ)"
}
```

### `WebSocket /ws/ask`

Streams the response token-by-token — perfect for UI integration.

---

## 📚 FHIR Ingestion Pipeline

- Handles `.json` FHIR Bundles and `.txt` notes.
- Uses `resourceType` to extract both:
  - Structured fields (e.g., Patient, Condition, Observation).
  - Narrative sections (e.g., `text.div`, `note[]`) for semantic embedding.
- Supports smart chunking via `CHUNK_SIZE` env var.

---

## 🔩 Tech Stack

| Layer         | Tool / Service         |
|---------------|------------------------|
| API Layer     | FastAPI                |
| Embeddings    | Ollama (any local model) |
| Retrieval     | OpenSearch (Text + Vector) |
| LLM Backend   | BlueHive / OpenAI      |
| Caching       | Redis            |
| DB Storage    | PostgreSQL + Prisma    |
| File Upload   | FastAPI Upload Service |
| Ingestion     | FHIR Parser     |
| Config        | `.env` driven          |

---

## 📁 Indexing Behavior

- Structured documents: stored with typed fields.
- Unstructured chunks: embedded with vector + narrative text.
- All records indexed in OpenSearch:
  - Supports both ANN (`embedding`) and text (`multi_match`) fields.
  - Supports HNSW parameters like `m`, `ef_construction`, and `cosinesimil`.

---

## 🔧 Dev & Debug Tips

- Use `redis-cli` to inspect cache:
  ```bash
  redis-cli lrange query_cache_lfu 0 -1
  ```
- Change embedding model at runtime by editing `.env`:
  ```env
  OLLAMA_EMBED_MODEL=jina-embed-en
  ```
- Control chunk sizes, ANN behavior, and similarity threshold via:
  ```env
  CHUNK_SIZE=512
  CACHE_SIM_THRESHOLD=0.96
  ```

---

## 💡 Future Roadmap

- [ ] LangChain + toolformer-like flows.
- [ ] Integrated frontend for querying and upload.
- [ ] Multi-hop QA support.
- [ ] Chat memory management across long sessions.
- [ ] Real-time citation-linked UI display.

---


## 🤝 Contributions & Feedback

Pull requests and issue reports are welcome! Feel free to reach out via Issues or Discussions.
