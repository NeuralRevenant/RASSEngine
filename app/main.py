import os
import json
import numpy as np
from typing import List, Tuple, Optional, Dict, AsyncGenerator

import asyncio
from contextlib import asynccontextmanager

import httpx
import uvicorn
import redis

from fastapi import FastAPI, Body, WebSocket, WebSocketDisconnect
from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.helpers import bulk

# ==============================================================================
# Global Constants & Configuration
# ==============================================================================
OLLAMA_API_URL = "http://localhost:11434/api"

EMBED_MODEL_NAME = "mxbai-embed-large:latest"

MAX_BLUEHIVE_CONCURRENCY = 5
MAX_EMBED_CONCURRENCY = 5

BLUEHIVE_BEARER_TOKEN = os.getenv("BLUEHIVE_BEARER_TOKEN", "")

BATCH_SIZE = 64
CHUNK_SIZE = 512
EMBED_DIM = 1024

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_MAX_ITEMS = 1000
REDIS_CACHE_LIST = "query_cache_lfu"
CACHE_SIM_THRESHOLD = 0.96

PMC_DIR = "PMC"

# ==============================================================================
# Redis Client & Cache Functions
# ==============================================================================
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def lfu_cache_get(query_emb: np.ndarray) -> Optional[str]:
    """Retrieve a cached answer if we find a sufficiently similar embedding."""
    cached_list = redis_client.lrange(REDIS_CACHE_LIST, 0, -1)
    if not cached_list:
        return None

    query_vec = query_emb[0]
    best_sim = -1.0
    best_index = -1
    best_entry_data = None

    for i, item in enumerate(cached_list):
        entry = json.loads(item)
        emb_list = entry["embedding"]

        cached_emb = np.array(emb_list, dtype=np.float32)
        sim = cosine_similarity(query_vec, cached_emb)
        if sim > best_sim:
            best_sim = sim
            best_index = i
            best_entry_data = entry

    if best_sim < CACHE_SIM_THRESHOLD:
        return None

    if best_entry_data:
        # increment freq
        best_entry_data["freq"] = best_entry_data.get("freq", 1) + 1
        redis_client.lset(REDIS_CACHE_LIST, best_index, json.dumps(best_entry_data))
        return best_entry_data["response"]

    return None


def _remove_least_frequent_item():
    """Helper to remove the least frequently used entry from Redis."""
    cached_list = redis_client.lrange(REDIS_CACHE_LIST, 0, -1)
    if not cached_list:
        return

    min_freq = float("inf")
    min_index = -1
    for i, item in enumerate(cached_list):
        entry = json.loads(item)
        freq = entry.get("freq", 1)
        if freq < min_freq:
            min_freq = freq
            min_index = i

    if min_index >= 0:
        item_str = cached_list[min_index]
        redis_client.lrem(REDIS_CACHE_LIST, 1, item_str)


def lfu_cache_put(query_emb: np.ndarray, response: str):
    """Insert a new entry into the LFU Redis cache."""
    entry = {"embedding": query_emb.tolist()[0], "response": response, "freq": 1}
    current_len = redis_client.llen(REDIS_CACHE_LIST)
    if current_len >= REDIS_MAX_ITEMS:
        _remove_least_frequent_item()

    redis_client.lpush(REDIS_CACHE_LIST, json.dumps(entry))


# ==============================================================================
# Asynchronous Functions for Ollama Embeddings
# ==============================================================================
async def ollama_embed_text(text: str, model: str = EMBED_MODEL_NAME) -> List[float]:
    """
    Request an embedding for `text` from Ollama via HTTP POST (using the Jina model).
    """
    async with httpx.AsyncClient() as client:
        payload = {"model": model, "prompt": text, "stream": False}
        resp = await client.post(
            f"{OLLAMA_API_URL}/embeddings", json=payload, timeout=30.0
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("embedding", [])


async def embed_texts_in_batches(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """
    Embeds a list of texts in smaller batches, respecting concurrency limits.
    """
    if not texts:
        return np.array([])

    all_embeddings = []
    concurrency_sem = asyncio.Semaphore(MAX_EMBED_CONCURRENCY)

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        async def embed_single(txt: str) -> List[float]:
            async with concurrency_sem:
                return await ollama_embed_text(txt)

        tasks = [embed_single(txt) for txt in batch]
        batch_embeddings = await asyncio.gather(*tasks)
        all_embeddings.extend(batch_embeddings)

    return np.array(all_embeddings, dtype=np.float32)


async def embed_query(query: str) -> np.ndarray:
    """
    Obtain an embedding for a single query from Ollama (Jina embedding).
    """
    if not query.strip():
        return np.array([])

    emb = await ollama_embed_text(query, EMBED_MODEL_NAME)
    return np.array([emb], dtype=np.float32)


# ==============================================================================
# BlueHive Completion Generation
# ==============================================================================
BLUEHIVE_SEMAPHORE = asyncio.Semaphore(MAX_BLUEHIVE_CONCURRENCY)


async def bluehive_generate_text(prompt: str, system_msg: str = "") -> str:
    """
    Asynchronously call BlueHive's /api/v1/completion endpoint.
    Passing a systemMessage plus the user prompt, and parse out the assistant's content.
    """
    headers = {
        "Authorization": f"Bearer {BLUEHIVE_BEARER_TOKEN}",
        "Content-Type": "application/json",
    }
    url = "https://ai.bluehive.com/api/v1/completion"

    payload = {
        "prompt": prompt,
        "systemMessage": system_msg,
    }

    try:
        async with BLUEHIVE_SEMAPHORE:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    url, json=payload, headers=headers, timeout=30.0
                )
                resp.raise_for_status()
                data = resp.json()

        # expecting data["choices"][0]["message"]["content"]
        choices = data.get("choices", [])
        if not choices:
            return "[ERROR] No choices in BlueHive response."

        first_choice = choices[0]
        message = first_choice.get("message", {})
        content = message.get("content", "")
        return content.strip()
    except httpx.HTTPStatusError as e:
        # handling the HTTP errors gracefully
        print(f"[ERROR] HTTPStatusError: {str(e)}")
        print(f"[DEBUG] Response Status Code: {e.response.status_code}")
        print(f"[DEBUG] Response Text: {e.response.text}")
        # return (
        #     "[ERROR] An unexpected error occurred.\n"
        #     f"Status Code: {e.response.status_code}\n"
        # )
        return None
    except httpx.RequestError as e:
        # handle request level errors like any connection issues
        print(f"[ERROR] RequestError: {str(e)}")
        # return "[ERROR] Failed connecting to the server. Please try again later."
        return None
    except Exception as e:
        # unexpected exceptions
        print(f"[ERROR] Unexpected Exception: {str(e)}")
        # return "[ERROR] An unexpected error occurred. Please try again."
        return None


# ==============================================================================
# OpenSearch Setup with HNSW ANN
# ==============================================================================
OPENSEARCH_HOST = os.environ.get("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.environ.get("OPENSEARCH_PORT", 9200))
OPENSEARCH_INDEX_NAME = "medical-search-index2"

os_client: Optional[OpenSearch] = None
try:
    os_client = OpenSearch(
        hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
        http_compress=True,
        use_ssl=False,
        verify_certs=False,
        connection_class=RequestsHttpConnection,
    )
    info = os_client.info()
    print(f"[INFO] Connected to OpenSearch: {info['version']}")

    if not os_client.indices.exists(OPENSEARCH_INDEX_NAME):
        index_body = {
            "settings": {"index": {"knn": True}},
            "mappings": {
                "properties": {
                    "doc_id": {"type": "keyword"},
                    "text": {"type": "text"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": EMBED_DIM,
                        "method": {
                            "name": "hnsw",
                            "engine": "nmslib",
                            "space_type": "cosinesimil",
                            "parameters": {"m": 64, "ef_construction": 500},
                        },
                    },
                }
            },
        }
        resp = os_client.indices.create(index=OPENSEARCH_INDEX_NAME, body=index_body)
        print(f"[INFO] Created '{OPENSEARCH_INDEX_NAME}' index with HNSW.")
    else:
        print(f"[INFO] OpenSearch index '{OPENSEARCH_INDEX_NAME}' is ready.")
except Exception as e:
    print(f"[WARNING] OpenSearch not initialized: {e}")
    os_client = None


class OpenSearchIndexer:
    """
    Index documents with embeddings into OpenSearch using the HNSW algorithm.
    """

    def __init__(self, client: OpenSearch, index_name: str):
        self.client = client
        self.index_name = index_name

    def has_any_data(self) -> bool:
        if not self.client:
            return False
        try:
            resp = self.client.count(index=self.index_name)
            return resp["count"] > 0
        except Exception:
            return False

    def add_embeddings(self, embeddings: np.ndarray, docs: List[Dict[str, str]]):
        if not self.client or embeddings.size == 0:
            print("[OpenSearchIndexer] No embeddings or no OpenSearch client.")
            return

        actions = []
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-9)

        for i, (doc_dict, emb) in enumerate(zip(docs, embeddings)):
            doc_id = doc_dict["doc_id"]
            text_content = doc_dict["text"]
            actions.append(
                {
                    "_op_type": "index",
                    "_index": self.index_name,
                    "_id": f"{doc_id}_{i}",
                    "_source": {
                        "doc_id": doc_id,
                        "text": text_content,
                        "embedding": emb.tolist(),
                    },
                }
            )
            if len(actions) >= BATCH_SIZE:
                self._bulk_index(actions)
                actions = []

        if actions:
            self._bulk_index(actions)

    def _bulk_index(self, actions):
        try:
            success, errors = bulk(self.client, actions)
            print(f"[OpenSearchIndexer] Inserted {success} docs, errors={errors}")
        except Exception as e:
            print(f"[OpenSearchIndexer] Bulk indexing error: {e}")

    def search(
        self, query_emb: np.ndarray, k: int = 3
    ) -> List[Tuple[Dict[str, str], float]]:
        if not self.client or query_emb.size == 0:
            return []

        q_norm = np.linalg.norm(query_emb, axis=1, keepdims=True)
        query_emb = query_emb / (q_norm + 1e-9)
        vector = query_emb[0].tolist()
        query_body = {
            "size": k,
            "query": {"knn": {"embedding": {"vector": vector, "k": k}}},
        }
        try:
            resp = self.client.search(index=self.index_name, body=query_body)
            hits = resp["hits"]["hits"]
            results = []
            for h in hits:
                doc_score = h["_score"]
                doc_source = h["_source"]
                results.append((doc_source, float(doc_score)))

            print(f"[OpenSearchIndexer] Found {len(results)} relevant results.")
            return results
        except Exception as e:
            print(f"[OpenSearchIndexer] Search error: {e}")
            return []


# ==============================================================================
# Basic Pre-processing Functions
# ==============================================================================
def basic_cleaning(text: str) -> str:
    return text.replace("\n", " ").strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """
    Splits text into chunks of roughly chunk_size words.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk.strip())

    return chunks


# ==============================================================================
# RAG Model
# ==============================================================================
class RAGModel:
    """
    Retrieval-Augmented Generation Model with:
    - OpenSearch for ANN-based retrieval
    - Redis for caching
    - Ollama for embeddings
    - BlueHive for text generation
    """

    def __init__(self):
        self.os_indexer: Optional[OpenSearchIndexer] = None
        if os_client:
            self.os_indexer = OpenSearchIndexer(os_client, OPENSEARCH_INDEX_NAME)

    async def build_embeddings_from_scratch(self, pmc_dir: str):
        """
        Reads text files from the given directory, splits them into chunks,
        obtains embeddings, and indexes them in OpenSearch.
        """
        if not self.os_indexer:
            print("[RAGModel] No OpenSearchIndexer => cannot build embeddings.")
            return

        if self.os_indexer.has_any_data():
            print("[RAGModel] OpenSearch already has data. Skipping embedding.")
            return

        print("[RAGModel] Building embeddings from scratch...")
        data_files = os.listdir(pmc_dir)
        all_docs = []

        for fname in data_files:
            if fname.startswith("PMC") and fname.endswith(".txt"):
                path = os.path.join(pmc_dir, fname)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        text = f.read()
                except UnicodeDecodeError:
                    with open(path, "r", encoding="latin-1") as f:
                        text = f.read()

                cleaned_text = basic_cleaning(text)
                text_chunks = chunk_text(cleaned_text, CHUNK_SIZE)
                for chunk_str in text_chunks:
                    all_docs.append({"doc_id": fname, "text": chunk_str})

        if not all_docs:
            print("[RAGModel] No text found in directory. Exiting.")
            return

        print(f"[RAGModel] Generating embeddings for {len(all_docs)} chunks...")
        chunk_texts = [d["text"] for d in all_docs]

        embs = await embed_texts_in_batches(chunk_texts, batch_size=64)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.os_indexer.add_embeddings, embs, all_docs)
        print("[RAGModel] Finished embedding & indexing data in OpenSearch.")

    def os_search(self, query_emb: np.ndarray, top_k: int = 3):
        """
        Synchronous call to OpenSearch indexer for approximate k-NN retrieval.
        """
        if not self.os_indexer:
            return []

        return self.os_indexer.search(query_emb, k=top_k)

    async def ask(self, query: str, top_k: int = 3) -> str:
        """
        Pipeline for a user query:
          1) Check if query is empty.
          2) Embed query, check Redis for a similar query.
          3) If not cached, retrieve top_k results from OpenSearch.
          4) Call BlueHive with final combined context.
          5) Cache new result to Redis.
          6) Return answer.
        """
        if not query.strip():
            return "[ERROR] Empty query."

        # Check cache
        query_emb = await embed_query(query)
        cached_resp = lfu_cache_get(query_emb)
        if cached_resp is not None:
            print("[RAGModel] Found a similar query in Redis. Returning cached result.")
            return cached_resp

        # OpenSearch Retrieval
        partial_results = self.os_search(query_emb, top_k)
        context_map = {}
        for doc_dict, _score in partial_results:
            doc_id = doc_dict["doc_id"]
            text_chunk = doc_dict["text"]
            if doc_id not in context_map:
                context_map[doc_id] = text_chunk
            else:
                context_map[doc_id] += "\n" + text_chunk

        # Build a short context
        context_text = ""
        for doc_id, doc_content in context_map.items():
            print(doc_id)
            context_text += f"--- Document ID: {doc_id} ---\n{doc_content}\n\n"

        # system and user prompts
        system_msg = (
            "You are a helpful AI assistant chatbot. You must follow these rules:\n"
            "1) Always cite document IDs from the context exactly as 'Document XYZ' without any file extensions like '.txt'.\n"
            "2) For every answer generated, there should be a reference or citation of the IDs of the documents from which the answer information was extracted at the end of the answer!\n"
            "3) If the context does not relate to the query, say 'I lack the context to answer your question.' For example, if the query is about gene mutations but the context is about climate change, acknowledge the mismatch and do not answer.\n"
            "4) Never ever give responses based on your own knowledge of the user query. Only use the provided context to extract information relevant to the question. You should not answer without document ID references from which the information was extracted.\n"
            "5) If you lack context, then say so.\n"
            "6) Do not add chain-of-thought.\n"
            # "7) Answer in at most 4 sentences.\n"
        )
        final_prompt = (
            f"User Query:\n{query}\n\n"
            f"Context:\n{context_text}\n"
            "--- End of context ---\n\n"
            "Provide your concise answer now."
        )

        answer = await bluehive_generate_text(
            prompt=final_prompt, system_msg=system_msg
        )
        if not answer:
            return "Error: No response was generated. Please try later!"

        lfu_cache_put(query_emb, answer)
        return answer


# ==============================================================================
# FastAPI Application Setup
# ==============================================================================
app = FastAPI(
    title="RAG with BlueHive/OpenAI + Jina Embeddings + OpenSearch + Redis",
    version="1.0.0",
    description=(
        "RAG pipeline using:\n"
        "- BlueHive AI/OpenAI for text generation\n"
        "- Ollama's Jina embeddings for document/query embeddings\n"
        "- OpenSearch for ANN retrieval\n"
        "- Redis for caching\n"
        "- HTTP + WebSockets for streaming tokens\n"
    ),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Lifespan] Initializing RAGModel...")
    global rag_model
    rag_model = RAGModel()

    await rag_model.build_embeddings_from_scratch(PMC_DIR)
    print("[Lifespan] RAGModel is ready.")
    yield
    print("[Lifespan] Server is shutting down...")


app.router.lifespan_context = lifespan


def get_rag_model() -> RAGModel:
    return rag_model


@app.post("/ask")
async def ask_route(payload: dict = Body(...)):
    """
    API endpoint that processes a user query through the RAG pipeline:
    1) Retrieval from OpenSearch
    2) BlueHive generation
    3) Redis caching
    """
    query: str = payload.get("query", "")
    if not query.strip():
        return {"query": "", "answer": "[ERROR] Empty query."}

    top_k = payload.get("top_k", 3)
    print(f"[Debug] query = {query}, top_k = {top_k}")
    rm = get_rag_model()
    if not rm:
        return {"error": "RAGModel not initialized"}

    answer = await rm.ask(query, top_k)
    return {"query": query, "answer": answer}


import openai

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


async def openai_generate_text_stream(
    prompt: str, system_msg: str = ""
) -> AsyncGenerator[str, None]:
    """
    Stream tokens or text chunks from OpenAI's GPT-4o model.
    """
    openai.api_key = OPENAI_API_KEY
    try:
        async with BLUEHIVE_SEMAPHORE:
            # OpenAI streaming call
            response = await openai.ChatCompletion.acreate(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1024,
                stream=True,  # enable token-by-token streaming
                temperature=0.7,
            )

            # Yield each token or piece of text as it's generated
            async for chunk in response:
                if "choices" in chunk:
                    delta = chunk["choices"][0]["delta"]
                    token = delta.get("content", "")
                    if token:
                        yield token

    except Exception as e:
        # If the API or network fails mid stream, yield an error token or message
        yield f"[ERROR] {str(e)}"


@app.websocket("/ws/ask")
async def ask_websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for streaming RAG-based answers token-by-token.
    Expects the client to send JSON: {"query": "...", "top_k": int}.
    """
    await websocket.accept()
    try:
        # Receive JSON from the client (the query + optional top_k)
        data_str = await websocket.receive_text()
        data = json.loads(data_str)
        query: str = data.get("query", "")
        if not query.strip():
            await websocket.send_text("[ERROR] Empty query.")
            await websocket.close()
            return

        top_k = data.get("top_k", 3)
        rm = get_rag_model()
        if not rm:
            await websocket.send_text(json.dumps({"error": "RAGModel not initialized"}))
            await websocket.close()
            return

        # Check Redis cache first
        query_emb = await embed_query(query)
        cached_resp = lfu_cache_get(query_emb)
        if cached_resp:
            # If there's a cached response, just send it either all at once or chunked
            await websocket.send_text(cached_resp)
            await websocket.close()
            return

        # Perform retrieval from OpenSearch
        partial_results = rm.os_search(query_emb, top_k)
        context_map = {}
        for doc_dict, _score in partial_results:
            doc_id = doc_dict["doc_id"]
            text_chunk = doc_dict["text"]
            if doc_id not in context_map:
                context_map[doc_id] = text_chunk
            else:
                context_map[doc_id] += "\n" + text_chunk

        # build the context prompt
        context_text = ""
        for doc_id, doc_content in context_map.items():
            print(doc_id)
            context_text += f"--- Document ID: {doc_id} ---\n{doc_content}\n\n"

        system_msg = (
            "You are a helpful AI assistant chatbot. You must follow these rules:\n"
            "1) Always cite document IDs from the context exactly as 'Document XYZ' without any file extensions like '.txt'.\n"
            "2) For every answer generated, there should be a reference or citation of the IDs of the documents from which the answer information was extracted at the end of the answer!\n"
            "3) If the context does not relate to the query, say 'I lack the context to answer your question.' For example, if the query is about gene mutations but the context is about climate change, acknowledge the mismatch and do not answer.\n"
            "4) Never ever give responses based on your own knowledge of the user query. Only use the provided context to extract information relevant to the question. You should not answer without document ID references from which the information was extracted.\n"
            "5) If you lack context, then say so.\n"
            "6) Do not add chain-of-thought.\n"
            # "7) Answer in at most 4 sentences.\n"
        )
        final_prompt = (
            f"User Query:\n{query}\n\n"
            f"Context:\n{context_text}\n"
            "--- End of context ---\n\n"
            "Provide your concise answer now."
        )

        # stream the generation token-by-token
        streamed_chunks = []
        async for chunk in openai_generate_text_stream(final_prompt, system_msg):
            streamed_chunks.append(chunk)
            # Send each chunk immediately to the client
            await websocket.send_text(chunk)

        # After finishing, store the full response in Redis
        final_answer = "".join(streamed_chunks)
        if final_answer.strip():
            lfu_cache_put(query_emb, final_answer)

        await websocket.close()

    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected mid-stream.")
    except Exception as e:
        print(f"[WebSocket] Unexpected error: {e}")
        await websocket.close()


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
