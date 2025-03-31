import os
from dotenv import load_dotenv
import shutil
import asyncio
import uvicorn
import numpy as np
import time
import pathlib
from typing import List

import httpx
from fastapi import FastAPI, File, UploadFile, Form, HTTPException

from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.helpers import bulk

import asyncpg

###############################################################################
# Load Environment Variables
###############################################################################
load_dotenv()

BASE_UPLOAD_DIR = "uploads"
os.makedirs(BASE_UPLOAD_DIR, exist_ok=True)

# Ollama embedding configuration
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api")
EMBED_MODEL_NAME = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large:latest")
MAX_EMBED_CONCURRENCY = 5  # concurrency for embedding calls
BATCH_SIZE = 64  # process 64 chunks per batch
CHUNK_SIZE = 512  # split text into ~512-word chunks
EMBED_DIM = 1024  # embedding dimension

# Base index name for user-specific indexes
BASE_OPENSEARCH_INDEX_NAME = os.getenv("OPENSEARCH_INDEX_NAME", "")

# OpenSearch connection
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", 9200))

# PostgreSQL credentials
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "")
POSTGRES_DB = os.getenv("POSTGRES_DB", "")
POSTGRES_USER = os.getenv("POSTGRES_USER", "")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))

###############################################################################
# FastAPI Application
###############################################################################
app = FastAPI(
    title="User-Specific Plain Text Upload & Embedding",
    version="1.0.0",
    description=(
        "A microservice that:\n"
        "1. Accepts text files + user_id.\n"
        "2. Derives doc_id from the file names + timestamps.\n"
        "3. Stores the files in 'uploads/{user_id}/doc_id.ext'.\n"
        "4. Chunks & embeds the text.\n"
        "5. Indexes all chunks under the same doc_id in the user-specific OpenSearch index.\n"
    ),
)

###############################################################################
# OpenSearch Setup
###############################################################################
try:
    os_client = OpenSearch(
        hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
        http_compress=True,
        use_ssl=False,
        verify_certs=False,
        connection_class=RequestsHttpConnection,
    )
except Exception as e:
    print(f"[ERROR] Could not connect to OpenSearch: {e}")
    os_client = None


def init_user_index(user_id: str):
    """
    Create user-specific index <BASE_OPENSEARCH_INDEX_NAME>-<user_id> if it doesn't exist.
    """
    if not os_client:
        print("[WARNING] No OpenSearch client => skipping index creation.")
        return

    index_name = f"{BASE_OPENSEARCH_INDEX_NAME}-{user_id}"
    if os_client.indices.exists(index_name):
        print(f"[INFO] Index '{index_name}' already exists.")
        return

    try:
        index_body = {
            "settings": {"index": {"knn": True}},
            "mappings": {
                "properties": {
                    # The single doc_id that identifies the file
                    "doc_id": {"type": "keyword"},
                    # The chunk text
                    "text": {"type": "text"},
                    # The embedding for that chunk
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
        os_client.indices.create(index=index_name, body=index_body)
        print(f"[INFO] Created user-specific index '{index_name}'.")
    except Exception as e:
        print(f"[ERROR] Failed creating index '{index_name}': {e}")


###############################################################################
# Chunking
###############################################################################
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """
    Splits large text into ~chunk_size-word chunks.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk_str = " ".join(words[i : i + chunk_size])
        chunks.append(chunk_str.strip())
    return chunks


###############################################################################
# Embedding Helpers
###############################################################################
async def ollama_embed_text(text: str, model: str = EMBED_MODEL_NAME) -> List[float]:
    """
    Request an embedding from Ollama via HTTP POST.
    """
    if not text.strip():
        return [0.0] * EMBED_DIM

    async with httpx.AsyncClient() as client:
        payload = {"model": model, "prompt": text, "stream": False}
        try:
            resp = await client.post(
                f"{OLLAMA_API_URL}/embeddings", json=payload, timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
            emb = data.get("embedding", [])
            if len(emb) != EMBED_DIM:
                print(
                    f"[WARNING] Mismatch embedding size. Expected {EMBED_DIM}, got {len(emb)}"
                )
            return emb
        except Exception as exc:
            print(f"[ERROR] Ollama embedding error: {exc}")
            return [0.0] * EMBED_DIM


async def embed_texts_in_batches(texts: List[str]) -> np.ndarray:
    """
    Concurrently embed text chunks in small batches. Returns [len(texts), EMBED_DIM] array.
    """
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)

    all_embeddings = []
    sem = asyncio.Semaphore(MAX_EMBED_CONCURRENCY)

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]

        async def embed_single(txt: str) -> List[float]:
            async with sem:
                return await ollama_embed_text(txt)

        tasks = [embed_single(t) for t in batch]
        results = await asyncio.gather(*tasks)
        all_embeddings.extend(results)

    return np.array(all_embeddings, dtype=np.float32)


###############################################################################
# Bulk Indexing (All Chunks => Single doc_id)
###############################################################################
def bulk_index_embeddings(
    user_id: str, doc_id: str, embeddings: np.ndarray, chunks: List[str]
):
    """
    Bulk index chunk embeddings for a single doc_id into user-specific index:
      - Each chunk is a separate doc in the index
      - `_id` = f"{doc_id}_{chunk_index}"
      - `_source.doc_id` = doc_id
      - `_source.text` = chunk
      - `_source.embedding` = the chunk’s embedding
    """
    if not os_client or embeddings.size == 0:
        print("[ERROR] Missing OpenSearch client or embeddings => cannot index.")
        return

    index_name = f"{BASE_OPENSEARCH_INDEX_NAME}-{user_id}"
    init_user_index(user_id)  # ensure user index is created

    # L2-normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_normed = embeddings / (norms + 1e-9)

    actions = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings_normed)):
        # keep chunk embedding '_id' unique by appending the chunk idx
        open_search_id = f"{doc_id}_{i}"
        source_doc = {
            "doc_id": doc_id,
            "text": chunk,
            "embedding": emb.tolist(),
        }

        actions.append(
            {
                "_op_type": "index",
                "_index": index_name,
                "_id": open_search_id,
                "_source": source_doc,
            }
        )

        if len(actions) >= BATCH_SIZE:
            try:
                success, errors = bulk(os_client, actions)
                print(
                    f"[OpenSearch] Bulk indexed {success} chunk docs for user={user_id}, doc_id={doc_id}"
                )
            except Exception as exc:
                print(
                    f"[OpenSearch] Bulk error (user={user_id} doc_id={doc_id}): {exc}"
                )
            actions = []

    # leftover
    if actions:
        try:
            success, errors = bulk(os_client, actions)
            print(
                f"[OpenSearch] Bulk indexed {success} chunk docs for user={user_id}, doc_id={doc_id}"
            )
        except Exception as exc:
            print(f"[OpenSearch] Bulk error (user={user_id} doc_id={doc_id}): {exc}")


###############################################################################
# PostgreSQL Connection & Authorization
###############################################################################
async def get_pg_connection():
    """
    Creates a new async PostgreSQL connection.
    Uses asyncpg for non-blocking operations.
    """
    try:
        conn = await asyncpg.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            database=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
        )
        return conn
    except Exception as exc:
        print(f"[ERROR] Unable to connect to PostgreSQL: {exc}")
        return None


async def check_user_authorized_in_postgres(user_id: str) -> bool:
    """
    Asynchronously checks if the user exists in the PostgreSQL database.

    1) Connects to Postgres using asyncpg.
    2) Queries the 'users' table to check if the user exists.
    3) Returns True if user is found & valid, otherwise False.
    """
    conn = await get_pg_connection()
    if not conn:
        print("[ERROR] Can't connect to Postgres => cannot verify user.")
        return False

    try:
        row = await conn.fetchrow("SELECT id, email FROM users WHERE id = $1;", user_id)

        if not row or not all(
            row.get(col) and str(row[col]).strip() for col in ["id", "email"]
        ):
            print(f"[AUTH] No valid user found with id={user_id}")
            return False

        return True
    except Exception as exc:
        print(f"[ERROR] Postgres query failed: {exc}")
        return False
    finally:
        await conn.close()


###############################################################################
# Endpoint: Upload & Store
###############################################################################
@app.post("/upload_text")
async def upload_text(
    user_id: str = Form(...),
    files: List[UploadFile] = File(...),
):
    """
    1) Validate the user via PostgreSQL calls
    2) Derive doc_id from filename + timestamp
    3) Save file => uploads/{user_id}/doc_id.ext
    4) Read & chunk text
    5) Embeddings => user-specific index, doc_id repeated for all chunks
    """

    is_authorized = await check_user_authorized_in_postgres(user_id)
    if not is_authorized:
        raise HTTPException(
            status_code=403,
            detail=f"User with id='{user_id}' does not exist or is not authorized.",
        )

    # If no files are provided
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    # Make user-specific folder
    user_folder = os.path.join(BASE_UPLOAD_DIR, user_id)
    os.makedirs(user_folder, exist_ok=True)

    for uploaded_file in files:
        # basic filename checks
        if not uploaded_file.filename.strip():
            raise HTTPException(status_code=400, detail="A file has no valid filename.")

        extension = pathlib.Path(uploaded_file.filename).suffix.lower()
        if extension != ".txt":
            raise HTTPException(
                status_code=403,
                detail=f"Invalid file format: {extension}. Only .txt allowed!",
            )

        # derive doc_id: stem + timestamp
        name_stem = pathlib.Path(uploaded_file.filename).stem
        now_ts = int(time.time())
        doc_id = f"{name_stem}_{now_ts}"

        # final path: uploads/user_id/doc_id.txt
        final_filename = f"{doc_id}{extension}"
        final_path = os.path.join(user_folder, final_filename)

        # save the file
        try:
            with open(final_path, "wb") as buffer:
                shutil.copyfileobj(uploaded_file.file, buffer)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"File save error for '{uploaded_file.filename}': {exc}",
            )

        # Read file text
        try:
            with open(final_path, "r", encoding="utf-8") as f:
                contents = f.read()
        except UnicodeDecodeError:
            with open(final_path, "r", encoding="latin-1") as f:
                contents = f.read()

        if not contents.strip():
            raise HTTPException(
                status_code=400,
                detail=f"File '{uploaded_file.filename}' is empty or has no text.",
            )

        # Chunk text
        chunks = chunk_text(contents, CHUNK_SIZE)
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail=f"File '{uploaded_file.filename}' produced no text chunks.",
            )

        # embed
        embeddings = await embed_texts_in_batches(chunks)
        if embeddings.shape[0] != len(chunks):
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Mismatch in chunk vs embedding counts for file '{uploaded_file.filename}'."
                ),
            )

        # index => user-specific doc_id
        bulk_index_embeddings(user_id, doc_id, embeddings, chunks)

    return f"Uploaded {len(files)} files & embedded documents for user='{user_id}'."


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9001, reload=False)
