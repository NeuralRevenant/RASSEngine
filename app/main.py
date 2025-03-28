import os
from dotenv import load_dotenv
import json
import numpy as np
from typing import List, Tuple, Optional, Dict, AsyncGenerator

import asyncio
from contextlib import asynccontextmanager

import httpx
import uvicorn
import redis.asyncio as aioredis

import concurrent.futures
import time

from prisma import Prisma

import openai

from fastapi import (
    FastAPI,
    Body,
    WebSocket,
    WebSocketDisconnect,
    Depends,
    HTTPException,
)

from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.helpers import bulk

from transformers import pipeline


# Load .env file
load_dotenv()

# ==============================================================================
# Global Constants & Configuration
# ==============================================================================
EMBED_MODEL_NAME = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large:latest")
QUERY_INTENT_CLASSIFICATION_MODEL = os.getenv(
    "QUERY_INTENT_CLASSIFICATION_MODEL", "facebook/bart-large-mnli"
)

MAX_BLUEHIVE_CONCURRENCY = int(os.getenv("MAX_BLUEHIVE_CONCURRENCY", 5))
MAX_EMBED_CONCURRENCY = int(os.getenv("MAX_EMBED_CONCURRENCY", 5))

BLUEHIVE_BEARER_TOKEN = os.getenv("BLUEHIVE_BEARER_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
BLUEHIVE_API_URL = os.getenv("BLUEHIVEAI_URL", "")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api")

BATCH_SIZE = int(os.getenv("BATCH_SIZE", 64))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
EMBED_DIM = int(os.getenv("EMBED_DIM", 1024))

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_MAX_ITEMS = int(os.getenv("REDIS_MAX_ITEMS", 1000))
REDIS_CACHE_LIST = os.getenv("REDIS_CACHE_LIST", "query_cache_lfu")
CACHE_SIM_THRESHOLD = int(os.getenv("CACHE_SIM_THRESHOLD", 0.96))
REDIS_SHORT_TTL_SECONDS = int(
    os.getenv("REDIS_SHORT_TTL_SECONDS", 600)
)  # short TTL for newly updated entries

EMB_DIR = os.getenv("EMB_DIR", "notes")

OPENSEARCH_HOST = os.environ.get("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.environ.get("OPENSEARCH_PORT", 9200))
OPENSEARCH_INDEX_NAME = os.getenv("OPENSEARCH_INDEX_NAME", "")

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "")
POSTGRES_USER = os.getenv("POSTGRES_USER", "")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
POSTGRES_DSN = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

MAX_CHAT_HISTORY = int(os.getenv("MAX_CHAT_HISTORY", 10))  # Limit to last 10 turns
ADAPTIVE_CHUNKING = bool(
    os.getenv("ADAPTIVE_CHUNKING", True)
)  # If True - chunk size can adapt based on text length


db = Prisma()  # prisma client init


# ==============================================================================
# Redis Client & Cache Functions
# ==============================================================================
redis_client = aioredis.from_url(
    f"redis://{REDIS_HOST}:{REDIS_PORT}", decode_responses=True
)


async def close_redis():
    await redis_client.close()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


async def lfu_cache_get(query_emb: np.ndarray) -> Optional[str]:
    """Retrieve a cached answer if we find a sufficiently similar embedding."""
    cached_list = await redis_client.lrange(REDIS_CACHE_LIST, 0, -1)
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
        current_time = int(time.time())
        if "expiry" in best_entry_data and best_entry_data["expiry"] < current_time:
            # remove expired entry
            await redis_client.lrem(REDIS_CACHE_LIST, 1, json.dumps(best_entry_data))
            return None

        # increment freq
        best_entry_data["freq"] = best_entry_data.get("freq", 1) + 1
        best_entry_data["last_used"] = current_time  # track last usage
        await redis_client.lset(
            REDIS_CACHE_LIST, best_index, json.dumps(best_entry_data)
        )
        return best_entry_data["response"]

    return None


async def _remove_least_frequent_item():
    """Helper to remove the least frequently used entry from Redis."""
    cached_list = await redis_client.lrange(REDIS_CACHE_LIST, 0, -1)
    if not cached_list:
        return

    min_freq = float("inf")
    min_index = -1
    min_entry = None
    for i, item in enumerate(cached_list):
        entry = json.loads(item)
        freq = entry.get("freq", 1)
        if freq < min_freq:
            min_freq = freq
            min_index = i
            min_entry = entry

    # if found - remove it
    if min_index >= 0 and min_entry:
        await redis_client.lrem(REDIS_CACHE_LIST, 1, json.dumps(min_entry))


async def lfu_cache_put(query_emb: np.ndarray, response: str):
    """Insert a new entry into the LFU Redis cache with a TTL to avoid stale data."""
    current_time = int(time.time())
    expiry_time = current_time + REDIS_SHORT_TTL_SECONDS  # exp time set

    entry = {
        "embedding": query_emb.tolist()[0],
        "response": response,
        "freq": 1,
        "last_used": current_time,
        "expiry": expiry_time,
    }

    current_len = await redis_client.llen(REDIS_CACHE_LIST)
    if current_len >= REDIS_MAX_ITEMS:
        await _remove_least_frequent_item()

    await redis_client.lpush(REDIS_CACHE_LIST, json.dumps(entry))


# ==============================================================================
# Ollama Embeddings
# ==============================================================================
async def ollama_embed_text(text: str) -> List[float]:
    """Get an embedding from Ollama for a single text."""
    if not text.strip():
        return [0.0] * EMBED_DIM

    async with httpx.AsyncClient() as client:
        payload = {"model": EMBED_MODEL_NAME, "prompt": text, "stream": False}
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

    emb_list = await ollama_embed_text(query)
    return np.array([emb_list], dtype=np.float32)


# ==============================================================================
# Ozwell/BlueHive LLM
# ==============================================================================
BLUEHIVE_SEMAPHORE = asyncio.Semaphore(MAX_BLUEHIVE_CONCURRENCY)


async def bluehive_generate_text(prompt: str, system_msg: str = "") -> str:
    """
    Asynchronous call to BlueHive's API endpoint. Passing a systemMessage plus
    the user prompt, and parse out the assistant's content.
    """
    if not BLUEHIVE_API_URL:
        return "[ERROR] LLM not configured."

    headers = {
        "Authorization": f"Bearer {BLUEHIVE_BEARER_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "prompt": prompt,
        "systemMessage": system_msg,
    }

    try:
        async with BLUEHIVE_SEMAPHORE, httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(BLUEHIVE_API_URL, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        if "choices" not in data or not data["choices"]:
            return "[ERROR] No choices in BlueHive response."
        # expecting data["choices"][0]["message"]["content"]

        return data["choices"][0].get("message", {}).get("content", "").strip()
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


# ==============================================================
# OpenSearch Retrieval
# =====================================================
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
            "settings": {"index": {"knn": True}},  # for vector search
            "mappings": {
                "properties": {
                    # ----------------------------------------------------------------
                    # Core Identifiers & Document Typing
                    # ----------------------------------------------------------------
                    "doc_id": {"type": "keyword"},
                    "doc_type": {"type": "keyword"},
                    "resourceType": {"type": "keyword"},
                    # ----------------------------------------------------------------
                    # FHIR "Patient" resource fields
                    # ----------------------------------------------------------------
                    "patientId": {"type": "keyword"},
                    "patientName": {"type": "text"},
                    "patientGender": {"type": "keyword"},
                    "patientDOB": {
                        "type": "date",
                        "format": "yyyy-MM-dd||strict_date_optional_time||epoch_millis",
                    },
                    "patientAddress": {"type": "text"},
                    "patientMaritalStatus": {"type": "keyword"},
                    "patientMultipleBirth": {"type": "integer"},
                    "patientTelecom": {"type": "text"},
                    "patientLanguage": {"type": "keyword"},
                    # ----------------------------------------------------------------
                    # FHIR "Condition" resource fields
                    # ----------------------------------------------------------------
                    "conditionId": {"type": "keyword"},
                    "conditionCodeText": {"type": "text"},
                    "conditionCategory": {"type": "keyword"},
                    "conditionClinicalStatus": {"type": "keyword"},
                    "conditionVerificationStatus": {"type": "keyword"},
                    "conditionOnsetDateTime": {
                        "type": "date",
                        "format": "date_time||strict_date_optional_time||epoch_millis",
                    },
                    "conditionRecordedDate": {
                        "type": "date",
                        "format": "date_time||strict_date_optional_time||epoch_millis",
                    },
                    "conditionSeverity": {"type": "keyword"},
                    "conditionNote": {"type": "text"},
                    # ----------------------------------------------------------------
                    # FHIR "Observation" resource fields
                    # ----------------------------------------------------------------
                    "observationId": {"type": "keyword"},
                    "observationCodeText": {"type": "text"},
                    "observationValue": {"type": "text"},
                    "observationUnit": {"type": "keyword"},
                    "observationInterpretation": {"type": "keyword"},
                    "observationEffectiveDateTime": {
                        "type": "date",
                        "format": "date_time||strict_date_optional_time||epoch_millis",
                    },
                    "observationIssued": {
                        "type": "date",
                        "format": "date_time||strict_date_optional_time||epoch_millis",
                    },
                    "observationReferenceRange": {"type": "text"},
                    "observationNote": {"type": "text"},
                    # ----------------------------------------------------------------
                    # FHIR "Encounter" resource fields
                    # ----------------------------------------------------------------
                    "encounterId": {"type": "keyword"},
                    "encounterStatus": {"type": "keyword"},
                    "encounterClass": {"type": "keyword"},
                    "encounterType": {"type": "text"},
                    "encounterReasonCode": {"type": "text"},
                    "encounterStart": {
                        "type": "date",
                        "format": "date_time||strict_date_optional_time||epoch_millis",
                    },
                    "encounterEnd": {
                        "type": "date",
                        "format": "date_time||strict_date_optional_time||epoch_millis",
                    },
                    "encounterLocation": {"type": "text"},
                    "encounterServiceProvider": {"type": "keyword"},
                    "encounterParticipant": {"type": "text"},
                    "encounterNote": {"type": "text"},
                    # ----------------------------------------------------------------
                    # FHIR "MedicationRequest" resource fields
                    # ----------------------------------------------------------------
                    "medRequestId": {"type": "keyword"},
                    "medRequestMedicationDisplay": {"type": "text"},
                    "medRequestAuthoredOn": {
                        "type": "date",
                        "format": "date_time||strict_date_optional_time||epoch_millis",
                    },
                    "medRequestIntent": {"type": "keyword"},
                    "medRequestStatus": {"type": "keyword"},
                    "medRequestPriority": {"type": "keyword"},
                    "medRequestDosageInstruction": {"type": "text"},
                    "medRequestDispenseRequest": {"type": "text"},
                    "medRequestNote": {"type": "text"},
                    # ----------------------------------------------------------------
                    # FHIR "Procedure" resource fields
                    # ----------------------------------------------------------------
                    "procedureId": {"type": "keyword"},
                    "procedureCodeText": {"type": "text"},
                    "procedureStatus": {"type": "keyword"},
                    "procedurePerformedDateTime": {
                        "type": "date",
                        "format": "date_time||strict_date_optional_time||epoch_millis",
                    },
                    "procedureFollowUp": {"type": "text"},
                    "procedureNote": {"type": "text"},
                    # ----------------------------------------------------------------
                    # FHIR "AllergyIntolerance" resource fields
                    # ----------------------------------------------------------------
                    "allergyId": {"type": "keyword"},
                    "allergyClinicalStatus": {"type": "keyword"},
                    "allergyVerificationStatus": {"type": "keyword"},
                    "allergyType": {"type": "keyword"},
                    "allergyCategory": {"type": "keyword"},
                    "allergyCriticality": {"type": "keyword"},
                    "allergyCodeText": {"type": "text"},
                    "allergyOnsetDateTime": {
                        "type": "date",
                        "format": "date_time||strict_date_optional_time||epoch_millis",
                    },
                    "allergyNote": {"type": "text"},
                    # ----------------------------------------------------------------
                    # FHIR "Practitioner" resource fields
                    # ----------------------------------------------------------------
                    "practitionerId": {"type": "keyword"},
                    "practitionerName": {"type": "text"},
                    "practitionerGender": {"type": "keyword"},
                    "practitionerSpecialty": {"type": "keyword"},
                    "practitionerAddress": {"type": "text"},
                    "practitionerTelecom": {"type": "text"},
                    # ----------------------------------------------------------------
                    # FHIR "Organization" resource fields
                    # ----------------------------------------------------------------
                    "organizationId": {"type": "keyword"},
                    "organizationName": {"type": "text"},
                    "organizationType": {"type": "keyword"},
                    "organizationAddress": {"type": "text"},
                    "organizationTelecom": {"type": "text"},
                    # ----------------------------------------------------------------
                    # Additional unstructured text from FHIR (like resource.text.div, resource.note[].text, etc.)
                    # ----------------------------------------------------------------
                    "unstructuredText": {"type": "text"},
                    # ----------------------------------------------------------------
                    # Vector embedding field for semantic / k-NN search
                    # ----------------------------------------------------------------
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": EMBED_DIM,
                        "method": {
                            "name": "hnsw",
                            "engine": "nmslib",
                            "space_type": "cosinesimil",
                            "parameters": {"m": 48, "ef_construction": 400},
                        },
                    },
                }
            },
        }

        os_client.indices.create(index=OPENSEARCH_INDEX_NAME, body=index_body)
        print(f"[INFO] Created '{OPENSEARCH_INDEX_NAME}' index.")
    else:
        print(f"[INFO] Using existing index '{OPENSEARCH_INDEX_NAME}'.")
except Exception as e:
    print(f"[WARNING] OpenSearch not initialized: {e}")
    os_client = None


# ==============================================================================
# Parsing FHIR Bundle
# ==============================================================================
def parse_fhir_bundle(bundle_json: Dict) -> Tuple[List[Dict], List[Dict]]:
    """
    Returns two lists:
      1) structured_docs: array of dicts with typed FHIR fields
      2) unstructured_docs: array of dicts for narrative text (to embed)
    Each resource may produce:
      - One doc capturing structured fields (doc_type='structured')
      - One or more doc(s) capturing unstructured text (doc_type='unstructured')

    We handle multiple resource types: Patient, Condition, Observation, Encounter,
    MedicationRequest, Procedure, AllergyIntolerance, Practitioner, Organization, etc.
    We gather text from resource.text.div, resource.note, etc. for the unstructured docs.
    """
    structured_docs = []
    unstructured_docs = []

    if not bundle_json or "entry" not in bundle_json:
        return (structured_docs, unstructured_docs)

    for entry in bundle_json["entry"]:
        resource = entry.get("resource", {})
        rtype = resource.get("resourceType", "")
        rid = resource.get("id", "")

        # create a 'structured doc' capturing typed fields:
        sdoc = {
            "doc_id": f"{rtype}-{rid}-structured",
            "doc_type": "structured",
            "resourceType": rtype,
            # Patient
            "patientId": None,
            "patientName": None,
            "patientGender": None,
            "patientDOB": None,
            "patientAddress": None,
            "patientMaritalStatus": None,
            "patientMultipleBirth": None,
            "patientTelecom": None,
            "patientLanguage": None,
            # Condition
            "conditionId": None,
            "conditionCodeText": None,
            "conditionCategory": None,
            "conditionClinicalStatus": None,
            "conditionVerificationStatus": None,
            "conditionOnsetDateTime": None,
            "conditionRecordedDate": None,
            "conditionSeverity": None,
            "conditionNote": None,
            # Observation
            "observationId": None,
            "observationCodeText": None,
            "observationValue": None,
            "observationUnit": None,
            "observationInterpretation": None,
            "observationEffectiveDateTime": None,
            "observationIssued": None,
            "observationReferenceRange": None,
            "observationNote": None,
            # Encounter
            "encounterId": None,
            "encounterStatus": None,
            "encounterClass": None,
            "encounterType": None,
            "encounterReasonCode": None,
            "encounterStart": None,
            "encounterEnd": None,
            "encounterLocation": None,
            "encounterServiceProvider": None,
            "encounterParticipant": None,
            "encounterNote": None,
            # MedicationRequest
            "medRequestId": None,
            "medRequestMedicationDisplay": None,
            "medRequestAuthoredOn": None,
            "medRequestIntent": None,
            "medRequestStatus": None,
            "medRequestPriority": None,
            "medRequestDosageInstruction": None,
            "medRequestDispenseRequest": None,
            "medRequestNote": None,
            # Procedure
            "procedureId": None,
            "procedureCodeText": None,
            "procedureStatus": None,
            "procedurePerformedDateTime": None,
            "procedureFollowUp": None,
            "procedureNote": None,
            # AllergyIntolerance
            "allergyId": None,
            "allergyClinicalStatus": None,
            "allergyVerificationStatus": None,
            "allergyType": None,
            "allergyCategory": None,
            "allergyCriticality": None,
            "allergyCodeText": None,
            "allergyOnsetDateTime": None,
            "allergyNote": None,
            # Practitioner
            "practitionerId": None,
            "practitionerName": None,
            "practitionerGender": None,
            "practitionerSpecialty": None,
            "practitionerAddress": None,
            "practitionerTelecom": None,
            # Organization
            "organizationId": None,
            "organizationName": None,
            "organizationType": None,
            "organizationAddress": None,
            "organizationTelecom": None,
            # We do not store embeddings in structured docs
            # unstructured docs will have "embedding" once we embed them
        }

        # gather unstructured text from various narrative or note fields
        unstructured_text_pieces = []

        # resource.text.div
        div_text = resource.get("text", {}).get("div", "")
        if div_text.strip():
            unstructured_text_pieces.append(div_text)

        # now parse resource-specific logic:
        if rtype == "Patient":
            sdoc["patientId"] = rid
            sdoc["patientGender"] = resource.get("gender")
            sdoc["patientDOB"] = resource.get("birthDate")

            if "name" in resource and len(resource["name"]) > 0:
                n = resource["name"][0]
                family = n.get("family", "")
                given = " ".join(n.get("given", []))
                sdoc["patientName"] = f"{given} {family}".strip()

            # address
            if "address" in resource and len(resource["address"]) > 0:
                addr = resource["address"][0]
                lines = addr.get("line", [])
                city = addr.get("city", "")
                state = addr.get("state", "")
                postal = addr.get("postalCode", "")
                address_str = " ".join(lines + [city, state, postal]).strip()
                sdoc["patientAddress"] = address_str

            # maritalStatus
            if "maritalStatus" in resource:
                ms = resource["maritalStatus"]
                sdoc["patientMaritalStatus"] = ms.get("text") or ms.get("coding", [{}])[
                    0
                ].get("code")

            # multipleBirth
            if "multipleBirthInteger" in resource:
                sdoc["patientMultipleBirth"] = resource["multipleBirthInteger"]
            elif "multipleBirthBoolean" in resource:
                # store as 1 or 0 or keep boolean as integer
                sdoc["patientMultipleBirth"] = (
                    1 if resource["multipleBirthBoolean"] else 0
                )

            # telecom
            if "telecom" in resource:
                telecom_strs = []
                for t in resource["telecom"]:
                    use = t.get("use", "")
                    val = t.get("value", "")
                    telecom_strs.append(f"{use}: {val}")
                if telecom_strs:
                    sdoc["patientTelecom"] = " | ".join(telecom_strs)

            # language
            if "communication" in resource and len(resource["communication"]) > 0:
                # take first
                comm = resource["communication"][0]
                c_lang = comm.get("language", {})
                sdoc["patientLanguage"] = c_lang.get("text") or c_lang.get(
                    "coding", [{}]
                )[0].get("code")

        elif rtype == "Condition":
            sdoc["conditionId"] = rid
            cstatus = resource.get("clinicalStatus", {})
            sdoc["conditionClinicalStatus"] = cstatus.get("text") or cstatus.get(
                "coding", [{}]
            )[0].get("code")

            vstatus = resource.get("verificationStatus", {})
            sdoc["conditionVerificationStatus"] = vstatus.get("text") or vstatus.get(
                "coding", [{}]
            )[0].get("code")

            # category
            if "category" in resource and len(resource["category"]) > 0:
                cat = resource["category"][0]
                sdoc["conditionCategory"] = cat.get("text") or cat.get("coding", [{}])[
                    0
                ].get("code")

            # severity
            if "severity" in resource:
                sev = resource["severity"]
                sdoc["conditionSeverity"] = sev.get("text") or sev.get("coding", [{}])[
                    0
                ].get("code")

            code_field = resource.get("code", {})
            ctext = code_field.get("text")
            if not ctext and "coding" in code_field and len(code_field["coding"]) > 0:
                ctext = code_field["coding"][0].get("display", "")
            sdoc["conditionCodeText"] = ctext

            sdoc["conditionOnsetDateTime"] = resource.get("onsetDateTime")
            sdoc["conditionRecordedDate"] = resource.get("recordedDate")

            # Condition note => unstructured
            if "note" in resource:
                all_notes = []
                for note_item in resource["note"]:
                    note_txt = note_item.get("text", "").strip()
                    if note_txt:
                        all_notes.append(note_txt)
                if all_notes:
                    sdoc["conditionNote"] = " | ".join(all_notes)
                    unstructured_text_pieces.extend(all_notes)

        elif rtype == "Observation":
            sdoc["observationId"] = rid
            code_info = resource.get("code", {})
            obs_code_text = code_info.get("text")
            if (
                not obs_code_text
                and "coding" in code_info
                and len(code_info["coding"]) > 0
            ):
                obs_code_text = code_info["coding"][0].get("display", "")
            sdoc["observationCodeText"] = obs_code_text

            # quantity
            if "valueQuantity" in resource:
                val = resource["valueQuantity"].get("value", "")
                un = resource["valueQuantity"].get("unit", "")
                sdoc["observationValue"] = str(val)
                sdoc["observationUnit"] = un

            if "interpretation" in resource and len(resource["interpretation"]) > 0:
                first_interp = resource["interpretation"][0]
                sdoc["observationInterpretation"] = first_interp.get(
                    "text"
                ) or first_interp.get("coding", [{}])[0].get("code")

            sdoc["observationEffectiveDateTime"] = resource.get("effectiveDateTime")
            sdoc["observationIssued"] = resource.get("issued")

            if "referenceRange" in resource and len(resource["referenceRange"]) > 0:
                # store them as text
                range_list = []
                for rr in resource["referenceRange"]:
                    low = rr.get("low", {}).get("value", "")
                    high = rr.get("high", {}).get("value", "")
                    range_str = f"Low: {low}, High: {high}".strip()
                    range_list.append(range_str)
                if range_list:
                    sdoc["observationReferenceRange"] = " ; ".join(range_list)

            if "note" in resource:
                obs_notes = []
                for nt in resource["note"]:
                    t = nt.get("text", "").strip()
                    if t:
                        obs_notes.append(t)
                if obs_notes:
                    sdoc["observationNote"] = " | ".join(obs_notes)
                    unstructured_text_pieces.extend(obs_notes)

        elif rtype == "Encounter":
            sdoc["encounterId"] = rid
            sdoc["encounterStatus"] = resource.get("status")
            sdoc["encounterClass"] = resource.get("class", {}).get("code")
            if "type" in resource and len(resource["type"]) > 0:
                t = resource["type"][0]
                sdoc["encounterType"] = t.get("text") or t.get("coding", [{}])[0].get(
                    "display"
                )

            if "reasonCode" in resource and len(resource["reasonCode"]) > 0:
                rc = resource["reasonCode"][0]
                sdoc["encounterReasonCode"] = rc.get("text") or rc.get("coding", [{}])[
                    0
                ].get("display")

            period = resource.get("period", {})
            sdoc["encounterStart"] = period.get("start")
            sdoc["encounterEnd"] = period.get("end")

            if "location" in resource and len(resource["location"]) > 0:
                first_loc = resource["location"][0]
                loc_display = first_loc.get("location", {}).get("display", "")
                sdoc["encounterLocation"] = loc_display

            if "serviceProvider" in resource:
                sp = resource["serviceProvider"]
                # might be a reference
                sdoc["encounterServiceProvider"] = sp.get("reference")

            if "participant" in resource and len(resource["participant"]) > 0:
                participants = []
                for p in resource["participant"]:
                    ip = p.get("individual", {})
                    p_disp = ip.get("display", "")
                    participants.append(p_disp)
                if participants:
                    sdoc["encounterParticipant"] = " | ".join(participants)

            if "note" in resource:
                enc_notes = []
                for note_item in resource["note"]:
                    note_txt = note_item.get("text", "").strip()
                    if note_txt:
                        enc_notes.append(note_txt)
                if enc_notes:
                    sdoc["encounterNote"] = " | ".join(enc_notes)
                    unstructured_text_pieces.extend(enc_notes)

        elif rtype == "MedicationRequest":
            sdoc["medRequestId"] = rid
            sdoc["medRequestIntent"] = resource.get("intent")
            sdoc["medRequestStatus"] = resource.get("status")
            sdoc["medRequestPriority"] = resource.get("priority")
            sdoc["medRequestAuthoredOn"] = resource.get("authoredOn")

            med_code = resource.get("medicationCodeableConcept", {})
            med_text = med_code.get("text")
            if (not med_text) and "coding" in med_code and len(med_code["coding"]) > 0:
                med_text = med_code["coding"][0].get("display", "")
            sdoc["medRequestMedicationDisplay"] = med_text

            # dosageInstruction
            if (
                "dosageInstruction" in resource
                and len(resource["dosageInstruction"]) > 0
            ):
                d_strs = []
                for di in resource["dosageInstruction"]:
                    txt = di.get("text", "")
                    d_strs.append(txt)
                if d_strs:
                    sdoc["medRequestDosageInstruction"] = " | ".join(d_strs)

            # dispenseRequest
            if "dispenseRequest" in resource:
                dr = resource["dispenseRequest"]
                sdoc["medRequestDispenseRequest"] = json.dumps(dr)

            # note
            if "note" in resource:
                mr_notes = []
                for n in resource["note"]:
                    txt = n.get("text", "").strip()
                    if txt:
                        mr_notes.append(txt)
                if mr_notes:
                    sdoc["medRequestNote"] = " | ".join(mr_notes)
                    unstructured_text_pieces.extend(mr_notes)

        elif rtype == "Procedure":
            sdoc["procedureId"] = rid
            sdoc["procedureStatus"] = resource.get("status")
            c = resource.get("code", {})
            c_text = c.get("text") or c.get("coding", [{}])[0].get("display")
            sdoc["procedureCodeText"] = c_text
            if "performedDateTime" in resource:
                sdoc["procedurePerformedDateTime"] = resource["performedDateTime"]
            if "followUp" in resource and len(resource["followUp"]) > 0:
                fu_arr = []
                for f in resource["followUp"]:
                    fu_txt = f.get("text", "")
                    fu_arr.append(fu_txt)
                if fu_arr:
                    sdoc["procedureFollowUp"] = " | ".join(fu_arr)

            if "note" in resource:
                proc_notes = []
                for n in resource["note"]:
                    t = n.get("text", "").strip()
                    if t:
                        proc_notes.append(t)
                if proc_notes:
                    sdoc["procedureNote"] = " | ".join(proc_notes)
                    unstructured_text_pieces.extend(proc_notes)

        elif rtype == "AllergyIntolerance":
            sdoc["allergyId"] = rid
            sdoc["allergyClinicalStatus"] = resource.get("clinicalStatus", {}).get(
                "text"
            )
            sdoc["allergyVerificationStatus"] = resource.get(
                "verificationStatus", {}
            ).get("text")
            sdoc["allergyType"] = resource.get("type")
            if "category" in resource and len(resource["category"]) > 0:
                sdoc["allergyCategory"] = resource["category"][0]
            sdoc["allergyCriticality"] = resource.get("criticality")

            code_field = resource.get("code", {})
            all_text = code_field.get("text")
            if (
                not all_text
                and "coding" in code_field
                and len(code_field["coding"]) > 0
            ):
                all_text = code_field["coding"][0].get("display", "")
            sdoc["allergyCodeText"] = all_text

            sdoc["allergyOnsetDateTime"] = resource.get("onsetDateTime")

            if "note" in resource:
                a_notes = []
                for note in resource["note"]:
                    ntxt = note.get("text", "").strip()
                    if ntxt:
                        a_notes.append(ntxt)
                if a_notes:
                    sdoc["allergyNote"] = " | ".join(a_notes)
                    unstructured_text_pieces.extend(a_notes)

        elif rtype == "Practitioner":
            sdoc["practitionerId"] = rid
            if "name" in resource and len(resource["name"]) > 0:
                nm = resource["name"][0]
                p_fam = nm.get("family", "")
                p_giv = " ".join(nm.get("given", []))
                sdoc["practitionerName"] = f"{p_giv} {p_fam}".strip()
            sdoc["practitionerGender"] = resource.get("gender")

            if "qualification" in resource and len(resource["qualification"]) > 0:
                # interpret "qualification" as specialty
                q = resource["qualification"][0]
                sdoc["practitionerSpecialty"] = q.get("code", {}).get("text")

            if "address" in resource and len(resource["address"]) > 0:
                adr = resource["address"][0]
                lines = adr.get("line", [])
                city = adr.get("city", "")
                state = adr.get("state", "")
                postal = adr.get("postalCode", "")
                paddr_str = " ".join(lines + [city, state, postal]).strip()
                sdoc["practitionerAddress"] = paddr_str

            if "telecom" in resource:
                tele_arr = []
                for t in resource["telecom"]:
                    use = t.get("use", "")
                    val = t.get("value", "")
                    tele_arr.append(f"{use}: {val}")
                if tele_arr:
                    sdoc["practitionerTelecom"] = " | ".join(tele_arr)

        elif rtype == "Organization":
            sdoc["organizationId"] = rid
            sdoc["organizationName"] = resource.get("name")
            if "type" in resource and len(resource["type"]) > 0:
                t0 = resource["type"][0]
                sdoc["organizationType"] = t0.get("text") or t0.get("coding", [{}])[
                    0
                ].get("code")

            if "address" in resource and len(resource["address"]) > 0:
                org_adr = resource["address"][0]
                lines = org_adr.get("line", [])
                city = org_adr.get("city", "")
                state = org_adr.get("state", "")
                postal = org_adr.get("postalCode", "")
                org_addr_str = " ".join(lines + [city, state, postal]).strip()
                sdoc["organizationAddress"] = org_addr_str

            if "telecom" in resource:
                otele_arr = []
                for t in resource["telecom"]:
                    use = t.get("use", "")
                    val = t.get("value", "")
                    otele_arr.append(f"{use}: {val}")
                if otele_arr:
                    sdoc["organizationTelecom"] = " | ".join(otele_arr)

        # more resource types can be added here later for future uses

        # add the structured doc
        structured_docs.append(sdoc)

        # If we have unstructured text, chunk it out for embedding
        # Because these are notes, we set doc_type="unstructured"
        # Each chunk becomes a separate doc due to chunking
        if unstructured_text_pieces:
            combined_text = "\n".join(unstructured_text_pieces).strip()
            if not combined_text:
                continue

            # Possibly chunk if large
            text_chunks = chunk_text(combined_text, chunk_size=CHUNK_SIZE)
            for c_i, chunk_str in enumerate(text_chunks):
                udoc = {
                    "doc_id": f"{rtype}-{rid}-unstructured-{c_i}",
                    "doc_type": "unstructured",
                    "resourceType": rtype,
                    # We'll store the text in "unstructuredText" field
                    "unstructuredText": chunk_str,
                }
                unstructured_docs.append(udoc)

    return (structured_docs, unstructured_docs)


async def store_fhir_docs_in_opensearch(
    structured_docs: List[Dict],
    unstructured_docs: List[Dict],
    client: OpenSearch,
    index_name: str,
) -> None:
    """
    1) Bulk index the structured docs (no embeddings).
    2) For unstructured docs, embed them => store text + embedding.
    This aligns with the new index structure that has one text field + embedding field.
    """
    if not client:
        print("[store_fhir_docs_in_opensearch] No OS client.")
        return

    # Bulk index structured docs
    bulk_actions_struct = []
    for doc in structured_docs:
        # We'll upsert them into the index
        # (If you want, you can differentiate by doc_type or resourceType here.)
        action_struct = {
            "_op_type": "index",
            "_index": index_name,
            "_id": doc["doc_id"],
            "_source": doc,
        }
        bulk_actions_struct.append(action_struct)

    if bulk_actions_struct:
        s_success, s_errors = bulk(client, bulk_actions_struct)
        print(f"[FHIR] Indexed {s_success} structured docs. Errors: {s_errors}")

    if not unstructured_docs:
        return

    # We'll embed the unstructured text, then store each doc with
    # "unstructuredText" plus the "embedding" field.
    un_texts = [d["unstructuredText"] for d in unstructured_docs]
    embeddings = await embed_texts_in_batches(un_texts, batch_size=BATCH_SIZE)

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_normed = embeddings / (norms + 1e-9)

    bulk_actions_unstruct = []
    for i, docu in enumerate(unstructured_docs):
        emb_vec = embeddings_normed[i]
        docu["embedding"] = emb_vec.tolist()  # store as float array
        # We'll store docu in index, using "doc_id" as _id
        action_unstruct = {
            "_op_type": "index",
            "_index": index_name,
            "_id": docu["doc_id"],
            "_source": docu,
        }
        bulk_actions_unstruct.append(action_unstruct)

        if len(bulk_actions_unstruct) >= BATCH_SIZE:
            u_success, u_errors = bulk(client, bulk_actions_unstruct)
            print(f"[FHIR] Indexed {u_success} unstructured docs. Errors: {u_errors}")
            bulk_actions_unstruct = []

    if bulk_actions_unstruct:
        u_success, u_errors = bulk(client, bulk_actions_unstruct)
        print(f"[FHIR] Indexed {u_success} unstructured docs. Errors: {u_errors}")


class OpenSearchIndexer:
    """
    Index documents with embeddings into OpenSearch with a hybrid search approach
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

    def exact_match_search(
        self, query_text: str, k: int = 3
    ) -> List[Tuple[Dict[str, str], float]]:
        if not self.client or not query_text.strip():
            return []

        query_body = {"size": k, "query": {"match": {"text": query_text}}}
        try:
            resp = self.client.search(index=OPENSEARCH_INDEX_NAME, body=query_body)
            hits = resp["hits"]["hits"]
            results = []
            for h in hits:
                doc_score = h["_score"]
                doc_source = h["_source"]
                results.append((doc_source, float(doc_score)))

            return results
        except Exception as e:
            print(f"[OpenSearchIndexer - Exact Match] Search error: {e}")
            return []

    def semantic_search(
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

            print(
                f"[OpenSearchIndexer - Semantic Search] Found {len(results)} relevant results."
            )
            return results
        except Exception as e:
            print(f"[OpenSearchIndexer - Semantic Search] Search error: {e}")
            return []

    def hybrid_search(
        self, query_text: str, query_emb: np.ndarray, k: int = 3
    ) -> List[Tuple[Dict[str, str], float]]:
        if not self.client or not query_text.strip() or not query_emb:
            return []

        q_norm = np.linalg.norm(query_emb, axis=1, keepdims=True)
        query_emb = query_emb / (q_norm + 1e-9)
        vector = query_emb[0].tolist()
        query_body = {
            "size": k,
            "query": {
                "bool": {
                    "should": [
                        {"match": {"text": query_text}},
                        {"knn": {"embedding": {"vector": vector, "k": k}}},
                    ],
                    "minimum_should_match": 1,
                }
            },
        }
        try:
            resp = os_client.search(index=OPENSEARCH_INDEX_NAME, body=query_body)
            hits = resp["hits"]["hits"]
            results = []
            for h in hits:
                doc_score = h["_score"]
                doc_source = h["_source"]
                results.append((doc_source, float(doc_score)))

            return results
        except Exception as e:
            print(f"[OpenSearchIndexer - Hybrid Search] Search error: {e}")
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


# ----------------------------------------------------------------------
# ZeroShotQueryClassifier
# ---------------------------------------------------------
class ZeroShotQueryClassifier:
    """
    A zero-shot classification approach for dynamic labeling of queries like:
    - "SEMANTIC" => Vector-based search
    - "KEYWORD"  => Exact / BM25 search
    - "HYBRID"   => Combination
    """

    def __init__(self):
        self.candidate_labels = ["SEMANTIC", "KEYWORD", "HYBRID"]
        # Create pipeline
        self._pipe = pipeline(
            task="zero-shot-classification",
            model=QUERY_INTENT_CLASSIFICATION_MODEL,
            tokenizer=QUERY_INTENT_CLASSIFICATION_MODEL,
        )
        # Thread executor to avoid blocking
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def _sync_classify(self, query: str) -> str:
        classification = self._pipe(
            sequences=query, candidate_labels=self.candidate_labels, multi_label=False
        )
        return classification["labels"][0]

    async def classify_intent(self, query: str) -> str:
        loop = asyncio.get_running_loop()
        label = await loop.run_in_executor(self._executor, self._sync_classify, query)
        return label


# a single global classifier created
intent_classifier = ZeroShotQueryClassifier()


# ==============================================================================
# RASS engine logic
# ==============================================================================
class RASSEngine:
    """
    RASS engine that:
      - Classifies query intent using DistilBERT
      - Uses appropriate search method (exact, semantic, or hybrid)
      - Uses BlueHive for final response
      - Uses Redis for caching w/ TTL
      - Saves user queries & answers in Postgres (via Prisma)
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
            print("[RASSEngine] No OpenSearchIndexer => cannot build embeddings.")
            return

        if self.os_indexer.has_any_data():
            print("[RASSEngine] OpenSearch already has data. Skipping embedding.")
            return

        print("[RASSEngine] Building embeddings from scratch...")
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
            print("[RASSEngine] No text found in directory. Exiting.")
            return

        print(f"[RASSEngine] Generating embeddings for {len(all_docs)} chunks...")
        chunk_texts = [d["text"] for d in all_docs]

        embs = await embed_texts_in_batches(chunk_texts, batch_size=64)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.os_indexer.add_embeddings, embs, all_docs)
        print("[RASSEngine] Finished embedding & indexing data in OpenSearch.")

    async def ingest_fhir_directory(self, fhir_dir: str) -> None:
        """
        Enumerate all FHIR .json files in fhir_dir, parse them, and store in OS.
        This is production-level logic for hybrid FHIR ingestion.
        """
        if not self.os_indexer:
            print("[RASSEngine] No OS indexer => cannot ingest FHIR data.")
            return

        all_files = [f for f in os.listdir(fhir_dir) if f.endswith(".json")]
        if not all_files:
            print(f"[RASSEngine] No .json files found in {fhir_dir}.")
            return

        for fname in all_files:
            path = os.path.join(fhir_dir, fname)
            print(f"[RASSEngine] Processing FHIR file: {path}")

            try:
                with open(path, "r", encoding="utf-8") as f:
                    bundle_json = json.load(f)
            except UnicodeDecodeError:
                with open(path, "r", encoding="latin-1") as f:
                    bundle_json = json.load(f)
            except Exception as e:
                print(f"[ERROR] Failed reading {path}: {e}")
                continue

            # parse the entire FHIR bundle
            structured_docs, unstructured_docs = parse_fhir_bundle(bundle_json)
            # store them in OS
            await store_fhir_docs_in_opensearch(
                structured_docs,
                unstructured_docs,
                self.os_indexer.client,
                self.os_indexer.index_name,
            )

        print("[RASSEngine] Completed ingesting FHIR data from directory.")

    async def ask(
        self,
        query: str,
        user_id: str,
        chat_id: str,
        top_k: int = 3,
    ) -> str:
        # non empty validation check
        if not query.strip():
            return "[ERROR] Empty query."

        if not chat_id:
            return "[ERROR] Incorrect account/chat details!"

        # verify user owns chat
        chat = await db.chat.find_unique(where={"id": chat_id}, include={"user": True})

        if not chat or chat.userId != user_id:
            raise HTTPException(
                status_code=403, detail="Chat not found or unauthorized"
            )

        # DistilBERT zero shot classification => "SEMANTIC", "KEYWORD", or "HYBRID"
        intent = await intent_classifier.classify_intent(query)
        print(f"[Debug] Query Intent = {intent}")

        # fetch last 'MAX_CHAT_HISTORY' messages for context
        messages = await db.message.find_many(
            where={"chatId": chat_id},
            order={"createdAt": "desc"},
            take=MAX_CHAT_HISTORY,
        )

        # build chat history
        # newest message is first => reverse to chronological
        messages.reverse()  # asc order
        chat_history_str = ""
        for m in messages:
            role_label = "User" if m.role == "user" else "AI"
            chat_history_str += f"{role_label}: {m.content}\n"

        # embed query & check cache
        query_emb = await embed_query(query)
        cached_resp = await lfu_cache_get(query_emb)
        if cached_resp:
            # store new user message + cached answer
            await db.message.create(
                data={"chatId": chat_id, "role": "user", "content": query}
            )
            await db.message.create(
                data={"chatId": chat_id, "role": "assistant", "content": cached_resp}
            )
            print("[CACHE-HIT] returning cached result.")
            return cached_resp

        if not self.os_indexer:
            return "[ERROR] No OS indexer"

        # OpenSearch Retrieval
        if intent == "SEMANTIC":
            partial_results = self.os_indexer.semantic_search(query_emb, k=top_k)
        elif intent == "KEYWORD":
            partial_results = self.os_indexer.exact_match_search(query, k=top_k)
        else:
            partial_results = self.os_indexer.hybrid_search(query, query_emb, k=top_k)

        context_map = {}
        for doc_dict, _score in partial_results:
            doc_id = doc_dict.get("doc_id", "UNKNOWN")

            if doc_dict.get("doc_type", "structured") == "unstructured":
                # just show the unstructuredText - improvement may be planned later
                raw_text = doc_dict.get("unstructuredText", "")
                snippet = f"[Unstructured Text]: {raw_text}"
            else:
                # gather all non-empty fields for "structured" docs
                snippet_pieces = []

                # patient
                if doc_dict.get("patientId"):
                    snippet_pieces.append(f"patientId={doc_dict['patientId']}")
                if doc_dict.get("patientName"):
                    snippet_pieces.append(f"patientName={doc_dict['patientName']}")
                if doc_dict.get("patientGender"):
                    snippet_pieces.append(f"patientGender={doc_dict['patientGender']}")
                if doc_dict.get("patientDOB"):
                    snippet_pieces.append(f"patientDOB={doc_dict['patientDOB']}")
                if doc_dict.get("patientAddress"):
                    snippet_pieces.append(
                        f"patientAddress={doc_dict['patientAddress']}"
                    )
                if doc_dict.get("patientMaritalStatus"):
                    snippet_pieces.append(
                        f"patientMaritalStatus={doc_dict['patientMaritalStatus']}"
                    )
                if doc_dict.get("patientMultipleBirth") is not None:
                    snippet_pieces.append(
                        f"patientMultipleBirth={doc_dict['patientMultipleBirth']}"
                    )
                if doc_dict.get("patientTelecom"):
                    snippet_pieces.append(
                        f"patientTelecom={doc_dict['patientTelecom']}"
                    )
                if doc_dict.get("patientLanguage"):
                    snippet_pieces.append(
                        f"patientLanguage={doc_dict['patientLanguage']}"
                    )

                # Condition
                if doc_dict.get("conditionId"):
                    snippet_pieces.append(f"conditionId={doc_dict['conditionId']}")
                if doc_dict.get("conditionCodeText"):
                    snippet_pieces.append(
                        f"conditionCodeText={doc_dict['conditionCodeText']}"
                    )
                if doc_dict.get("conditionCategory"):
                    snippet_pieces.append(
                        f"conditionCategory={doc_dict['conditionCategory']}"
                    )
                if doc_dict.get("conditionClinicalStatus"):
                    snippet_pieces.append(
                        f"conditionClinicalStatus={doc_dict['conditionClinicalStatus']}"
                    )
                if doc_dict.get("conditionVerificationStatus"):
                    snippet_pieces.append(
                        f"conditionVerificationStatus={doc_dict['conditionVerificationStatus']}"
                    )
                if doc_dict.get("conditionOnsetDateTime"):
                    snippet_pieces.append(
                        f"conditionOnsetDateTime={doc_dict['conditionOnsetDateTime']}"
                    )
                if doc_dict.get("conditionRecordedDate"):
                    snippet_pieces.append(
                        f"conditionRecordedDate={doc_dict['conditionRecordedDate']}"
                    )
                if doc_dict.get("conditionSeverity"):
                    snippet_pieces.append(
                        f"conditionSeverity={doc_dict['conditionSeverity']}"
                    )
                if doc_dict.get("conditionNote"):
                    snippet_pieces.append(f"conditionNote={doc_dict['conditionNote']}")

                # Observation
                if doc_dict.get("observationId"):
                    snippet_pieces.append(f"observationId={doc_dict['observationId']}")
                if doc_dict.get("observationCodeText"):
                    snippet_pieces.append(
                        f"observationCodeText={doc_dict['observationCodeText']}"
                    )
                if doc_dict.get("observationValue"):
                    snippet_pieces.append(
                        f"observationValue={doc_dict['observationValue']}"
                    )
                if doc_dict.get("observationUnit"):
                    snippet_pieces.append(
                        f"observationUnit={doc_dict['observationUnit']}"
                    )
                if doc_dict.get("observationInterpretation"):
                    snippet_pieces.append(
                        f"observationInterpretation={doc_dict['observationInterpretation']}"
                    )
                if doc_dict.get("observationEffectiveDateTime"):
                    snippet_pieces.append(
                        f"observationEffectiveDateTime={doc_dict['observationEffectiveDateTime']}"
                    )
                if doc_dict.get("observationIssued"):
                    snippet_pieces.append(
                        f"observationIssued={doc_dict['observationIssued']}"
                    )
                if doc_dict.get("observationReferenceRange"):
                    snippet_pieces.append(
                        f"observationReferenceRange={doc_dict['observationReferenceRange']}"
                    )
                if doc_dict.get("observationNote"):
                    snippet_pieces.append(
                        f"observationNote={doc_dict['observationNote']}"
                    )

                # Encounter
                if doc_dict.get("encounterId"):
                    snippet_pieces.append(f"encounterId={doc_dict['encounterId']}")
                if doc_dict.get("encounterStatus"):
                    snippet_pieces.append(
                        f"encounterStatus={doc_dict['encounterStatus']}"
                    )
                if doc_dict.get("encounterClass"):
                    snippet_pieces.append(
                        f"encounterClass={doc_dict['encounterClass']}"
                    )
                if doc_dict.get("encounterType"):
                    snippet_pieces.append(f"encounterType={doc_dict['encounterType']}")
                if doc_dict.get("encounterReasonCode"):
                    snippet_pieces.append(
                        f"encounterReasonCode={doc_dict['encounterReasonCode']}"
                    )
                if doc_dict.get("encounterStart"):
                    snippet_pieces.append(
                        f"encounterStart={doc_dict['encounterStart']}"
                    )
                if doc_dict.get("encounterEnd"):
                    snippet_pieces.append(f"encounterEnd={doc_dict['encounterEnd']}")
                if doc_dict.get("encounterLocation"):
                    snippet_pieces.append(
                        f"encounterLocation={doc_dict['encounterLocation']}"
                    )
                if doc_dict.get("encounterServiceProvider"):
                    snippet_pieces.append(
                        f"encounterServiceProvider={doc_dict['encounterServiceProvider']}"
                    )
                if doc_dict.get("encounterParticipant"):
                    snippet_pieces.append(
                        f"encounterParticipant={doc_dict['encounterParticipant']}"
                    )
                if doc_dict.get("encounterNote"):
                    snippet_pieces.append(f"encounterNote={doc_dict['encounterNote']}")

                # MedicationRequest
                if doc_dict.get("medRequestId"):
                    snippet_pieces.append(f"medRequestId={doc_dict['medRequestId']}")
                if doc_dict.get("medRequestMedicationDisplay"):
                    snippet_pieces.append(
                        f"medRequestMedicationDisplay={doc_dict['medRequestMedicationDisplay']}"
                    )
                if doc_dict.get("medRequestAuthoredOn"):
                    snippet_pieces.append(
                        f"medRequestAuthoredOn={doc_dict['medRequestAuthoredOn']}"
                    )
                if doc_dict.get("medRequestIntent"):
                    snippet_pieces.append(
                        f"medRequestIntent={doc_dict['medRequestIntent']}"
                    )
                if doc_dict.get("medRequestStatus"):
                    snippet_pieces.append(
                        f"medRequestStatus={doc_dict['medRequestStatus']}"
                    )
                if doc_dict.get("medRequestPriority"):
                    snippet_pieces.append(
                        f"medRequestPriority={doc_dict['medRequestPriority']}"
                    )
                if doc_dict.get("medRequestDosageInstruction"):
                    snippet_pieces.append(
                        f"medRequestDosageInstruction={doc_dict['medRequestDosageInstruction']}"
                    )
                if doc_dict.get("medRequestDispenseRequest"):
                    snippet_pieces.append(
                        f"medRequestDispenseRequest={doc_dict['medRequestDispenseRequest']}"
                    )
                if doc_dict.get("medRequestNote"):
                    snippet_pieces.append(
                        f"medRequestNote={doc_dict['medRequestNote']}"
                    )

                # Procedure
                if doc_dict.get("procedureId"):
                    snippet_pieces.append(f"procedureId={doc_dict['procedureId']}")
                if doc_dict.get("procedureCodeText"):
                    snippet_pieces.append(
                        f"procedureCodeText={doc_dict['procedureCodeText']}"
                    )
                if doc_dict.get("procedureStatus"):
                    snippet_pieces.append(
                        f"procedureStatus={doc_dict['procedureStatus']}"
                    )
                if doc_dict.get("procedurePerformedDateTime"):
                    snippet_pieces.append(
                        f"procedurePerformedDateTime={doc_dict['procedurePerformedDateTime']}"
                    )
                if doc_dict.get("procedureFollowUp"):
                    snippet_pieces.append(
                        f"procedureFollowUp={doc_dict['procedureFollowUp']}"
                    )
                if doc_dict.get("procedureNote"):
                    snippet_pieces.append(f"procedureNote={doc_dict['procedureNote']}")

                # AllergyIntolerance
                if doc_dict.get("allergyId"):
                    snippet_pieces.append(f"allergyId={doc_dict['allergyId']}")
                if doc_dict.get("allergyClinicalStatus"):
                    snippet_pieces.append(
                        f"allergyClinicalStatus={doc_dict['allergyClinicalStatus']}"
                    )
                if doc_dict.get("allergyVerificationStatus"):
                    snippet_pieces.append(
                        f"allergyVerificationStatus={doc_dict['allergyVerificationStatus']}"
                    )
                if doc_dict.get("allergyType"):
                    snippet_pieces.append(f"allergyType={doc_dict['allergyType']}")
                if doc_dict.get("allergyCategory"):
                    snippet_pieces.append(
                        f"allergyCategory={doc_dict['allergyCategory']}"
                    )
                if doc_dict.get("allergyCriticality"):
                    snippet_pieces.append(
                        f"allergyCriticality={doc_dict['allergyCriticality']}"
                    )
                if doc_dict.get("allergyCodeText"):
                    snippet_pieces.append(
                        f"allergyCodeText={doc_dict['allergyCodeText']}"
                    )
                if doc_dict.get("allergyOnsetDateTime"):
                    snippet_pieces.append(
                        f"allergyOnsetDateTime={doc_dict['allergyOnsetDateTime']}"
                    )
                if doc_dict.get("allergyNote"):
                    snippet_pieces.append(f"allergyNote={doc_dict['allergyNote']}")

                # Practitioner
                if doc_dict.get("practitionerId"):
                    snippet_pieces.append(
                        f"practitionerId={doc_dict['practitionerId']}"
                    )
                if doc_dict.get("practitionerName"):
                    snippet_pieces.append(
                        f"practitionerName={doc_dict['practitionerName']}"
                    )
                if doc_dict.get("practitionerGender"):
                    snippet_pieces.append(
                        f"practitionerGender={doc_dict['practitionerGender']}"
                    )
                if doc_dict.get("practitionerSpecialty"):
                    snippet_pieces.append(
                        f"practitionerSpecialty={doc_dict['practitionerSpecialty']}"
                    )
                if doc_dict.get("practitionerAddress"):
                    snippet_pieces.append(
                        f"practitionerAddress={doc_dict['practitionerAddress']}"
                    )
                if doc_dict.get("practitionerTelecom"):
                    snippet_pieces.append(
                        f"practitionerTelecom={doc_dict['practitionerTelecom']}"
                    )

                # Organization
                if doc_dict.get("organizationId"):
                    snippet_pieces.append(
                        f"organizationId={doc_dict['organizationId']}"
                    )
                if doc_dict.get("organizationName"):
                    snippet_pieces.append(
                        f"organizationName={doc_dict['organizationName']}"
                    )
                if doc_dict.get("organizationType"):
                    snippet_pieces.append(
                        f"organizationType={doc_dict['organizationType']}"
                    )
                if doc_dict.get("organizationAddress"):
                    snippet_pieces.append(
                        f"organizationAddress={doc_dict['organizationAddress']}"
                    )
                if doc_dict.get("organizationTelecom"):
                    snippet_pieces.append(
                        f"organizationTelecom={doc_dict['organizationTelecom']}"
                    )

                snippet = "[Structured Resource] " + " | ".join(snippet_pieces)

            if doc_id not in context_map:
                context_map[doc_id] = snippet
            else:
                context_map[doc_id] += "\n" + snippet

        # Build a short context
        context_text = ""
        for doc_id, doc_content in context_map.items():
            print(doc_id)
            context_text += f"--- Document ID: {doc_id} ---\n{doc_content}\n\n"

        # build the prompt with chat history and current query
        system_msg = (
            "You are a helpful AI assistant chatbot. You must follow these rules:\n"
            "1) Always cite document IDs from the context exactly as 'Document XYZ' without any file extensions like '.txt'.\n"
            "2) For every answer generated, there should be a reference or citation of the IDs of the documents from which the answer information was extracted at the end of the answer!\n"
            "3) If the context does not relate to the query, say 'I lack the context to answer your question.' For example, if the query is about gene mutations but the context is about climate change, acknowledge the mismatch and do not answer.\n"
            "4) Never ever give responses based on your own knowledge of the user query. Use ONLY the provided context + chat history to extract information relevant to the question. You should not answer without document ID references from which the information was extracted.\n"
            "5) If you lack context, then say so.\n"
            "6) Do not add chain-of-thought.\n"
            # "7) Answer in at most 4 sentences.\n"
        )
        final_prompt = (
            f"Chat History:\n{chat_history_str}\n\n"
            f"User Query:\n{query}\n\n"
            f"Context:\n{context_text}\n"
            "--- End of context ---\n\n"
            "Provide your concise answer now."
        )

        answer = await bluehive_generate_text(
            prompt=final_prompt, system_msg=system_msg
        )
        if not answer:
            return "[Error] No response was generated."

        # store new user message + AI response
        await db.message.create(
            data={"chatId": chat_id, "role": "user", "content": query}
        )
        await db.message.create(
            data={"chatId": chat_id, "role": "assistant", "content": answer}
        )
        # cache the resp
        await lfu_cache_put(query_emb, answer)
        return answer


# ==============================================================================
# FastAPI Application Setup
# ==============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Handles database connection lifecycle
    await db.connect()
    print("[Lifespan] Initializing RASSEngine...")
    global rass_engine
    rass_engine = RASSEngine()

    await rass_engine.ingest_fhir_directory(EMB_DIR)
    print("[Lifespan] RASSEngine is ready.")
    yield
    print("[Lifespan] Server is shutting down...")
    await db.disconnect()
    await close_redis()


app = FastAPI(
    title="RASS Engine - /ask Query Microservice",
    version="1.0.0",
    lifespan=lifespan,
)

app.router.lifespan_context = lifespan


def get_rass_engine() -> RASSEngine:
    return rass_engine


@app.post("/ask")
async def ask_route(payload: dict = Body(...)):
    """
    RASS endpoint:
      1) user_id, chat_id, query => verify user owns chat
      2) fetch last 10 messages for context
      3) embed query => check redis => if new
      4) retrieve from OpenSearch => build context
      5) call BlueHive => store new user query + new response in DB
    """
    query: str = payload.get("query", "")
    user_id = payload.get("user_id", "")
    chat_id = payload.get("chat_id", "")
    top_k = int(payload.get("top_k", 3))

    if not user_id or not chat_id or not query.strip():
        raise HTTPException(status_code=400, detail="Provide user_id, chat_id, query")

    print(
        f"[Debug] query = {query}, user_id={user_id}, chat_id={chat_id}, top_k = {top_k}"
    )

    rass_engine = get_rass_engine()
    if not rass_engine:
        return {"error": "RASS engine not initialized"}

    answer = await rass_engine.ask(query, user_id, chat_id, top_k)
    return {"query": query, "answer": answer}


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
    WebSocket endpoint for streaming answers token-by-token.
    Expects JSON from the client {'query':str, 'user_id':str, 'chat_id':str, 'top_k':int}
    """
    await websocket.accept()
    try:
        # Receive & parse JSON data
        data_str = await websocket.receive_text()
        data = json.loads(data_str)

        user_id: str = data.get("user_id", "")
        chat_id: str = data.get("chat_id", "")
        query: str = data.get("query", "")
        top_k: int = int(data.get("top_k", 3))

        # validate required fields
        if not query.strip() or not user_id or not chat_id:
            await websocket.send_text(
                json.dumps({"error": "Missing required parameters."})
            )
            await websocket.close()
            return

        print(
            f"[WebSocket Debug] user_id={user_id}, chat_id={chat_id}, query={query}, top_k={top_k}"
        )

        # RASS engine instance init
        rass_engine = get_rass_engine()
        if not rass_engine:
            await websocket.send_text(
                json.dumps({"error": "RASS engine not initialized"})
            )
            await websocket.close()
            return

        # verify user owns chat
        chat = await db.chat.find_unique(where={"id": chat_id}, include={"user": True})
        if not chat or chat.userId != user_id:
            await websocket.send_text(
                json.dumps({"error": "Chat not found or unauthorized access."})
            )
            await websocket.close()
            return

        # Fetch chat history
        messages = await db.message.find_many(
            where={"chatId": chat_id},
            order={"createdAt": "desc"},
            take=MAX_CHAT_HISTORY,
        )

        messages.reverse()  # asc order
        chat_history_str = ""
        for m in messages:
            role_label = "User" if m.role == "user" else "AI"
            chat_history_str += f"{role_label}: {m.content}\n"

        # embed & check cache
        query_emb = await embed_query(query)
        cached_resp = await lfu_cache_get(query_emb)
        if cached_resp:
            # save query and cached resp in DB
            await db.message.create(
                data={"chatId": chat_id, "role": "user", "content": query}
            )
            await db.message.create(
                data={"chatId": chat_id, "role": "assistant", "content": cached_resp}
            )
            await websocket.send_text(cached_resp)
            await websocket.close()
            print("[WebSocket CACHE-HIT] Returned cached response.")
            return

        # zero-shot classify intent asynchronously
        intent = await intent_classifier.classify_intent(query)
        print(f"[WebSocket Debug] Query Intent = {intent}")

        # perform retrieval based on intent
        if intent == "SEMANTIC":
            partial_results = rass_engine.os_indexer.semantic_search(query_emb, k=3)
        elif intent == "KEYWORD":
            partial_results = rass_engine.os_indexer.exact_match_search(query, k=3)
        else:  # hyb
            partial_results = rass_engine.os_indexer.hybrid_search(
                query, query_emb, k=3
            )

        # Build contxt
        context_map = {}
        for doc_dict, _score in partial_results:
            doc_id = doc_dict["doc_id"]
            text_chunk = doc_dict["text"]
            if doc_id not in context_map:
                context_map[doc_id] = text_chunk
            else:
                context_map[doc_id] += "\n" + text_chunk

        context_text = ""
        for doc_id, doc_content in context_map.items():
            print(doc_id)
            context_text += f"--- Document ID: {doc_id} ---\n{doc_content}\n\n"

        # build the prompt with chat history and current query
        system_msg = (
            "You are a helpful AI assistant chatbot. You must follow these rules:\n"
            "1) Always cite document IDs from the context exactly as 'Document XYZ' without any file extensions like '.txt'.\n"
            "2) For every answer generated, there should be a reference or citation of the IDs of the documents from which the answer information was extracted at the end of the answer!\n"
            "3) If the context does not relate to the query, say 'I lack the context to answer your question.' For example, if the query is about gene mutations but the context is about climate change, acknowledge the mismatch and do not answer.\n"
            "4) Never ever give responses based on your own knowledge of the user query. Use ONLY the provided context + chat history to extract information relevant to the question. You should not answer without document ID references from which the information was extracted.\n"
            "5) If you lack context, then say so.\n"
            "6) Do not add chain-of-thought.\n"
            # "7) Answer in at most 4 sentences.\n"
        )

        final_prompt = (
            f"Chat History:\n{chat_history_str}\n\n"
            f"User Query:\n{query}\n\n"
            f"Context:\n{context_text}\n"
            "--- End of context ---\n\n"
            "Provide your concise answer now."
        )

        # stream the generation token-by-token
        streamed_chunks = []
        async for chunk in openai_generate_text_stream(final_prompt, system_msg):
            streamed_chunks.append(chunk)
            # send/stream each chunk immediately to the client
            await websocket.send_text(chunk)

        # After finishing, store the full response in Redis
        final_answer = "".join(streamed_chunks).strip()
        if final_answer:
            # Store query & answer/resp in DB
            await db.message.create(
                data={"chatId": chat_id, "role": "user", "content": query}
            )
            await db.message.create(
                data={"chatId": chat_id, "role": "assistant", "content": final_answer}
            )
            # Cache the response
            await lfu_cache_put(query_emb, final_answer)

        await websocket.close()

    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected mid-stream.")
    except Exception as e:
        print(f"[WebSocket] Unexpected error: {e}")
        await websocket.send_text(
            json.dumps({"error": "Internal server error occurred."})
        )
        await websocket.close()


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
