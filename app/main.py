import os
from dotenv import load_dotenv
import json
import numpy as np
from typing import List, Tuple, Optional, Dict, AsyncGenerator

import asyncio
from contextlib import asynccontextmanager

import httpx
import uvicorn


from prisma import Prisma

import openai

from fastapi import (
    FastAPI,
    Body,
    WebSocket,
    WebSocketDisconnect,
    HTTPException,
)

from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.helpers import bulk

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertForTokenClassification,
    BertTokenizerFast,
)
import torch
from datetime import datetime, timezone

import logging
from pathlib import Path
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Load .env file
load_dotenv()

# ==============================================================================
# Global Constants & Configuration
# ==============================================================================
NER_MODEL_PATH = os.getenv("NER_MODEL_PATH", "./ner_model/final")
INTENT_MODEL_PATH = os.getenv("INTENT_MODEL_PATH", "./intent_model/final")
EMBED_MODEL_NAME = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large:latest")


MAX_BLUEHIVE_CONCURRENCY = int(os.getenv("MAX_BLUEHIVE_CONCURRENCY", 5))
MAX_EMBED_CONCURRENCY = int(os.getenv("MAX_EMBED_CONCURRENCY", 5))

BLUEHIVE_BEARER_TOKEN = os.getenv("BLUEHIVE_BEARER_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
BLUEHIVE_API_URL = os.getenv("BLUEHIVEAI_URL", "")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api")

BATCH_SIZE = int(os.getenv("BATCH_SIZE", 64))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
EMBED_DIM = int(os.getenv("EMBED_DIM", 1024))

EMB_DIR = os.getenv("EMB_DIR", "sample_dataset")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "")

OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", 9200))
OPENSEARCH_INDEX_NAME = os.getenv("OPENSEARCH_INDEX_NAME", "")
TOP_K = int(os.getenv("TOP_K", "3"))
SHARD_COUNT = int(os.getenv("SHARD_COUNT", 1))
REPLICA_COUNT = int(os.getenv("REPLICA_COUNT", 0))

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

SUPPORTED_FILE_EXTENSIONS = (".json", ".md", ".txt")
FILE_TYPE_JSON = "json"
FILE_TYPE_MARKDOWN = "markdown"
FILE_TYPE_TEXT = "text"
MAX_FILES_PER_PATIENT = int(os.getenv("MAX_FILES_PER_PATIENT", 5))


db = Prisma()  # prisma client init


# ==============================================================================
# Load Trained Models
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

try:
    ner_model = BertForTokenClassification.from_pretrained(NER_MODEL_PATH).to(device)
    ner_tokenizer = BertTokenizerFast.from_pretrained(NER_MODEL_PATH)
    logger.info(f"Loaded NER model from {NER_MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load NER model: {e}")
    raise

try:
    intent_model = AutoModelForSequenceClassification.from_pretrained(
        INTENT_MODEL_PATH
    ).to(device)
    intent_tokenizer = AutoTokenizer.from_pretrained(INTENT_MODEL_PATH)
    logger.info(f"Loaded intent model from {INTENT_MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load intent model: {e}")
    raise

# Entity-to-field mapping for OpenSearch filters
ENTITY_FIELD_MAP = {
    "PERSON": "patientName",
    "DOCTOR": "practitionerName",
    "CONDITION": "conditionCodeText",
    "MEDICATION": "medRequestMedicationDisplay",
    "PROCEDURE": "procedureCodeText",
    "LABTEST": "observationCodeText",
    "ANATOMY": "observationCodeText",
    "OBS_VALUE": "observationValue",
    "ICD10_CODE": "conditionCodeText",
    "CPT_CODE": "procedureCodeText",
    "LOINC_CODE": "observationCodeText",
    "DATE": [
        "conditionOnsetDateTime",
        "observationIssued",
        "encounterStart",
        "medRequestAuthoredOn",
        "procedurePerformedDateTime",
        "allergyOnsetDateTime",
    ],
    "GENDER": "patientGender",
    "PHONE": "patientTelecom",
    "EMAIL": "patientTelecom",
    "ADDRESS": "patientAddress",
    "ORGANIZATION": "organizationName",
    "SEVERITY": "conditionSeverity",
    "ALLERGY": "allergyCodeText",
}


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


async def embed_texts_in_batches(
    texts: List[str], batch_size: int = BATCH_SIZE
) -> np.ndarray:
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
os_client: Optional[OpenSearch] = OpenSearch(
    hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
    http_compress=True,
    use_ssl=False,
    verify_certs=False,
    connection_class=RequestsHttpConnection,
)


def get_index_name(user_id: str) -> str:
    return f"{OPENSEARCH_INDEX_NAME}-{user_id}"


async def ensure_index_exists(client: OpenSearch, index_name: str):
    try:
        if not client.indices.exists(index_name):
            index_body = {
                "settings": {
                    "index": {
                        "knn": True,
                        "number_of_shards": SHARD_COUNT,
                        "number_of_replicas": REPLICA_COUNT,
                    }
                },  # for vector search
                "mappings": {
                    "properties": {
                        # ----------------------------------------------------------------
                        # Core Identifiers & Document Typing
                        # ----------------------------------------------------------------
                        "doc_id": {"type": "keyword"},
                        "doc_type": {"type": "keyword"},
                        "resourceType": {"type": "keyword"},
                        "file_path": {"type": "keyword"},
                        "file_type": {"type": "keyword"},  # json, markdown, text
                        # ----------------------------------------------------------------
                        # FHIR "Patient" resource fields
                        # ----------------------------------------------------------------
                        "patientId": {"type": "keyword"},
                        "patientName": {
                            "type": "text",
                            "fields": {
                                "keyword": {"type": "keyword", "ignore_above": 256}
                            },
                        },
                        "patientGender": {"type": "keyword"},
                        "patientDOB": {
                            "type": "date",
                            "format": "yyyy-MM-dd||strict_date_optional_time||epoch_millis",
                        },
                        "patientAddress": {
                            "type": "text",
                            "fields": {
                                "keyword": {"type": "keyword", "ignore_above": 256}
                            },
                        },
                        "patientMaritalStatus": {"type": "keyword"},
                        "patientMultipleBirth": {"type": "integer"},
                        "patientTelecom": {
                            "type": "text",
                            "fields": {
                                "keyword": {"type": "keyword", "ignore_above": 256}
                            },
                        },
                        "patientLanguage": {"type": "keyword"},
                        # ----------------------------------------------------------------
                        # FHIR "Condition" resource fields
                        # ----------------------------------------------------------------
                        "conditionId": {"type": "keyword"},
                        "conditionCodeText": {
                            "type": "text",
                            "fields": {
                                "keyword": {"type": "keyword", "ignore_above": 256}
                            },
                        },
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
                        "observationCodeText": {
                            "type": "text",
                            "fields": {
                                "keyword": {"type": "keyword", "ignore_above": 256}
                            },
                        },
                        "observationValue": {
                            "type": "text",
                            "fields": {
                                "keyword": {"type": "keyword", "ignore_above": 256}
                            },
                        },
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
                        "medRequestMedicationDisplay": {
                            "type": "text",
                            "fields": {
                                "keyword": {"type": "keyword", "ignore_above": 256}
                            },
                        },
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
                        "procedureCodeText": {
                            "type": "text",
                            "fields": {
                                "keyword": {"type": "keyword", "ignore_above": 256}
                            },
                        },
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
                        "allergyCodeText": {
                            "type": "text",
                            "fields": {
                                "keyword": {"type": "keyword", "ignore_above": 256}
                            },
                        },
                        "allergyOnsetDateTime": {
                            "type": "date",
                            "format": "date_time||strict_date_optional_time||epoch_millis",
                        },
                        "allergyNote": {"type": "text"},
                        # ----------------------------------------------------------------
                        # FHIR "Practitioner" resource fields
                        # ----------------------------------------------------------------
                        "practitionerId": {"type": "keyword"},
                        "practitionerName": {
                            "type": "text",
                            "fields": {
                                "keyword": {"type": "keyword", "ignore_above": 256}
                            },
                        },
                        "practitionerGender": {"type": "keyword"},
                        "practitionerSpecialty": {"type": "keyword"},
                        "practitionerAddress": {"type": "text"},
                        "practitionerTelecom": {"type": "text"},
                        # ----------------------------------------------------------------
                        # FHIR "Organization" resource fields
                        # ----------------------------------------------------------------
                        "organizationId": {"type": "keyword"},
                        "organizationName": {
                            "type": "text",
                            "fields": {
                                "keyword": {"type": "keyword", "ignore_above": 256}
                            },
                        },
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
            client.indices.create(index=index_name, body=index_body)
            print(f"[INFO] Created index '{index_name}' in OpenSearch.")
    except Exception as e:
        print(f"[Error] OpenSearch Index could not be created: {e}")


# ========================================================
# Parsing FHIR Bundle
# ====================================================
def extract_code_text(field) -> str:
    if isinstance(field, dict):
        return field.get("text") or field.get("coding", [{}])[0].get("code", "")
    elif isinstance(field, str):
        return field

    return str(field)


def parse_fhir_bundle(bundle_json: Dict) -> Tuple[List[Dict], List[Dict]]:
    """
    Returns two lists:
      1) structured_docs: array of dicts with typed FHIR fields
      2) unstructured_docs: array of dicts for narrative text (to embed)
    Each resource may produce:
      - One doc capturing structured fields (doc_type='structured')
      - One or more docs capturing unstructured text (doc_type='unstructured')

    We handle multiple resource types: Patient, Condition, Observation, Encounter,
    MedicationRequest, Procedure, AllergyIntolerance, Practitioner, Organization, etc.
    and gather text from resource.text.div, resource.note, etc. for the unstructured docs.
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
            "file_path": None,
            "file_type": FILE_TYPE_JSON,
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
            # We don't store embeddings in structured docs - unstructured docs will have "embedding" once embedded
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
                # ms = resource["maritalStatus"]
                # sdoc["patientMaritalStatus"] = ms.get("text") or ms.get("coding", [{}])[
                #     0
                # ].get("code")
                sdoc["patientMaritalStatus"] = extract_code_text(
                    resource["maritalStatus"]
                )

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
                # c_lang = comm.get("language", {})
                # sdoc["patientLanguage"] = c_lang.get("text") or c_lang.get(
                #     "coding", [{}]
                # )[0].get("code")
                sdoc["patientLanguage"] = extract_code_text(comm.get("language", {}))

        elif rtype == "Condition":
            sdoc["conditionId"] = rid
            # cstatus = resource.get("clinicalStatus", {})
            # sdoc["conditionClinicalStatus"] = cstatus.get("text") or cstatus.get(
            #     "coding", [{}]
            # )[0].get("code")

            sdoc["conditionClinicalStatus"] = extract_code_text(
                resource.get("clinicalStatus", {})
            )

            # vstatus = resource.get("verificationStatus", {})
            # sdoc["conditionVerificationStatus"] = vstatus.get("text") or vstatus.get(
            #     "coding", [{}]
            # )[0].get("code")
            sdoc["conditionVerificationStatus"] = extract_code_text(
                resource.get("verificationStatus", {})
            )

            # category
            if "category" in resource and len(resource["category"]) > 0:
                cat = resource["category"][0]
                # sdoc["conditionCategory"] = cat.get("text") or cat.get("coding", [{}])[
                #     0
                # ].get("code")
                sdoc["conditionCategory"] = extract_code_text(cat)

            # severity
            if "severity" in resource:
                sev = resource["severity"]
                # sdoc["conditionSeverity"] = sev.get("text") or sev.get("coding", [{}])[
                #     0
                # ].get("code")
                sdoc["conditionSeverity"] = extract_code_text(sev)

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
            # sdoc["allergyClinicalStatus"] = resource.get("clinicalStatus", {}).get(
            #     "text"
            # )
            sdoc["allergyClinicalStatus"] = extract_code_text(
                resource.get("clinicalStatus")
            )
            # sdoc["allergyVerificationStatus"] = resource.get(
            #     "verificationStatus", {}
            # ).get("text")
            sdoc["allergyVerificationStatus"] = extract_code_text(
                resource.get("verificationStatus")
            )

            sdoc["allergyType"] = resource.get("type")
            if "category" in resource and len(resource["category"]) > 0:
                # sdoc["allergyCategory"] = resource["category"][0]
                catval = resource["category"][0]
                sdoc["allergyCategory"] = extract_code_text(catval)

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
                # sdoc["practitionerSpecialty"] = q.get("code", {}).get("text")
                sdoc["practitionerSpecialty"] = extract_code_text(q.get("code", {}))

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

        # ... if needed in the future more resource types can be added ...

        # add the structured doc
        structured_docs.append(sdoc)

        # chunk unstructured text for embedding as these are notes, we set
        # doc_type="unstructured". Each chunk becomes a separate doc due to chunking
        if unstructured_text_pieces:
            combined_text = "\n".join(unstructured_text_pieces).strip()
            if not combined_text:
                continue

            # chunk long text
            text_chunks = chunk_text(combined_text, chunk_size=CHUNK_SIZE)
            for c_i, chunk_str in enumerate(text_chunks):
                udoc = {
                    "doc_id": f"{rtype}-{rid}-unstructured-{c_i}",
                    "doc_type": "unstructured",
                    "resourceType": rtype,
                    "file_path": None,
                    "file_type": FILE_TYPE_JSON,
                    "patientId": sdoc["patientId"],
                    # storing the text in 'unstructuredText' field
                    "unstructuredText": chunk_str,
                }
                unstructured_docs.append(udoc)

    return (structured_docs, unstructured_docs)


def parse_fhir_bundle_with_path(
    bundle_json: Dict, file_path: str
) -> Tuple[List[Dict], List[Dict]]:
    structured_docs, unstructured_docs = parse_fhir_bundle(bundle_json)
    file_upload_dir = Path(UPLOAD_DIR).resolve()
    absolute_path = Path(file_path).resolve()
    try:
        relative_file_path = str(absolute_path.relative_to(file_upload_dir))
    except ValueError:
        logger.warning(
            f"File {file_path} is not under FILE_DIR {file_upload_dir}, using absolute path"
        )
        relative_file_path = str(absolute_path)

    for doc in structured_docs:
        doc["file_path"] = relative_file_path

    for doc in unstructured_docs:
        doc["file_path"] = relative_file_path

    return structured_docs, unstructured_docs


def infer_patient_id_from_filename(filename: str) -> Optional[str]:
    """
    Infer patientId from filename, e.g., 'patient_123_notes.md' -> '123'.
    Returns None if no patientId is found.
    """
    match = re.search(r"patient_(\d+)", filename, re.IGNORECASE)
    return match.group(1) if match else None


def parse_text_file(file_path: str, file_type: str) -> Tuple[List[Dict], List[Dict]]:
    structured_docs = []
    unstructured_docs = []
    emb_dir = Path(UPLOAD_DIR).resolve()
    absolute_path = Path(file_path).resolve()
    try:
        relative_file_path = str(absolute_path.relative_to(emb_dir))
    except ValueError:
        logger.warning(
            f"File {file_path} is not under EMB_DIR {emb_dir}, using absolute path"
        )
        relative_file_path = str(absolute_path)
    patient_id = infer_patient_id_from_filename(absolute_path.name)
    try:
        with absolute_path.open("r", encoding="utf-8") as f:
            content = f.read().strip()
    except UnicodeDecodeError:
        with absolute_path.open("r", encoding="latin-1") as f:
            content = f.read().strip()
    except Exception as e:
        logger.error(f"Failed to read text file {file_path}: {e}")
        return structured_docs, unstructured_docs
    if not content:
        logger.warning(f"Empty text file: {file_path}")
        return structured_docs, unstructured_docs

    text_chunks = chunk_text(content, chunk_size=CHUNK_SIZE)
    for i, chunk in enumerate(text_chunks):
        doc_id = f"{file_type}-{absolute_path.stem}-{i}"
        unstructured_docs.append(
            {
                "doc_id": doc_id,
                "doc_type": "unstructured",
                "resourceType": file_type,
                "file_path": relative_file_path,
                "file_type": file_type,
                "patientId": patient_id,
                "unstructuredText": chunk,
            }
        )

    return structured_docs, unstructured_docs


async def store_fhir_docs_in_opensearch(
    structured_docs: List[Dict],
    unstructured_docs: List[Dict],
    client: OpenSearch,
    index_name: str,
) -> None:
    if not client:
        print("[store_fhir_docs_in_opensearch] No OS client.")
        return

    await ensure_index_exists(client, index_name)

    # Bulk index structured docs
    bulk_actions_struct = [
        {
            "_op_type": "index",
            "_index": index_name,
            "_id": doc["doc_id"],
            "_source": doc,
            "_routing": doc.get("patientId"),
        }
        for doc in structured_docs
    ]

    if bulk_actions_struct:
        try:
            success, errors = bulk(client, bulk_actions_struct)
            logger.info(f"Indexed {success} structured docs, errors: {errors}")
        except Exception as e:
            logger.error(f"Structured docs indexing error: {e}")

    if not unstructured_docs:
        return

    # embed the unstructured text, then store each doc with "unstructuredText" plus the "embedding" field.
    un_texts = [d["unstructuredText"] for d in unstructured_docs]
    embeddings = await embed_texts_in_batches(un_texts, batch_size=BATCH_SIZE)

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-9)

    bulk_actions_unstruct = []
    for i, docu in enumerate(unstructured_docs):
        # store as float array - emb_vector
        docu["embedding"] = embeddings[i].tolist()
        # storing docu in index using the doc_id as _id
        action_unstruct = {
            "_op_type": "index",
            "_index": index_name,
            "_id": docu["doc_id"],
            "_source": docu,
            "_routing": docu.get("patientId"),
        }
        bulk_actions_unstruct.append(action_unstruct)

        if len(bulk_actions_unstruct) >= BATCH_SIZE:
            try:
                u_success, u_errors = bulk(client, bulk_actions_unstruct)
                logger.info(
                    f"Indexed {u_success} unstructured docs, errors: {u_errors}"
                )
                bulk_actions_unstruct = []
            except Exception as e:
                logger.error(f"Unstructured docs indexing error: {e}")

    if bulk_actions_unstruct:
        try:
            u_success, u_errors = bulk(client, bulk_actions_unstruct)
            logger.info(f"Indexed {u_success} unstructured docs, errors: {u_errors}")
        except Exception as e:
            logger.error(f"Unstructured docs indexing error: {e}")


async def ingest_fhir_directory(fhir_dir: str, user_id: str) -> None:
    index_name = get_index_name(user_id)
    await ensure_index_exists(os_client, index_name)
    all_files = []
    for root, _, files in os.walk(fhir_dir):
        for f in files:
            if f.lower().endswith(SUPPORTED_FILE_EXTENSIONS):
                all_files.append(os.path.join(root, f))

    if not all_files:
        logger.info(f"No supported files found in {fhir_dir}")
        return

    for path in all_files:
        valid_path = validate_file_path(path, read=True)
        if not valid_path:
            logger.error(f"Skipping invalid file: {path}")
            continue

        logger.info(f"Processing file: {path}")
        try:
            file_ext = valid_path.suffix.lower()
            if file_ext == ".json":
                with valid_path.open("r", encoding="utf-8") as f:
                    bundle_json = json.load(f)
                structured_docs, unstructured_docs = parse_fhir_bundle_with_path(
                    bundle_json, path
                )
            else:
                file_type = FILE_TYPE_MARKDOWN if file_ext == ".md" else FILE_TYPE_TEXT
                structured_docs, unstructured_docs = parse_text_file(path, file_type)

            await store_fhir_docs_in_opensearch(
                structured_docs, unstructured_docs, os_client, index_name
            )
        except Exception as e:
            logger.error(f"Error processing {path}: {e}")


def validate_file_path(
    file_path: str, base_dir: str = None, read: bool = False
) -> Optional[Path]:
    """
    Validate a file path. If base_dir is provided, treat file_path as relative.
    If read=True, check readability.
    Accepts .json, .md, .txt files.
    Returns Path object if valid, None otherwise.
    """
    try:
        if base_dir:
            path = Path(base_dir) / file_path
        else:
            path = Path(file_path)

        path = path.resolve()
        if not path.exists():
            logger.error(f"File does not exist: {path}")
            return None

        if not path.is_file():
            logger.error(f"Path is not a file: {path}")
            return None

        if path.suffix.lower() not in SUPPORTED_FILE_EXTENSIONS:
            logger.error(f"Unsupported file extension: {path}")
            return None

        if read:
            with path.open("r", encoding="utf-8") as f:
                content = f.read()
                if not content.strip():
                    logger.error(f"File is empty: {path}")
                    return None

        return path
    except PermissionError:
        logger.error(f"Permission denied for file: {path}")
        return None
    except UnicodeDecodeError:
        logger.error(f"File is not valid UTF-8: {path}")
        return None
    except Exception as e:
        logger.error(f"Invalid file path {path}: {e}")
        return None


async def retrieve_ehr_document(file_path: str) -> Optional[Dict]:
    valid_path = validate_file_path(file_path, base_dir=EMB_DIR, read=True)
    if not valid_path:
        logger.error(f"Cannot retrieve EHR document: {file_path}")
        return None

    try:
        file_ext = valid_path.suffix.lower()
        if file_ext == ".json":
            with valid_path.open("r", encoding="utf-8") as f:
                content = json.load(f)

            return {"file_type": FILE_TYPE_JSON, "content": content}
        else:
            with valid_path.open("r", encoding="utf-8") as f:
                content = f.read()

            file_type = FILE_TYPE_MARKDOWN if file_ext == ".md" else FILE_TYPE_TEXT
            return {"file_type": file_type, "content": content}
    except Exception as e:
        logger.error(f"Error reading EHR document {valid_path}: {e}")
        return None


class OpenSearchIndexer:
    """
    Index documents with embeddings into OpenSearch with a hybrid search approach
    """

    def __init__(self, client: OpenSearch, index_name: str):
        self.client = client
        self.index_name = index_name
        self.text_fields = [
            "unstructuredText^3",
            "patientName^3",
            "patientAddress^3",
            "patientTelecom^3",
            "conditionCodeText^2",
            "conditionNote^2",
            "observationCodeText",
            "observationValue",
            "observationReferenceRange",
            "observationNote^2",
            "encounterType",
            "encounterReasonCode",
            "encounterLocation",
            "encounterNote",
            "medRequestMedicationDisplay",
            "medRequestNote",
            "procedureCodeText",
            "procedureNote",
            "allergyCodeText",
            "allergyNote^2",
            "practitionerName^3",
            "practitionerAddress",
            "practitionerTelecom",
            "organizationName^3",
            "organizationAddress",
            "organizationTelecom",
        ]
        self.keyword_fields = [
            "patientGender^3",
            "patientMaritalStatus^2",
            "patientLanguage^3",
            "conditionCategory^2",
            "conditionClinicalStatus",
            "conditionVerificationStatus",
            "conditionSeverity",
            "observationUnit",
            "observationInterpretation",
            "encounterStatus",
            "encounterClass",
            "encounterServiceProvider",
            "medRequestIntent",
            "medRequestStatus",
            "medRequestPriority",
            "procedureStatus",
            "allergyClinicalStatus",
            "allergyVerificationStatus",
            "allergyType",
            "allergyCategory",
            "allergyCriticality",
            "practitionerGender",
            "practitionerSpecialty",
            "organizationType",
        ]
        self.date_fields = [
            "patientDOB",
            "conditionOnsetDateTime",
            "conditionRecordedDate",
            "observationEffectiveDateTime",
            "observationIssued",
            "encounterStart",
            "encounterEnd",
            "medRequestAuthoredOn",
            "procedurePerformedDateTime",
            "allergyOnsetDateTime",
        ]

    def has_any_data(self) -> bool:
        if not self.client:
            return False

        try:
            resp = self.client.count(index=self.index_name)
            return resp["count"] > 0
        except Exception:
            return False

    def exact_match_search(
        self,
        query: str,
        k: int = TOP_K,
        filter_clause: Optional[Dict] = None,
        patient_id: Optional[str] = None,
    ) -> List[Tuple[Dict, float]]:
        if not query.strip():
            return []
        bool_query = {
            "should": [
                {
                    "multi_match": {
                        "query": query,
                        "fields": self.text_fields,
                        "type": "phrase",
                        "boost": 2.0,
                    }
                },
                {
                    "multi_match": {
                        "query": query,
                        "fields": self.keyword_fields,
                        "type": "phrase",
                    }
                },
            ],
            "minimum_should_match": 1,
        }
        if filter_clause or patient_id:
            bool_query["filter"] = []
            if filter_clause:
                bool_query["filter"].append(filter_clause)
            if patient_id:
                bool_query["filter"].append({"term": {"patientId": patient_id}})
        query_body = {"size": k, "query": {"bool": bool_query}, "terminate_after": k}
        try:
            resp = self.client.search(
                index=self.index_name, body=query_body, routing=patient_id
            )
            return [
                (hit["_source"], float(hit["_score"])) for hit in resp["hits"]["hits"]
            ]
        except Exception as e:
            logger.error(f"Exact match search error: {e}")
            return []

    def semantic_search(
        self,
        query_emb: np.ndarray,
        k: int = TOP_K,
        filter_clause: Optional[Dict] = None,
        patient_id: Optional[str] = None,
    ) -> List[Tuple[Dict, float]]:
        if query_emb.size == 0:
            return []
        norms = np.linalg.norm(query_emb, axis=1, keepdims=True)
        vector = (query_emb / (norms + 1e-9))[0].tolist()
        query_body = {
            "size": k,
            "query": {"knn": {"embedding": {"vector": vector, "k": k}}},
            "terminate_after": k,
        }
        if filter_clause or patient_id:
            bool_query = {"must": [query_body["query"]]}
            bool_query["filter"] = []
            if filter_clause:
                bool_query["filter"].append(filter_clause)
            if patient_id:
                bool_query["filter"].append({"term": {"patientId": patient_id}})
            query_body["query"] = {"bool": bool_query}
        try:
            resp = self.client.search(
                index=self.index_name, body=query_body, routing=patient_id
            )
            return [
                (hit["_source"], float(hit["_score"])) for hit in resp["hits"]["hits"]
            ]
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return []

    def hybrid_search(
        self,
        query: str,
        query_emb: np.ndarray,
        k: int = TOP_K,
        filter_clause: Optional[Dict] = None,
        patient_id: Optional[str] = None,
    ) -> List[Tuple[Dict, float]]:
        if not query.strip() or query_emb.size == 0:
            return []
        norms = np.linalg.norm(query_emb, axis=1, keepdims=True)
        vector = (query_emb / (norms + 1e-9))[0].tolist()
        bool_query = {
            "should": [
                {
                    "multi_match": {
                        "query": query,
                        "fields": self.text_fields,
                        "type": "best_fields",
                        "operator": "or",
                        "fuzziness": "AUTO",
                        "boost": 1.5,
                    }
                },
                {
                    "multi_match": {
                        "query": query,
                        "fields": self.keyword_fields,
                        "type": "best_fields",
                        "operator": "or",
                        "boost": 1.0,
                    }
                },
                {"knn": {"embedding": {"vector": vector, "k": k, "boost": 2.0}}},
            ],
            "minimum_should_match": 1,
        }
        if filter_clause or patient_id:
            bool_query["filter"] = []
            if filter_clause:
                bool_query["filter"].append(filter_clause)
            if patient_id:
                bool_query["filter"].append({"term": {"patientId": patient_id}})
        query_body = {"size": k, "query": {"bool": bool_query}, "terminate_after": k}
        try:
            resp = self.client.search(
                index=self.index_name, body=query_body, routing=patient_id
            )
            return [
                (hit["_source"], float(hit["_score"])) for hit in resp["hits"]["hits"]
            ]
        except Exception as e:
            logger.error(f"Hybrid search error: {e}")
            return []

    def structured_search(
        self,
        query: str,
        k: int = TOP_K,
        filter_clause: Optional[Dict] = None,
        patient_id: Optional[str] = None,
    ) -> List[Tuple[Dict, float]]:
        if not query.strip():
            return []
        structured_fields = [
            "patientName^3",
            "patientGender^3",
            "patientDOB",
            "patientTelecom^3",
            "conditionCodeText^2",
            "conditionClinicalStatus",
            "conditionSeverity",
            "observationCodeText",
            "observationValue",
            "observationUnit",
            "encounterStatus",
            "encounterClass",
            "medRequestMedicationDisplay",
            "medRequestStatus",
            "procedureCodeText",
            "procedureStatus",
            "allergyCodeText",
            "allergyClinicalStatus",
            "practitionerName^3",
            "organizationName^3",
        ]
        bool_query = {
            "must": [
                {
                    "multi_match": {
                        "query": query,
                        "fields": structured_fields,
                        "type": "phrase_prefix",
                        "operator": "and",
                    }
                }
            ]
        }
        if filter_clause or patient_id:
            bool_query["filter"] = []
            if filter_clause:
                bool_query["filter"].append(filter_clause)
            if patient_id:
                bool_query["filter"].append({"term": {"patientId": patient_id}})
        bool_query["filter"].append({"term": {"doc_type": "structured"}})
        query_body = {"size": k, "query": {"bool": bool_query}, "terminate_after": k}
        try:
            resp = self.client.search(
                index=self.index_name, body=query_body, routing=patient_id
            )
            return [
                (hit["_source"], float(hit["_score"])) for hit in resp["hits"]["hits"]
            ]
        except Exception as e:
            logger.error(f"Structured search error: {e}")
            return []

    def hybrid_structured_search(
        self,
        query: str,
        query_emb: np.ndarray,
        k: int = TOP_K,
        filter_clause: Optional[Dict] = None,
        patient_id: Optional[str] = None,
    ) -> List[Tuple[Dict, float]]:
        if not query.strip() or query_emb.size == 0:
            return []
        norms = np.linalg.norm(query_emb, axis=1, keepdims=True)
        vector = (query_emb / (norms + 1e-9))[0].tolist()
        structured_fields = [
            "patientName^3",
            "patientGender^3",
            "patientTelecom^3",
            "conditionCodeText^2",
            "conditionClinicalStatus",
            "conditionSeverity",
            "observationCodeText",
            "observationValue",
            "observationUnit",
            "encounterStatus",
            "encounterClass",
            "medRequestMedicationDisplay",
            "medRequestStatus",
            "procedureCodeText",
            "procedureStatus",
            "allergyCodeText",
            "allergyClinicalStatus",
            "practitionerName^3",
            "organizationName^3",
        ]
        bool_query = {
            "should": [
                {
                    "multi_match": {
                        "query": query,
                        "fields": structured_fields,
                        "type": "phrase_prefix",
                        "operator": "and",
                        "boost": 1.5,
                    }
                },
                {"knn": {"embedding": {"vector": vector, "k": k, "boost": 2.0}}},
            ],
            "minimum_should_match": 1,
        }
        if filter_clause or patient_id:
            bool_query["filter"] = []
            if filter_clause:
                bool_query["filter"].append(filter_clause)
            if patient_id:
                bool_query["filter"].append({"term": {"patientId": patient_id}})
        bool_query["filter"].append({"term": {"doc_type": "structured"}})
        query_body = {"size": k, "query": {"bool": bool_query}, "terminate_after": k}
        try:
            resp = self.client.search(
                index=self.index_name, body=query_body, routing=patient_id
            )
            return [
                (hit["_source"], float(hit["_score"])) for hit in resp["hits"]["hits"]
            ]
        except Exception as e:
            logger.error(f"Hybrid structured search error: {e}")
            return []

    def aggregate_search(
        self,
        query: str,
        filter_clause: Optional[Dict] = None,
        patient_id: Optional[str] = None,
    ) -> Dict:
        agg_body = {
            "size": 0,
            "aggs": {
                "by_condition": {
                    "terms": {"field": "conditionCodeText.keyword", "size": 5}
                },
                "by_resource": {"terms": {"field": "resourceType.keyword", "size": 5}},
                "by_patient": {"terms": {"field": "patientId", "size": 5}},
            },
        }
        if filter_clause or patient_id:
            agg_body["query"] = {"bool": {"filter": []}}
            if filter_clause:
                agg_body["query"]["bool"]["filter"].append(filter_clause)
            if patient_id:
                agg_body["query"]["bool"]["filter"].append(
                    {"term": {"patientId": patient_id}}
                )
        try:
            resp = self.client.search(
                index=self.index_name, body=agg_body, routing=patient_id
            )
            return resp["aggregations"]
        except Exception as e:
            logger.error(f"Aggregate search error: {e}")
            return {}

    def comparison_search(
        self,
        query: str,
        k: int = TOP_K,
        filter_clause: Optional[Dict] = None,
        patient_id: Optional[str] = None,
    ) -> List[Tuple[Dict, float]]:
        if not query.strip():
            return []
        compare_fields = [
            "conditionCodeText^2",
            "observationValue",
            "observationUnit",
            "medRequestMedicationDisplay",
            "procedureCodeText",
            "allergyCodeText",
        ]
        bool_query = {
            "should": [
                {
                    "multi_match": {
                        "query": query,
                        "fields": compare_fields,
                        "type": "best_fields",
                        "operator": "or",
                        "fuzziness": "AUTO",
                    }
                }
            ],
            "minimum_should_match": 1,
        }
        if filter_clause or patient_id:
            bool_query["filter"] = []
            if filter_clause:
                bool_query["filter"].append(filter_clause)
            if patient_id:
                bool_query["filter"].append({"term": {"patientId": patient_id}})
        query_body = {
            "size": k,
            "query": {"bool": bool_query},
            "aggs": {
                "by_field": {"terms": {"field": "conditionCodeText.keyword", "size": 3}}
            },
            "terminate_after": k,
        }
        try:
            resp = self.client.search(
                index=self.index_name, body=query_body, routing=patient_id
            )
            return [
                (hit["_source"], float(hit["_score"])) for hit in resp["hits"]["hits"]
            ]
        except Exception as e:
            logger.error(f"Comparison search error: {e}")
            return []

    def temporal_search(
        self,
        query: str,
        k: int = TOP_K,
        filter_clause: Optional[Dict] = None,
        patient_id: Optional[str] = None,
    ) -> List[Tuple[Dict, float]]:
        if not query.strip():
            return []
        date_query = {
            "bool": {
                "should": [
                    {"range": {field: {"gte": "now-1y", "lte": "now"}}}
                    for field in self.date_fields
                ],
                "minimum_should_match": 1,
            }
        }
        bool_query = {
            "must": [
                {
                    "multi_match": {
                        "query": query,
                        "fields": self.text_fields + self.keyword_fields,
                        "type": "best_fields",
                        "operator": "or",
                    }
                },
                date_query,
            ]
        }
        if filter_clause or patient_id:
            bool_query["filter"] = []
            if filter_clause:
                bool_query["filter"].append(filter_clause)
            if patient_id:
                bool_query["filter"].append({"term": {"patientId": patient_id}})
        query_body = {
            "size": k,
            "query": {"bool": bool_query},
            "sort": [{"conditionOnsetDateTime": {"order": "desc"}}],
            "terminate_after": k,
        }
        try:
            resp = self.client.search(
                index=self.index_name, body=query_body, routing=patient_id
            )
            return [
                (hit["_source"], float(hit["_score"])) for hit in resp["hits"]["hits"]
            ]
        except Exception as e:
            logger.error(f"Temporal search error: {e}")
            return []

    def explanatory_search(
        self,
        query: str,
        k: int = TOP_K,
        filter_clause: Optional[Dict] = None,
        patient_id: Optional[str] = None,
    ) -> List[Tuple[Dict, float]]:
        if not query.strip():
            return []
        note_fields = [
            "conditionNote^3",
            "observationNote^3",
            "encounterNote^3",
            "medRequestNote^3",
            "procedureNote^3",
            "allergyNote^3",
            "unstructuredText^2",
        ]
        bool_query = {
            "must": [
                {
                    "multi_match": {
                        "query": query,
                        "fields": note_fields,
                        "type": "best_fields",
                        "operator": "or",
                        "fuzziness": "AUTO",
                    }
                }
            ]
        }
        if filter_clause or patient_id:
            bool_query["filter"] = []
            if filter_clause:
                bool_query["filter"].append(filter_clause)
            if patient_id:
                bool_query["filter"].append({"term": {"patientId": patient_id}})
        query_body = {"size": k, "query": {"bool": bool_query}, "terminate_after": k}
        try:
            resp = self.client.search(
                index=self.index_name, body=query_body, routing=patient_id
            )
            return [
                (hit["_source"], float(hit["_score"])) for hit in resp["hits"]["hits"]
            ]
        except Exception as e:
            logger.error(f"Explanatory search error: {e}")
            return []

    def multi_intent_search(
        self,
        query: str,
        query_emb: np.ndarray,
        k: int = TOP_K,
        filter_clause: Optional[Dict] = None,
        patient_id: Optional[str] = None,
    ) -> List[Tuple[Dict, float]]:
        if not query.strip() or query_emb.size == 0:
            return []

        norms = np.linalg.norm(query_emb, axis=1, keepdims=True)
        vector = (query_emb / (norms + 1e-9))[0].tolist()
        bool_query = {
            "should": [
                {
                    "multi_match": {
                        "query": query,
                        "fields": self.text_fields,
                        "type": "best_fields",
                        "operator": "or",
                        "fuzziness": "AUTO",
                        "boost": 1.0,
                    }
                },
                {
                    "multi_match": {
                        "query": query,
                        "fields": self.keyword_fields,
                        "type": "best_fields",
                        "operator": "or",
                        "boost": 0.5,
                    }
                },
                {"knn": {"embedding": {"vector": vector, "k": k, "boost": 1.5}}},
                {
                    "range": {field: {"gte": "now-1y", "lte": "now", "boost": 0.5}}
                    for field in self.date_fields
                },
            ],
            "minimum_should_match": 1,
        }
        if filter_clause or patient_id:
            bool_query["filter"] = []
            if filter_clause:
                bool_query["filter"].append(filter_clause)
            if patient_id:
                bool_query["filter"].append({"term": {"patientId": patient_id}})
        query_body = {"size": k, "query": {"bool": bool_query}, "terminate_after": k}
        try:
            resp = self.client.search(
                index=self.index_name, body=query_body, routing=patient_id
            )
            return [
                (hit["_source"], float(hit["_score"])) for hit in resp["hits"]["hits"]
            ]
        except Exception as e:
            logger.error(f"Multi-intent search error: {e}")
            return []

    def entity_specific_search(
        self,
        query: str,
        k: int = TOP_K,
        filter_clause: Optional[Dict] = None,
        patient_id: Optional[str] = None,
    ) -> List[Tuple[Dict, float]]:
        if not query.strip():
            return []
        entity_fields = [
            "patientName^4",
            "patientId^4",
            "patientGender^3",
            "patientTelecom^3",
            "practitionerName^3",
            "organizationName^3",
        ]
        bool_query = {
            "must": [
                {
                    "multi_match": {
                        "query": query,
                        "fields": entity_fields,
                        "type": "phrase",
                        "operator": "and",
                    }
                }
            ]
        }
        if filter_clause or patient_id:
            bool_query["filter"] = []
            if filter_clause:
                bool_query["filter"].append(filter_clause)
            if patient_id:
                bool_query["filter"].append({"term": {"patientId": patient_id}})
        query_body = {"size": k, "query": {"bool": bool_query}, "terminate_after": k}
        try:
            resp = self.client.search(
                index=self.index_name, body=query_body, routing=patient_id
            )
            return [
                (hit["_source"], float(hit["_score"])) for hit in resp["hits"]["hits"]
            ]
        except Exception as e:
            logger.error(f"Entity-specific search error: {e}")
            return []

    def document_fetch_search(
        self,
        query: str,
        k: int = TOP_K,
        filter_clause: Optional[Dict] = None,
        patient_id: Optional[str] = None,
    ) -> List[Tuple[Dict, float]]:
        if not query.strip():
            return []
        bool_query = {
            "must": [
                {
                    "multi_match": {
                        "query": query,
                        "fields": ["patientId^4", "file_path^3"],
                        "type": "phrase",
                    }
                }
            ]
        }
        if filter_clause or patient_id:
            bool_query["filter"] = []
            if filter_clause:
                bool_query["filter"].append(filter_clause)
            if patient_id:
                bool_query["filter"].append({"term": {"patientId": patient_id}})

        query_body = {
            "size": k,
            "query": {"bool": bool_query},
            "collapse": {"field": "patientId"},
            "terminate_after": k,
        }
        try:
            resp = self.client.search(
                index=self.index_name, body=query_body, routing=patient_id
            )
            return [
                (hit["_source"], float(hit["_score"])) for hit in resp["hits"]["hits"]
            ]
        except Exception as e:
            logger.error(f"Document fetch search error: {e}")
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


# Initialize the model pipeline
intent_classifier = pipeline(
    "zero-shot-classification",
    model=QUERY_INTENT_CLASSIFICATION_MODEL,
    device=0 if torch.cuda.is_available() else -1,
)

# Intent categories
INTENT_CATEGORIES = [
    "SEMANTIC",
    "KEYWORD",
    "HYBRID",
    "STRUCTURED",
    "HYBRID_STRUCTURED",
    "AGGREGATE",
    "COMPARISON",
    "TEMPORAL",
    "EXPLANATORY",
    "MULTI_INTENT",
    "ENTITY_SPECIFIC",
    "DOCUMENT_FETCH",
]

# Few-shot examples embedded in the classifier prompt
FEW_SHOT_EXAMPLES = """
Classify the query into one of these intents: SEMANTIC, KEYWORD, HYBRID, STRUCTURED, HYBRID_STRUCTURED, AGGREGATE, COMPARISON, TEMPORAL, EXPLANATORY, MULTI_INTENT, ENTITY_SPECIFIC, DOCUMENT_FETCH.

Examples:
1. Query: "What are the symptoms of diabetes?" Intent: EXPLANATORY
2. Query: "Fetch the medical records for patient John Doe." Intent: DOCUMENT_FETCH
3. Query: "How many patients have hypertension?" Intent: AGGREGATE
4. Query: "Compare the outcomes of heart surgery vs. medication." Intent: COMPARISON
5. Query: "Show me trends in blood pressure for patient 123 over time." Intent: TEMPORAL
6. Query: "Find patients with heart disease." Intent: HYBRID
7. Query: "Get details for patient Jane Smith." Intent: ENTITY_SPECIFIC
8. Query: "Search for diabetes treatment options." Intent: SEMANTIC
9. Query: "List all procedures with CPT code 99213." Intent: STRUCTURED
10. Query: "Find patients with both asthma and allergies." Intent: HYBRID_STRUCTURED
11. Query: "Explain the procedure for knee replacement and list patients who had it." Intent: MULTI_INTENT
12. Query: "Look up ICD-10 code I21." Intent: KEYWORD
13. Query: "Fetch records of patient with name Mary Johnson or number 456 or address 123 Main St." Intent: ENTITY_SPECIFIC
14. Query: "Fetch me the details of patients with heart problems." Intent: HYBRID
15. Query: "What is the status of the medication request for patient 789?" Intent: EXPLANATORY
16. Query: "Get the latest lab results for patient 101." Intent: DOCUMENT_FETCH
17. Query: "How many patients were treated in the last month?" Intent: AGGREGATE
18. Query: "Compare the lab results of patient 202 and patient 303." Intent: COMPARISON
19. Query: "Show me the trends in cholesterol levels for patient 404 over the last year." Intent: TEMPORAL
20. Query: "Get me the details of the procedure performed on patient 505." Intent: ENTITY_SPECIFIC
21. Query: "Get me the document for Julian140" Intent: DOCUMENT_FETCH
22. Query: "Get me the document for Julian140 and the procedure code 99213" Intent: MULTI_INTENT
"""


# ==============================================================================
# NER and Intent Preprocessing
# ==============================================================================
def ner_preprocess(query: str) -> Optional[Dict]:
    inputs = ner_tokenizer(
        query, return_tensors="pt", truncation=True, padding=True, max_length=128
    ).to(device)
    with torch.no_grad():
        outputs = ner_model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)[0].cpu().numpy()
    tokens = ner_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    id2label = ner_model.config.id2label

    entities = []
    current_entity = None
    current_text = []
    for token, pred_id in zip(tokens, predictions):
        label = id2label[pred_id]
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
        if label.startswith("B-"):
            if current_entity:
                entities.append(
                    {"text": " ".join(current_text), "label": current_entity}
                )
                current_text = []
            current_entity = label[2:]
            current_text.append(token.replace("##", ""))
        elif label.startswith("I-") and current_entity == label[2:]:
            current_text.append(token.replace("##", ""))
        else:
            if current_entity:
                entities.append(
                    {"text": " ".join(current_text), "label": current_entity}
                )
                current_text = []
                current_entity = None
            if label != "O":
                current_entity = label[2:] if label.startswith("B-") else None
                current_text = [token.replace("##", "")] if current_entity else []

    if current_entity and current_text:
        entities.append({"text": " ".join(current_text), "label": current_entity})

    must_filters = []
    for entity in entities:
        label = entity["label"]
        value = entity["text"].strip().lower()
        if label not in ENTITY_FIELD_MAP:
            continue
        fields = ENTITY_FIELD_MAP[label]
        if label == "DATE":
            if isinstance(fields, list):
                for field in fields:
                    try:
                        must_filters.append(
                            {"range": {field: {"gte": value, "lte": value}}}
                        )
                    except Exception as e:
                        logger.warning(f"Invalid date format for {value}: {e}")
        else:
            field = fields if isinstance(fields, str) else fields[0]
            must_filters.append({"match_phrase": {field: value}})

    return {"bool": {"must": must_filters}} if must_filters else None


def classify_intent(query: str) -> str:
    inputs = intent_tokenizer(
        query, return_tensors="pt", truncation=True, padding=True, max_length=128
    ).to(device)
    with torch.no_grad():
        outputs = intent_model(**inputs)

    logits = outputs.logits
    predicted_id = torch.argmax(logits, dim=-1).item()
    try:
        intent = intent_model.config.id2label[predicted_id].upper()
        if intent not in INTENT_CATEGORIES:
            logger.warning(f"Invalid intent '{intent}', falling back to HYBRID")
            return "HYBRID"
        return intent
    except KeyError:
        logger.error("Intent ID not found in id2label, falling back to HYBRID")
        return "HYBRID"


# ==============================================================================
# RASS engine logic
# ==============================================================================
async def ask(
    query: str,
    user_id: str,
    chat_id: str,
    top_k: int = TOP_K,
) -> str:
    # non empty validation check
    if not query.strip():
        return "[ERROR] Empty query."

    if not chat_id:
        return "[ERROR] Incorrect account/chat details!"

    # verify user owns chat
    chat = await db.chat.find_unique(where={"id": chat_id}, include={"user": True})

    if not chat or chat.userId != user_id:
        raise HTTPException(status_code=403, detail="Chat not found or unauthorized")

    #  apply NER and intent classification
    filter_clause = ner_preprocess(query)
    intent = classify_intent(query)
    patient_id = infer_patient_id_from_filename(query) or None
    logger.info(
        f"Intent: {intent}, Filter Clause: {filter_clause}, Patient ID: {patient_id}"
    )

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

    query_emb = await embed_query(query)
    index_name = get_index_name(user_id)
    await ensure_index_exists(os_client, index_name)
    os_indexer = OpenSearchIndexer(os_client, index_name)

    if intent == "DOCUMENT_FETCH":
        results = os_indexer.document_fetch_search(
            query, k=top_k, filter_clause=filter_clause, patient_id=patient_id
        )
        if not results:
            return "No matching documents found."

        patient_files = {}
        for doc, score in results:
            patient_id = doc.get("patientId")
            file_path = doc.get("file_path")
            file_type = doc.get("file_type", FILE_TYPE_JSON)
            if patient_id and file_path:
                if patient_id not in patient_files:
                    patient_files[patient_id] = set()

                patient_files[patient_id].add((file_path, file_type))

        if not patient_files:
            return "No documents with valid patient ID or file path found."

        retrieved_docs = []
        for patient_id, file_info in patient_files.items():
            file_count = 0
            for file_path, file_type in file_info:
                if file_count >= MAX_FILES_PER_PATIENT:
                    logger.warning(
                        f"Reached MAX_FILES_PER_PATIENT ({MAX_FILES_PER_PATIENT}) for patient {patient_id}"
                    )
                    break

                if ehr_doc := await retrieve_ehr_document(file_path):
                    retrieved_docs.append(
                        {
                            "patientId": patient_id,
                            "file_path": file_path,
                            "file_type": file_type,
                            "content": ehr_doc["content"],
                        }
                    )
                    file_count += 1

        if not retrieved_docs:
            return "No accessible documents found for the patient."

        return json.dumps({"patient_records": retrieved_docs}, indent=2)

    search_methods = {
        "SEMANTIC": os_indexer.semantic_search,
        "KEYWORD": os_indexer.exact_match_search,
        "HYBRID": os_indexer.hybrid_search,
        "STRUCTURED": os_indexer.structured_search,
        "HYBRID_STRUCTURED": os_indexer.hybrid_structured_search,
        "AGGREGATE": os_indexer.aggregate_search,
        "COMPARISON": os_indexer.comparison_search,
        "TEMPORAL": os_indexer.temporal_search,
        "EXPLANATORY": os_indexer.explanatory_search,
        "MULTI_INTENT": os_indexer.multi_intent_search,
        "ENTITY_SPECIFIC": os_indexer.entity_specific_search,
    }
    search_method = search_methods.get(intent, os_indexer.hybrid_search)
    if intent == "AGGREGATE":
        result = search_method(
            query, filter_clause=filter_clause, patient_id=patient_id
        )
        return json.dumps(result, indent=2)

    if intent in ["SEMANTIC", "HYBRID", "HYBRID_STRUCTURED", "MULTI_INTENT"]:
        partial_results = search_method(
            query=query,
            query_emb=query_emb,
            k=top_k,
            filter_clause=filter_clause,
            patient_id=patient_id,
        )
    else:
        partial_results = search_method(
            query=query,
            k=top_k,
            filter_clause=filter_clause,
            patient_id=patient_id,
        )

    context_map = {}
    for doc_dict, _score in partial_results:
        doc_id = doc_dict.get("doc_id", "UNKNOWN")

        if doc_dict.get("doc_type", "structured") == "unstructured":
            # just show the unstructuredText - improvement may be planned later
            raw_text = doc_dict.get("unstructuredText", "")
            snippet = f"[Unstructured Text]: {raw_text}"
        else:
            # gather all non-empty fields for "structured" docs
            snippet_pieces = [
                f"{k}={v}"
                for k, v in doc_dict.items()
                if v is not None
                and k not in ["doc_id", "doc_type", "resourceType", "embedding"]
            ]
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
        "You are a helpful medical AI assistant chatbot with access to FHIR-based, markdown, and plain-text EHR data. You must follow these rules and provide accurate and professional answers based on the query and context:\n"
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

    answer = await bluehive_generate_text(prompt=final_prompt, system_msg=system_msg)
    if not answer:
        return "[Error] No response was generated."

    # store new user message + AI response
    current_time = datetime.now(timezone.utc).isoformat()
    await db.message.create_many(
        [
            {
                "chatId": chat_id,
                "role": "user",
                "content": query,
                "createdAt": current_time,
            },
            {
                "chatId": chat_id,
                "role": "assistant",
                "content": answer,
                "createdAt": current_time,
            },
        ]
    )
    return answer


# ====================================
# FastAPI Application Setup
# ==================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Handles database connection lifecycle
    await db.connect()
    print("[Lifespan] Initializing RASSEngine...")

    # Check if OpenSearch has any data
    # if os_indexer.has_any_data():
    #     print("[RASSEngine] OpenSearch already has data. Skipping embedding.")
    # else:
    #     await ingest_fhir_directory(EMB_DIR)
    #     await build_embeddings_from_scratch(EMB_DIR)

    # print("[Lifespan] RASSEngine is ready.")
    yield
    print("[Lifespan] RASSEngine is shutting down...")
    await db.disconnect()


app = FastAPI(
    title="RASS Engine - /ask Query Microservice",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/ask")
async def ask_route(payload: dict = Body(...)):
    """
    RASS endpoint:
      1) user_id, chat_id, query => verify user owns chat
      2) fetch last 10 messages for context
      3) retrieve from OpenSearch => build context
      4) call LLM => store new user query + new response in DB
    """
    query: str = payload.get("query", "")
    user_id = payload.get("user_id", "")
    chat_id = payload.get("chat_id", "")
    top_k = int(payload.get("top_k", TOP_K))

    if not user_id or not chat_id or not query.strip():
        raise HTTPException(status_code=400, detail="Provide user_id, chat_id, query")

    print(
        f"[Debug] query = {query}, user_id={user_id}, chat_id={chat_id}, top_k = {top_k}"
    )

    answer = await ask(query, user_id, chat_id, top_k)
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
                    token = chunk["choices"][0]["delta"].get("content", "")
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
        top_k: int = int(data.get("top_k", TOP_K))

        # validate required fields
        if not query.strip() or not user_id or not chat_id:
            await websocket.send_text(
                json.dumps({"error": "Missing required parameters."})
            )
            await websocket.close()
            return

        logger.info(
            f"[WebSocket Debug] user_id={user_id}, chat_id={chat_id}, query={query}, top_k={top_k}"
        )

        # verify user owns chat
        chat = await db.chat.find_unique(where={"id": chat_id}, include={"user": True})
        if not chat or chat.userId != user_id:
            await websocket.send_text(
                json.dumps({"error": "Chat not found or unauthorized access."})
            )
            await websocket.close()
            return

        # apply NER and intent classification
        filter_clause = ner_preprocess(query)
        intent = classify_intent(query)
        patient_id = infer_patient_id_from_filename(query) or None
        logger.info(
            f"[WebSocket Debug] Intent: {intent}, Filter Clause: {filter_clause}, Patient ID: {patient_id}"
        )

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

        query_emb = await embed_query(query)
        index_name = get_index_name(user_id)
        await ensure_index_exists(os_client, index_name)
        os_indexer = OpenSearchIndexer(os_client, index_name)

        # perform retrieval based on intent
        if intent == "DOCUMENT_FETCH":
            results = os_indexer.document_fetch_search(
                query, k=top_k, filter_clause=filter_clause, patient_id=patient_id
            )
            if not results:
                await websocket.send_text(
                    json.dumps({"answer": "No matching documents found."})
                )
                await websocket.close()
                return

            patient_files = {}
            for doc, score in results:
                patient_id = doc.get("patientId")
                file_path = doc.get("file_path")
                file_type = doc.get("file_type", FILE_TYPE_JSON)
                if patient_id and file_path:
                    if patient_id not in patient_files:
                        patient_files[patient_id] = set()
                    patient_files[patient_id].add((file_path, file_type))

            if not patient_files:
                await websocket.send_text(
                    json.dumps(
                        {
                            "answer": "No documents with valid patient ID or file path found."
                        }
                    )
                )
                await websocket.close()
                return

            retrieved_docs = []
            for patient_id, file_info in patient_files.items():
                file_count = 0
                for file_path, file_type in file_info:
                    if file_count >= MAX_FILES_PER_PATIENT:
                        logger.warning(
                            f"Reached MAX_FILES_PER_PATIENT ({MAX_FILES_PER_PATIENT}) for patient {patient_id}"
                        )
                        break
                    if ehr_doc := await retrieve_ehr_document(file_path):
                        retrieved_docs.append(
                            {
                                "patientId": patient_id,
                                "file_path": file_path,
                                "file_type": file_type,
                                "content": ehr_doc["content"],
                            }
                        )
                        file_count += 1

            if not retrieved_docs:
                await websocket.send_text(
                    json.dumps(
                        {"answer": "No accessible documents found for the patient."}
                    )
                )
                await websocket.close()
                return

            final_answer = json.dumps({"patient_records": retrieved_docs}, indent=2)
            await websocket.send_text(final_answer)

            current_time = datetime.now(timezone.utc).isoformat()
            await db.message.create_many(
                [
                    {
                        "chatId": chat_id,
                        "role": "user",
                        "content": query,
                        "createdAt": current_time,
                    },
                    {
                        "chatId": chat_id,
                        "role": "assistant",
                        "content": final_answer,
                        "createdAt": current_time,
                    },
                ]
            )
            await websocket.close()
            return

        search_methods = {
            "SEMANTIC": os_indexer.semantic_search,
            "KEYWORD": os_indexer.exact_match_search,
            "HYBRID": os_indexer.hybrid_search,
            "STRUCTURED": os_indexer.structured_search,
            "HYBRID_STRUCTURED": os_indexer.hybrid_structured_search,
            "AGGREGATE": os_indexer.aggregate_search,
            "COMPARISON": os_indexer.comparison_search,
            "TEMPORAL": os_indexer.temporal_search,
            "EXPLANATORY": os_indexer.explanatory_search,
            "MULTI_INTENT": os_indexer.multi_intent_search,
            "ENTITY_SPECIFIC": os_indexer.entity_specific_search,
        }
        search_method = search_methods.get(intent, os_indexer.hybrid_search)

        if intent == "AGGREGATE":
            result = search_method(
                query, filter_clause=filter_clause, patient_id=patient_id
            )
            final_answer = json.dumps(result, indent=2)
            await websocket.send_text(final_answer)

            current_time = datetime.now(timezone.utc).isoformat()
            await db.message.create_many(
                [
                    {
                        "chatId": chat_id,
                        "role": "user",
                        "content": query,
                        "createdAt": current_time,
                    },
                    {
                        "chatId": chat_id,
                        "role": "assistant",
                        "content": final_answer,
                        "createdAt": current_time,
                    },
                ]
            )
            await websocket.close()
            return

        if intent in ["SEMANTIC", "HYBRID", "HYBRID_STRUCTURED", "MULTI_INTENT"]:
            partial_results = search_method(
                query=query,
                query_emb=query_emb,
                k=top_k,
                filter_clause=filter_clause,
                patient_id=patient_id,
            )
        else:
            partial_results = search_method(
                query=query,
                k=top_k,
                filter_clause=filter_clause,
                patient_id=patient_id,
            )

        context_map = {}
        for doc_dict, _score in partial_results:
            doc_id = doc_dict.get("doc_id", "UNKNOWN")

            if doc_dict.get("doc_type", "structured") == "unstructured":
                raw_text = doc_dict.get("unstructuredText", "")
                snippet = f"[Unstructured Text]: {raw_text}"
            else:
                snippet_pieces = [
                    f"{k}={v}"
                    for k, v in doc_dict.items()
                    if v is not None
                    and k not in ["doc_id", "doc_type", "resourceType", "embedding"]
                ]
                snippet = "[Structured Resource] " + " | ".join(snippet_pieces)

            if doc_id not in context_map:
                context_map[doc_id] = snippet
            else:
                context_map[doc_id] += "\n" + snippet

        context_text = ""
        for doc_id, doc_content in context_map.items():
            context_text += f"--- Document ID: {doc_id} ---\n{doc_content}\n\n"

        system_msg = (
            "You are a helpful medical AI assistant chatbot with access to FHIR-based, markdown, and plain-text EHR data. You must follow these rules and provide accurate and professional answers based on the query and context:\n"
            "1) Always cite document IDs from the context exactly as 'Document XYZ' without any file extensions like '.txt'.\n"
            "2) For every answer generated, there should be a reference or citation of the IDs of the documents from which the answer information was extracted at the end of the answer!\n"
            "3) If the context does not relate to the query, say 'I lack the context to answer your question.' For example, if the query is about gene mutations but the context is about climate change, acknowledge the mismatch and do not answer.\n"
            "4) Never ever give responses based on your own knowledge of the user query. Use ONLY the provided context + chat history to extract information relevant to the question. You should not answer without document ID references from which the information was extracted.\n"
            "5) If you lack context, then say so.\n"
            "6) Do not add chain-of-thought.\n"
        )
        final_prompt = (
            f"Chat History:\n{chat_history_str}\n\n"
            f"User Query:\n{query}\n\n"
            f"Context:\n{context_text}\n"
            "--- End of context ---\n\n"
            "Provide your concise answer now."
        )

        streamed_chunks = []
        async for chunk in openai_generate_text_stream(final_prompt, system_msg):
            streamed_chunks.append(chunk)
            await websocket.send_text(chunk)

        final_answer = "".join(streamed_chunks).strip()
        if final_answer:
            current_time = datetime.now(timezone.utc).isoformat()
            await db.message.create_many(
                [
                    {
                        "chatId": chat_id,
                        "role": "user",
                        "content": query,
                        "createdAt": current_time,
                    },
                    {
                        "chatId": chat_id,
                        "role": "assistant",
                        "content": final_answer,
                        "createdAt": current_time,
                    },
                ]
            )

        await websocket.close()

    except WebSocketDisconnect:
        logger.info("[WebSocket] Client disconnected mid-stream.")
    except Exception as e:
        logger.error(f"[WebSocket] Unexpected error: {e}")
        await websocket.send_text(
            json.dumps({"error": "Internal server error occurred."})
        )
        await websocket.close()


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
