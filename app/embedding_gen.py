import os
import shutil
import asyncio
import uvicorn
import numpy as np
import time
import pathlib
import json
from typing import List, Dict, Tuple, Optional

import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.helpers import bulk
from dotenv import load_dotenv
from prisma import Prisma
from werkzeug.utils import secure_filename

import markdown
from bs4 import BeautifulSoup

import logging
from pathlib import Path
import re
from uuid import uuid4
import tempfile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

##########################################
# Load environment variables
#################################
load_dotenv()

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "")
os.makedirs(UPLOAD_DIR, exist_ok=True)

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api")
EMBED_MODEL_NAME = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large:latest")
MAX_EMBED_CONCURRENCY = int(os.getenv("MAX_EMBED_CONCURRENCY", "5"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))

OPENSEARCH_INDEX_NAME = os.getenv("OPENSEARCH_INDEX_NAME", "")
SHARD_COUNT = int(os.getenv("SHARD_COUNT", 1))
REPLICA_COUNT = int(os.getenv("REPLICA_COUNT", 0))
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "9200"))

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "")
POSTGRES_DB = os.getenv("POSTGRES_DB", "")
POSTGRES_USER = os.getenv("POSTGRES_USER", "")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))

SUPPORTED_FILE_EXTENSIONS = (".json", ".md", ".txt")
FILE_TYPE_JSON = "json"
FILE_TYPE_MARKDOWN = "markdown"
FILE_TYPE_TEXT = "text"

MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10 MB
MAX_FILES_PER_REQUEST = int(os.getenv("MAX_FILES_PER_REQUEST", "5"))
MAX_CONCURRENT_FILES = int(os.getenv("MAX_CONCURRENT_FILES", "5"))

db = Prisma()


######################################
# FastAPI app
########################################
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[Lifespan] Starting Embedding Service...")
    await db.connect()
    logger.info("[Lifespan] Embedding Service is ready.")
    yield
    logger.info("[Lifespan] Embedding Service is shutting down...")
    await db.disconnect()


app = FastAPI(
    title="Embedding Service to handle file uploads",
    version="1.0.0",
    lifespan=lifespan,
)


############################################
# Markdown file processing
########################################
def parse_markdown_file(path: str) -> str:
    """
    Reads a Markdown (.md) file from the given path, converts it to HTML using the
    python-markdown library, and then uses BeautifulSoup to extract plain text.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            md_content = f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            md_content = f.read()

    # Convert the Markdown content to HTML
    html = markdown.markdown(md_content)
    # Use BeautifulSoup to parse the HTML and extract plain text
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ")
    return text


##############################################
# OpenSearch
########################################
try:
    os_client = OpenSearch(
        hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
        http_compress=True,
        use_ssl=False,
        verify_certs=False,
        connection_class=RequestsHttpConnection,
    )
    info = os_client.info()
    logger.info(f"Connected to OpenSearch: {info['version']}")
except Exception as e:
    logger.error(f"Could not connect to OpenSearch: {e}")
    os_client = None


###############################################################################
# Helper to chunk text
###############################################################################
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk_str = " ".join(words[i : i + chunk_size])
        chunks.append(chunk_str.strip())

    return chunks


###############################################################################
# Embedding with Ollama
###############################################################################
async def ollama_embed_text(text: str) -> List[float]:
    if not text.strip():
        return [0.0] * EMBED_DIM
    async with httpx.AsyncClient() as client:
        payload = {"model": EMBED_MODEL_NAME, "prompt": text, "stream": False}
        try:
            resp = await client.post(
                f"{OLLAMA_API_URL}/embeddings", json=payload, timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
            emb = data.get("embedding", [])
            if len(emb) != EMBED_DIM:
                print("[WARN] embedding length mismatch:", len(emb), "vs", EMBED_DIM)

            return emb
        except Exception as ex:
            print("[ERROR] Ollama embed request:", ex)
            return [0.0] * EMBED_DIM


async def embed_texts_in_batches(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)

    all_embeds = []
    concurrency_sem = asyncio.Semaphore(MAX_EMBED_CONCURRENCY)

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]

        async def run_embed(txt: str) -> List[float]:
            async with concurrency_sem:
                return await ollama_embed_text(txt)

        tasks = [run_embed(t) for t in batch]
        results = await asyncio.gather(*tasks)
        all_embeds.extend(results)

    arr = np.array(all_embeds, dtype=np.float32)
    return arr


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
    embeddings = await embed_texts_in_batches(un_texts)

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


###############################################################################
# PostgreSQL DB Async Connection
###############################################################################
async def check_user_authorized(user_id: str) -> bool:
    user = await db.user.find_unique(where={"id": user_id})
    return bool(user)


###############################################################################
# Validation Helpers
###############################################################################
def validate_user_id(user_id: str) -> bool:
    """Validate user ID: alphanumeric or UUID."""
    pattern = r"^[a-zA-Z0-9_-]{1,36}$|^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    return bool(re.match(pattern, user_id))


async def read_file_content(file: UploadFile, max_size: int) -> bytes:
    """Read file content with size limit."""
    content = bytearray()
    while chunk := await file.read(8192):  # 8KB chunks
        content.extend(chunk)
        if len(content) > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File {file.filename} exceeds size limit of {max_size} bytes",
            )

    return bytes(content)


###############################################################
# Endpoint: upload data => handle EHR data
#############################################################
@app.post("/upload_data")
async def upload_data(user_id: str = Form(...), files: List[UploadFile] = File(...)):
    """
    Endpoint to upload and index files (txt, md, json/FHIR) for RAG/RASS engine.
    Parses files into structured and unstructured documents, embeds and stores unstructured text.
    """
    if not validate_user_id(user_id):
        logger.warning(f"Invalid user ID format: {user_id}")
        raise HTTPException(status_code=400, detail="Invalid user ID format")

    authorized = await check_user_authorized(user_id)
    if not authorized:
        logger.warning(f"Unauthorized access attempt by user: {user_id}")
        raise HTTPException(status_code=403, detail="User not authorized")

    # Validate file count
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    if len(files) > MAX_FILES_PER_REQUEST:
        logger.warning(
            f"Too many files uploaded: {len(files)} > {MAX_FILES_PER_REQUEST}"
        )
        raise HTTPException(
            status_code=400,
            detail=f"Too many files uploaded (max {MAX_FILES_PER_REQUEST})",
        )

    # Prepare user folder
    user_folder = os.path.join(UPLOAD_DIR, secure_filename(user_id))
    os.makedirs(user_folder, exist_ok=True)
    index_name = get_index_name(user_id)
    await ensure_index_exists(os_client, index_name)

    processed_files = 0
    concurrency_sem = asyncio.Semaphore(MAX_CONCURRENT_FILES)

    async def process_file(file: UploadFile) -> Tuple[int, List[Dict], List[Dict]]:
        async with concurrency_sem:
            # Sanitize filename
            filename = secure_filename(file.filename)
            extension = pathlib.Path(filename).suffix.lower()
            if extension not in SUPPORTED_FILE_EXTENSIONS:
                logger.warning(f"Unsupported file type: {filename}")
                return 0, [], []

            # Read and validate file size
            try:
                content = await read_file_content(file, MAX_FILE_SIZE)
            except HTTPException as e:
                logger.error(f"File size error for {filename}: {e.detail}")
                raise e
            except Exception as e:
                logger.error(f"Error reading file {filename}: {e}")
                return 0, [], []

            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            try:
                # Validate file path
                valid_path = validate_file_path(tmp_path, read=True)
                if not valid_path:
                    logger.error(f"Invalid file after saving: {filename}")
                    return 0, [], []

                # Generate unique doc ID
                doc_id = f"{pathlib.Path(filename).stem}_{uuid4().hex[:8]}"
                final_path = os.path.join(user_folder, f"{doc_id}{extension}")

                # Process file based on type
                structured_docs, unstructured_docs = [], []
                if extension == ".json":
                    try:
                        content_str = content.decode("utf-8")
                        bundle_json = json.loads(content_str)
                        if (
                            not isinstance(bundle_json, dict)
                            or "entry" not in bundle_json
                        ):
                            logger.error(f"Invalid FHIR bundle: {filename}")
                            raise HTTPException(
                                status_code=400, detail="Invalid FHIR bundle"
                            )

                        structured_docs, unstructured_docs = (
                            parse_fhir_bundle_with_path(bundle_json, final_path)
                        )
                    except UnicodeDecodeError:
                        content_str = content.decode("latin-1")
                        bundle_json = json.loads(content_str)
                        structured_docs, unstructured_docs = (
                            parse_fhir_bundle_with_path(bundle_json, final_path)
                        )
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON parse error for {filename}: {e}")
                        raise HTTPException(
                            status_code=400, detail=f"JSON parse error: {e}"
                        )
                else:
                    file_type = (
                        FILE_TYPE_MARKDOWN if extension == ".md" else FILE_TYPE_TEXT
                    )
                    structured_docs, unstructured_docs = parse_text_file(
                        tmp_path, file_type
                    )
                    if not unstructured_docs and not structured_docs:
                        logger.warning(f"No data extracted from {filename}")
                        return 0, [], []

                # Move temporary file to final location
                shutil.move(tmp_path, final_path)
                logger.info(f"Successfully processed file: {filename}")
                return 1, structured_docs, unstructured_docs
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
                return 0, [], []
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

    # Process files concurrently
    tasks = [process_file(file) for file in files]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Collect all documents for batch indexing
    all_structured_docs = []
    all_unstructured_docs = []
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Task failed: {result}")
            continue

        count, structured, unstructured = result
        processed_files += count
        all_structured_docs.extend(structured)
        all_unstructured_docs.extend(unstructured)

    # Batch index all documents
    if all_structured_docs or all_unstructured_docs:
        await store_fhir_docs_in_opensearch(
            all_structured_docs, all_unstructured_docs, os_client, index_name
        )

    if processed_files == 0:
        logger.warning("No valid files were processed")
        raise HTTPException(status_code=400, detail="No valid files were processed")

    return {
        "message": f"Uploaded and indexed {processed_files} file(s) for user={user_id}"
    }


if __name__ == "__main__":
    uvicorn.run("embedding_gen:app", host="0.0.0.0", port=8001, reload=False)
