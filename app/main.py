import os
from dotenv import load_dotenv
import json
import numpy as np
from typing import List, Tuple, Optional, Dict, AsyncGenerator

import asyncio
from contextlib import asynccontextmanager

import httpx
import uvicorn

import concurrent.futures

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

EMB_DIR = os.getenv("EMB_DIR", "sample_dataset")

OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", 9200))
OPENSEARCH_INDEX_NAME = os.getenv("OPENSEARCH_INDEX_NAME", "")
TOP_K = int(os.getenv("TOP_K", "3"))

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
                    # storing the text in 'unstructuredText' field
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

    await ensure_index_exists(client, index_name)

    # Bulk index structured docs
    bulk_actions_struct = [
        {
            "_op_type": "index",
            "_index": index_name,
            "_id": doc["doc_id"],
            "_source": doc,
        }
        for doc in structured_docs
    ]

    if bulk_actions_struct:
        s_success, s_errors = bulk(client, bulk_actions_struct)
        print(f"[FHIR] Indexed {s_success} structured docs. Errors: {s_errors}")

    if not unstructured_docs:
        return

    # embed the unstructured text, then store each doc with "unstructuredText" plus the "embedding" field.
    un_texts = [d["unstructuredText"] for d in unstructured_docs]
    embeddings = await embed_texts_in_batches(un_texts, batch_size=BATCH_SIZE)

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_normed = embeddings / (norms + 1e-9)

    bulk_actions_unstruct = []
    for i, docu in enumerate(unstructured_docs):
        # store as float array - emb_vector
        docu["embedding"] = embeddings_normed[i].tolist()
        # storing docu in index using the doc_id as _id
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
        if not self.client:
            print(
                "[OpenSearchIndexer - add_embeddings] No OpenSearch client available."
            )
            return

        if embeddings.size == 0 or not docs:
            print("[OpenSearchIndexer - add_embeddings] No embeddings or empty docs.")
            return

        if embeddings.shape[0] != len(docs):
            print(
                f"[OpenSearchIndexer - add_embeddings] Mismatch: embeddings={embeddings.shape[0]}, docs={len(docs)}"
            )
            return

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-9)

        actions = []
        for i, (doc_dict, emb) in enumerate(zip(docs, embeddings)):
            doc_id = doc_dict.get("doc_id", "")
            unstructured_text = doc_dict.get("unstructuredText", "")
            if not unstructured_text and "text" in doc_dict:
                unstructured_text = doc_dict["text"]

            source_body = {
                "doc_id": doc_id,
                "doc_type": "unstructured",
                "unstructuredText": unstructured_text,
                "embedding": emb.tolist(),
            }
            actions.append(
                {
                    "_op_type": "index",
                    "_index": self.index_name,
                    "_id": f"{doc_id}_{i}",
                    "_source": source_body,
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
        self, query_text: str, k: int = TOP_K
    ) -> List[Tuple[Dict[str, str], float]]:
        if not self.client or not query_text.strip():
            return []

        text_fields = [
            "unstructuredText^3",  # Boost unstructured text, other fields similarly ...
            "patientName^3",
            "patientAddress^3",
            "patientTelecom^3",
            "conditionCodeText^2",  # Boost condition text, other fields similarly ...
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

        keyword_fields = [
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

        # text-based subquery (only text fields)
        text_subquery = {
            "multi_match": {
                "query": query_text,
                "fields": text_fields,
                "type": "best_fields",
                "operator": "or",  # relaxed to find broader matches
                "fuzziness": "AUTO",
            }
        }

        # keyword subquery (exact matches on keyword fields)
        keyword_subquery = {
            "multi_match": {
                "query": query_text,
                "fields": keyword_fields,
                "type": "best_fields",
                "operator": "or",
            }
        }

        # patient-specific filter for queries mentioning 'patients'
        if "patient" in query_text.lower():
            bool_query["filter"] = [{"exists": {"field": "patientId"}}]

        # build the query
        bool_query = {
            "should": [
                text_subquery,
                keyword_subquery,
            ],
            "minimum_should_match": 1,  # At least one condition must match
        }

        query_body = {
            "size": k,
            "query": {
                "bool": bool_query,
            },
        }

        try:
            resp = self.client.search(index=self.index_name, body=query_body)
            return [
                (hit["_source"], float(hit["_score"])) for hit in resp["hits"]["hits"]
            ]
        except Exception as e:
            print(f"[OpenSearchIndexer - exact_match_search] Error: {e}")
            return []

    def semantic_search(
        self, query_emb: np.ndarray, k: int = TOP_K, filter_clause: Dict = None
    ) -> List[Tuple[Dict[str, str], float]]:
        if not self.client or query_emb.size == 0:
            return []

        norms = np.linalg.norm(query_emb, axis=1, keepdims=True)
        query_emb = query_emb / (norms + 1e-9)
        vector = query_emb[0].tolist()

        if filter_clause:
            query_body = {
                "size": k,
                "query": {
                    "bool": {
                        "must": [
                            {"knn": {"embedding": {"vector": vector, "k": k}}},
                            filter_clause,
                        ]
                    }
                },
            }
        else:
            query_body = {
                "size": k,
                "query": {"knn": {"embedding": {"vector": vector, "k": k}}},
            }

        try:
            resp = self.client.search(index=self.index_name, body=query_body)
            return [
                (hit["_source"], float(hit["_score"])) for hit in resp["hits"]["hits"]
            ]
        except Exception as e:
            print(f"[OpenSearchIndexer - semantic_search] Error: {e}")
            return []

    def hybrid_search(
        self,
        query_text: str,
        query_emb: np.ndarray,
        k: int = TOP_K,
        filter_clause: Dict = None,
    ) -> List[Tuple[Dict[str, str], float]]:
        if not self.client or not query_text.strip() or query_emb.size == 0:
            return []

        norms = np.linalg.norm(query_emb, axis=1, keepdims=True)
        query_emb = query_emb / (norms + 1e-9)
        vector = query_emb[0].tolist()

        text_fields = [
            "unstructuredText^3",  # Boost unstructured text, ....
            "patientName^3",
            "patientAddress^3",
            "patientTelecom^3",
            "conditionCodeText^2",  # Boost condition text, ...
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

        keyword_fields = [
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

        knn_subquery = {"knn": {"embedding": {"vector": vector, "k": k}}}

        text_subquery = {
            "multi_match": {
                "query": query_text,
                "fields": text_fields,
                "type": "best_fields",
                "operator": "or",
                "fuzziness": "AUTO",
            }
        }

        keyword_subquery = {
            "multi_match": {
                "query": query_text,
                "fields": keyword_fields,
                "type": "best_fields",
                "operator": "or",
            }
        }

        bool_query = {
            "should": [
                text_subquery,
                keyword_subquery,
                knn_subquery,
            ],
            "minimum_should_match": 1,
        }

        if filter_clause or "patient" in query_text.lower():
            bool_query["filter"] = []
            if filter_clause:
                bool_query["filter"].append(filter_clause)
            if "paient" in query_text.lower():
                bool_query["filter"].append({"exists": {"field": "patientId"}})

        query_body = {
            "size": k,
            "query": {"bool": bool_query},
        }

        try:
            resp = self.client.search(index=self.index_name, body=query_body)
            return [
                (hit["_source"], float(hit["_score"])) for hit in resp["hits"]["hits"]
            ]
        except Exception as e:
            print(f"[OpenSearchIndexer - hybrid_search] Error: {e}")
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
        return await loop.run_in_executor(
            self._executor, self._sync_classify, query
        )  # label


intent_classifier = ZeroShotQueryClassifier()


# ==============================================================================
# RASS engine logic
# ==============================================================================
"""
RASS engine:
    - Classifies query intent using DistilBERT
    - Uses appropriate search method (exact, semantic, or hybrid)
    - Uses BlueHive for final response
    - Saves user queries & answers in Postgres (via Prisma)
"""


async def build_embeddings_from_scratch(pmc_dir: str, user_id: str) -> None:
    index_name = get_index_name(user_id)
    os_indexer = OpenSearchIndexer(os_client, index_name)
    await ensure_index_exists(os_client, index_name)

    if not os_indexer:
        print(
            "[build_embeddings_from_scratch] No OpenSearchIndexer => cannot build embeddings."
        )
        return

    print("[build_embeddings_from_scratch] Building embeddings from scratch...")
    all_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(pmc_dir)
        for f in files
        if f.endswith(".txt") and f.startswith("PMC")
    ]

    all_docs = []
    for path in all_files:
        base_name = os.path.basename(path)
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(path, "r", encoding="latin-1") as f:
                text = f.read()

        cleaned_text = basic_cleaning(text)
        text_chunks = chunk_text(cleaned_text, CHUNK_SIZE)
        all_docs.extend(
            {"doc_id": base_name, "unstructuredText": chunk_str}
            for chunk_str in text_chunks
        )

    if not all_docs:
        print("[build_embeddings_from_scratch] No text found in directory. Exiting.")
        return

    print(
        f"[build_embeddings_from_scratch] Generating embeddings for {len(all_docs)} chunks..."
    )
    chunk_texts = [d["unstructuredText"] for d in all_docs]
    embs = await embed_texts_in_batches(chunk_texts, batch_size=BATCH_SIZE)
    os_indexer.add_embeddings(embs, all_docs)


async def ingest_fhir_directory(fhir_dir: str, user_id: str) -> None:
    """
    Enumerate all FHIR Json documents in fhir_dir, parse them, and store.
    """
    index_name = get_index_name(user_id)
    os_indexer = OpenSearchIndexer(os_client, index_name)
    await ensure_index_exists(os_client, index_name)

    if not os_indexer:
        print("[ingest_fhir_directory] No OS indexer => cannot ingest FHIR data.")
        return

    all_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(fhir_dir)
        for f in files
        if f.endswith(".json")
    ]
    if not all_files:
        print(f"[ingest_fhir_directory] No .json files found in {fhir_dir}.")
        return

    for path in all_files:
        print(f"[ingest_fhir_directory] Processing FHIR file: {path}")
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
            os_client,
            index_name,
        )


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

    query_emb = await embed_query(query)
    index_name = get_index_name(user_id)
    await ensure_index_exists(os_client, index_name)
    os_indexer = OpenSearchIndexer(os_client, index_name)

    # OpenSearch Retrieval
    if intent == "SEMANTIC":
        partial_results = os_indexer.semantic_search(query_emb, k=top_k)
    elif intent == "KEYWORD":
        partial_results = os_indexer.exact_match_search(query, k=top_k)
    else:
        partial_results = os_indexer.hybrid_search(query, query_emb, k=top_k)

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
                if v and k not in ["doc_id", "doc_type", "resourceType", "embedding"]
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

    answer = await bluehive_generate_text(prompt=final_prompt, system_msg=system_msg)
    if not answer:
        return "[Error] No response was generated."

    # store new user message + AI response
    await db.message.create(data={"chatId": chat_id, "role": "user", "content": query})
    await db.message.create(
        data={"chatId": chat_id, "role": "assistant", "content": answer}
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

        print(
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

        # zero-shot classify intent asynchronously
        intent = await intent_classifier.classify_intent(query)
        print(f"[WebSocket Debug] Query Intent = {intent}")

        index_name = get_index_name(user_id)
        await ensure_index_exists(os_client, index_name)
        os_indexer = OpenSearchIndexer(os_client, index_name)

        # perform retrieval based on intent
        if intent == "SEMANTIC":
            partial_results = os_indexer.semantic_search(query_emb, k=top_k)
        elif intent == "KEYWORD":
            partial_results = os_indexer.exact_match_search(query, k=top_k)
        else:  # hyb
            partial_results = os_indexer.hybrid_search(query, query_emb, k=top_k)

        context_map = {}
        for doc_dict, _score in partial_results:
            doc_id = doc_dict.get("doc_id", "UNKNOWN")

            if doc_dict.get("doc_type", "structured") == "unstructured":
                # just show the unstructuredText - improvement may be planned later
                snippet = f"[Unstructured Text]: {doc_dict.get('unstructuredText', '')}"
            else:
                # gather all non-empty fields for "structured" docs
                snippet_pieces = [
                    f"{k}={v}"
                    for k, v in doc_dict.items()
                    if v
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
