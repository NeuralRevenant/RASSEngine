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

import markdown
from bs4 import BeautifulSoup


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

OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "9200"))

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "")
POSTGRES_DB = os.getenv("POSTGRES_DB", "")
POSTGRES_USER = os.getenv("POSTGRES_USER", "")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))

db = Prisma()


######################################
# FastAPI app
########################################
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Lifespan] Starting Embedding Service...")
    # Handles database connection lifecycle
    await db.connect()
    print("[Lifespan] Embedding Service is ready.")
    yield
    print("[Lifespan] Embedding Service is shutting down...")
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
    print("[INFO] Connected to OpenSearch:", info["version"])
except Exception as e:
    print("[ERROR] Could not connect to OpenSearch:", e)
    os_client = None


#############################################################
# Create user-specific index
############################################################
def init_user_index(user_id: str):
    if not os_client:
        print("[WARNING] No OpenSearch client => cannot init user index.")
        return

    index_name = f"{OPENSEARCH_INDEX_NAME}-{user_id}"
    if os_client.indices.exists(index_name):
        print(f"[INFO] Index already exists: {index_name}")
        return

    mapping_body = {
        "settings": {"index": {"knn": True}},
        "mappings": {
            "properties": {
                "doc_id": {"type": "keyword"},
                "doc_type": {"type": "keyword"},
                "resourceType": {"type": "keyword"},
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
                "procedureId": {"type": "keyword"},
                "procedureCodeText": {"type": "text"},
                "procedureStatus": {"type": "keyword"},
                "procedurePerformedDateTime": {
                    "type": "date",
                    "format": "date_time||strict_date_optional_time||epoch_millis",
                },
                "procedureFollowUp": {"type": "text"},
                "procedureNote": {"type": "text"},
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
                "practitionerId": {"type": "keyword"},
                "practitionerName": {"type": "text"},
                "practitionerGender": {"type": "keyword"},
                "practitionerSpecialty": {"type": "keyword"},
                "practitionerAddress": {"type": "text"},
                "practitionerTelecom": {"type": "text"},
                "organizationId": {"type": "keyword"},
                "organizationName": {"type": "text"},
                "organizationType": {"type": "keyword"},
                "organizationAddress": {"type": "text"},
                "organizationTelecom": {"type": "text"},
                "unstructuredText": {"type": "text"},
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
    try:
        os_client.indices.create(index=index_name, body=mapping_body)
        print(f"[INFO] Created user-specific index: {index_name}")
    except Exception as exc:
        print(f"[ERROR] Could not create index {index_name}: {exc}")


###############################################################################
# PostgreSQL DB Async Connection
###############################################################################
async def check_user_authorized(user_id: str) -> bool:
    user = await db.user.find_unique(where={"id": user_id})
    if user:
        return True

    return False


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


################################################################
# Bulk indexing unstructured chunks
##########################################################
def bulk_index_unstructured(
    user_id: str,
    doc_id: str,
    text_chunks: List[str],
    embeddings: np.ndarray,
    resourceType: Optional[str] = None,
):
    if not os_client:
        print("[ERROR] No OpenSearch => cannot index unstructured docs.")
        return

    index_name = f"{OPENSEARCH_INDEX_NAME}-{user_id}"
    init_user_index(user_id)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-9)

    actions = []
    for i, chunk in enumerate(text_chunks):
        chunk_vec = embeddings[i].tolist()
        chunk_id = f"{doc_id}_chunk_{i}"
        body_doc = {
            "doc_id": doc_id,
            "doc_type": "unstructured",
            "unstructuredText": chunk,
            "embedding": chunk_vec,
        }
        if resourceType:
            body_doc["resourceType"] = resourceType

        actions.append(
            {
                "_op_type": "index",
                "_index": index_name,
                "_id": chunk_id,
                "_source": body_doc,
            }
        )

        if len(actions) >= BATCH_SIZE:
            try:
                success, errors = bulk(os_client, actions)
                if errors:
                    print(
                        "[OpenSearch] partial errors on unstructured indexing:", errors
                    )
            except Exception as e:
                print("[OpenSearch] bulk error unstructured:", e)

            actions = []

    if actions:
        try:
            success, errors = bulk(os_client, actions)
            if errors:
                print("[OpenSearch] partial errors on unstructured indexing:", errors)
        except Exception as e:
            print("[OpenSearch] bulk error unstructured:", e)


###############################################################################
# Bulk indexing structured docs
###############################################################################
def bulk_index_structured(user_id: str, doc_id: str, structured_docs: List[Dict]):
    if not os_client:
        print("[ERROR] No OpenSearch => cannot index structured docs.")
        return

    index_name = f"{OPENSEARCH_INDEX_NAME}-{user_id}"
    init_user_index(user_id)

    actions = []
    for i, sdoc in enumerate(structured_docs):
        if "doc_type" not in sdoc:
            sdoc["doc_type"] = "structured"

        final_id = sdoc.get("doc_id", f"{doc_id}_structured_{i}")
        actions.append(
            {
                "_op_type": "index",
                "_index": index_name,
                "_id": final_id,
                "_source": sdoc,
            }
        )

        if len(actions) >= BATCH_SIZE:
            try:
                success, errors = bulk(os_client, actions)
                if errors:
                    print("[OpenSearch] partial errors on structured indexing:", errors)
            except Exception as ex:
                print("[OpenSearch] bulk error structured:", ex)
            actions = []

    if actions:
        try:
            success, errors = bulk(os_client, actions)
            if errors:
                print("[OpenSearch] partial errors on structured indexing:", errors)
        except Exception as ex:
            print("[OpenSearch] bulk error structured:", ex)


################################################
# FHIR Parser
################################################
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
                unstructured_docs.append(
                    {
                        "doc_id": f"{rtype}-{rid}-unstructured-{c_i}",
                        "doc_type": "unstructured",
                        "resourceType": rtype,
                        # storing the text in 'unstructuredText' field
                        "unstructuredText": chunk_str,
                    }
                )

    return (structured_docs, unstructured_docs)


###############################################################
# Endpoint: upload data => handle text (or) Json/FHIR
#############################################################
@app.post("/upload_data")
async def upload_data(user_id: str = Form(...), files: List[UploadFile] = File(...)):
    authorized = await check_user_authorized(user_id)
    if not authorized:
        raise HTTPException(
            status_code=403, detail=f"User '{user_id}' not authorized in Postgres."
        )

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    user_folder = os.path.join(UPLOAD_DIR, user_id)
    os.makedirs(user_folder, exist_ok=True)

    for upl_file in files:
        extension = pathlib.Path(upl_file.filename).suffix.lower()
        name_stem = pathlib.Path(upl_file.filename).stem
        now_ts = int(time.time())
        doc_id = f"{name_stem}_{now_ts}"
        final_filename = f"{doc_id}{extension}"
        final_path = os.path.join(user_folder, final_filename)

        try:
            with open(final_path, "wb") as bf:
                shutil.copyfileobj(upl_file.file, bf)
        except Exception as ex:
            raise HTTPException(status_code=500, detail=f"File save error: {ex}")

        if extension == ".txt":
            text_str = ""
            try:
                with open(final_path, "r", encoding="utf-8") as ff:
                    text_str = ff.read()
            except UnicodeDecodeError:
                with open(final_path, "r", encoding="latin-1") as ff:
                    text_str = ff.read()

            if not text_str.strip():
                raise HTTPException(
                    status_code=400, detail=f"File {upl_file.filename} is empty."
                )

            txt_chunks = chunk_text(text_str, CHUNK_SIZE)
            txt_embs = await embed_texts_in_batches(txt_chunks)
            if txt_embs.shape[0] != len(txt_chunks):
                raise HTTPException(
                    status_code=500, detail="Mismatch chunk vs embedding count."
                )

            bulk_index_unstructured(
                user_id, doc_id, txt_chunks, txt_embs, resourceType=None
            )

        elif (
            extension == ".md"
        ):  # for Markdown files, convert to plain text using our helper function
            md_text = ""
            try:
                md_text = parse_markdown_file(final_path)
            except Exception as ex:
                raise HTTPException(
                    status_code=500, detail=f"Markdown parse error: {ex}"
                )

            if not md_text.strip():
                raise HTTPException(
                    status_code=400,
                    detail=f"File {upl_file.filename} is empty after parsing.",
                )

            md_chunks = chunk_text(md_text, CHUNK_SIZE)
            md_embs = await embed_texts_in_batches(md_chunks)
            if md_embs.shape[0] != len(md_chunks):
                raise HTTPException(
                    status_code=500,
                    detail="Mismatch chunk vs embedding count for Markdown file.",
                )

            bulk_index_unstructured(
                user_id, doc_id, md_chunks, md_embs, resourceType=None
            )

        elif extension == ".json":
            try:
                with open(final_path, "r", encoding="utf-8") as ff:
                    bundle_json = json.load(ff)
            except UnicodeDecodeError:
                with open(final_path, "r", encoding="latin-1") as ff:
                    bundle_json = json.load(ff)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"JSON parse error: {e}")

            structured_docs, unstructured_docs = parse_fhir_bundle(bundle_json)
            bulk_index_structured(user_id, doc_id, structured_docs)
            for udoc in unstructured_docs:
                text = udoc.get("unstructuredText", "")
                resType = udoc.get("resourceType", "")
                sub_doc_id = udoc.get("doc_id", f"{doc_id}_{resType}")
                if text.strip():
                    un_chunks = chunk_text(text, CHUNK_SIZE)
                    un_embs = await embed_texts_in_batches(un_chunks)
                    if un_embs.shape[0] != len(un_chunks):
                        raise HTTPException(
                            status_code=500,
                            detail="Mismatch chunk vs embedding count for unstructured text in FHIR.",
                        )

                    bulk_index_unstructured(
                        user_id, sub_doc_id, un_chunks, un_embs, resourceType=resType
                    )

        else:
            raise HTTPException(
                status_code=400, detail=f"Unsupported file type: {extension}"
            )

    return {"message": f"Uploaded and indexed {len(files)} file(s) for user={user_id}"}


if __name__ == "__main__":
    uvicorn.run("embedding_gen:app", host="0.0.0.0", port=8001, reload=False)
