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
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.helpers import bulk
from dotenv import load_dotenv
from prisma import Prisma

import markdown
from bs4 import BeautifulSoup


##########################################
# Load environment variables
#############################
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

######################################
# FastAPI app
########################################
app = FastAPI(
    title="Embedding Microservice",
    version="1.0.0",
)


db = Prisma()


@app.on_event("startup")
async def on_startup():
    await db.connect()


@app.on_event("shutdown")
async def on_shutdown():
    await db.disconnect()


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

        action = {
            "_op_type": "index",
            "_index": index_name,
            "_id": chunk_id,
            "_source": body_doc,
        }
        actions.append(action)

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
        action = {
            "_op_type": "index",
            "_index": index_name,
            "_id": final_id,
            "_source": sdoc,
        }
        actions.append(action)

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
def parse_fhir_bundle(bundle_json: Dict) -> Tuple[List[Dict], List[Tuple[str, str]]]:
    structured_docs = []
    unstructured_data = []

    if not bundle_json or "entry" not in bundle_json:
        return structured_docs, unstructured_data

    for entry in bundle_json["entry"]:
        resource = entry.get("resource", {})
        rtype = resource.get("resourceType", "")
        rid = resource.get("id", "")
        sdoc = {
            "doc_id": f"{rtype}-{rid}-structured",
            "doc_type": "structured",
            "resourceType": rtype,
            "patientId": None,
            "patientName": None,
            "patientGender": None,
            "patientDOB": None,
            "patientAddress": None,
            "patientMaritalStatus": None,
            "patientMultipleBirth": None,
            "patientTelecom": None,
            "patientLanguage": None,
            "conditionId": None,
            "conditionCodeText": None,
            "conditionCategory": None,
            "conditionClinicalStatus": None,
            "conditionVerificationStatus": None,
            "conditionOnsetDateTime": None,
            "conditionRecordedDate": None,
            "conditionSeverity": None,
            "conditionNote": None,
            "observationId": None,
            "observationCodeText": None,
            "observationValue": None,
            "observationUnit": None,
            "observationInterpretation": None,
            "observationEffectiveDateTime": None,
            "observationIssued": None,
            "observationReferenceRange": None,
            "observationNote": None,
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
            "medRequestId": None,
            "medRequestMedicationDisplay": None,
            "medRequestAuthoredOn": None,
            "medRequestIntent": None,
            "medRequestStatus": None,
            "medRequestPriority": None,
            "medRequestDosageInstruction": None,
            "medRequestDispenseRequest": None,
            "medRequestNote": None,
            "procedureId": None,
            "procedureCodeText": None,
            "procedureStatus": None,
            "procedurePerformedDateTime": None,
            "procedureFollowUp": None,
            "procedureNote": None,
            "allergyId": None,
            "allergyClinicalStatus": None,
            "allergyVerificationStatus": None,
            "allergyType": None,
            "allergyCategory": None,
            "allergyCriticality": None,
            "allergyCodeText": None,
            "allergyOnsetDateTime": None,
            "allergyNote": None,
            "practitionerId": None,
            "practitionerName": None,
            "practitionerGender": None,
            "practitionerSpecialty": None,
            "practitionerAddress": None,
            "practitionerTelecom": None,
            "organizationId": None,
            "organizationName": None,
            "organizationType": None,
            "organizationAddress": None,
            "organizationTelecom": None,
        }

        unstructured_pieces = []
        div_text = resource.get("text", {}).get("div", "")
        if div_text.strip():
            unstructured_pieces.append(div_text)

        # Parse Patient
        if rtype == "Patient":
            sdoc["patientId"] = rid
            sdoc["patientGender"] = resource.get("gender")
            sdoc["patientDOB"] = resource.get("birthDate")
            if "name" in resource and len(resource["name"]) > 0:
                nm = resource["name"][0]
                f_ = nm.get("family", "")
                g_ = " ".join(nm.get("given", []))
                sdoc["patientName"] = (g_ + " " + f_).strip()
            if "address" in resource and len(resource["address"]) > 0:
                a0 = resource["address"][0]
                lines = a0.get("line", [])
                city = a0.get("city", "")
                state = a0.get("state", "")
                postal = a0.get("postalCode", "")
                sdoc["patientAddress"] = " ".join(lines + [city, state, postal]).strip()
            if "maritalStatus" in resource:
                ms = resource["maritalStatus"]
                sdoc["patientMaritalStatus"] = ms.get("text") or ms.get("coding", [{}])[
                    0
                ].get("code")
            if "multipleBirthInteger" in resource:
                sdoc["patientMultipleBirth"] = resource["multipleBirthInteger"]
            elif "multipleBirthBoolean" in resource:
                sdoc["patientMultipleBirth"] = (
                    1 if resource["multipleBirthBoolean"] else 0
                )
            if "telecom" in resource:
                tarr = []
                for t in resource["telecom"]:
                    use = t.get("use", "")
                    val = t.get("value", "")
                    tarr.append(use + ":" + val)
                sdoc["patientTelecom"] = " | ".join(tarr)
            if "communication" in resource and len(resource["communication"]) > 0:
                first_comm = resource["communication"][0]
                lang_obj = first_comm.get("language", {})
                sdoc["patientLanguage"] = lang_obj.get("text") or lang_obj.get(
                    "coding", [{}]
                )[0].get("code")

        # Parse Condition
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
            if "category" in resource and len(resource["category"]) > 0:
                cat_ = resource["category"][0]
                sdoc["conditionCategory"] = cat_.get("text") or cat_.get(
                    "coding", [{}]
                )[0].get("code")
            if "severity" in resource:
                sev = resource["severity"]
                sdoc["conditionSeverity"] = sev.get("text") or sev.get("coding", [{}])[
                    0
                ].get("code")
            c_field = resource.get("code", {})
            code_txt = c_field.get("text", "")
            if not code_txt and "coding" in c_field and len(c_field["coding"]) > 0:
                code_txt = c_field["coding"][0].get("display", "")
            sdoc["conditionCodeText"] = code_txt
            sdoc["conditionOnsetDateTime"] = resource.get("onsetDateTime")
            sdoc["conditionRecordedDate"] = resource.get("recordedDate")
            if "note" in resource:
                notes_ = []
                for note_item in resource["note"]:
                    n_txt = note_item.get("text", "").strip()
                    if n_txt:
                        notes_.append(n_txt)
                if notes_:
                    sdoc["conditionNote"] = " | ".join(notes_)
                    unstructured_pieces.extend(notes_)

        # Parse Observation
        elif rtype == "Observation":
            sdoc["observationId"] = rid
            c_obj = resource.get("code", {})
            obs_code_text = c_obj.get("text", "")
            if not obs_code_text and "coding" in c_obj and len(c_obj["coding"]) > 0:
                obs_code_text = c_obj["coding"][0].get("display", "")
            sdoc["observationCodeText"] = obs_code_text
            if "valueQuantity" in resource:
                val_ = resource["valueQuantity"].get("value", "")
                un_ = resource["valueQuantity"].get("unit", "")
                sdoc["observationValue"] = str(val_)
                sdoc["observationUnit"] = un_
            if "interpretation" in resource and len(resource["interpretation"]) > 0:
                first_i = resource["interpretation"][0]
                inter_txt = first_i.get("text") or first_i.get("coding", [{}])[0].get(
                    "code"
                )
                sdoc["observationInterpretation"] = inter_txt
            sdoc["observationEffectiveDateTime"] = resource.get("effectiveDateTime")
            sdoc["observationIssued"] = resource.get("issued")
            if "referenceRange" in resource and len(resource["referenceRange"]) > 0:
                rng_list = []
                for rng_item in resource["referenceRange"]:
                    l_ = rng_item.get("low", {}).get("value", "")
                    h_ = rng_item.get("high", {}).get("value", "")
                    rng_list.append(f"Low:{l_},High:{h_}")
                if rng_list:
                    sdoc["observationReferenceRange"] = " ; ".join(rng_list)
            if "note" in resource:
                obs_notes = []
                for nt in resource["note"]:
                    obs_txt = nt.get("text", "").strip()
                    if obs_txt:
                        obs_notes.append(obs_txt)
                if obs_notes:
                    sdoc["observationNote"] = " | ".join(obs_notes)
                    unstructured_pieces.extend(obs_notes)

        # Parse Encounter
        elif rtype == "Encounter":
            sdoc["encounterId"] = rid
            sdoc["encounterStatus"] = resource.get("status")
            sdoc["encounterClass"] = resource.get("class", {}).get("code")
            if "type" in resource and len(resource["type"]) > 0:
                first_enc_type = resource["type"][0]
                e_text = first_enc_type.get("text") or first_enc_type.get(
                    "coding", [{}]
                )[0].get("display", "")
                sdoc["encounterType"] = e_text
            if "reasonCode" in resource and len(resource["reasonCode"]) > 0:
                rcode0 = resource["reasonCode"][0]
                r_text = rcode0.get("text") or rcode0.get("coding", [{}])[0].get(
                    "display", ""
                )
                sdoc["encounterReasonCode"] = r_text
            period_ = resource.get("period", {})
            sdoc["encounterStart"] = period_.get("start")
            sdoc["encounterEnd"] = period_.get("end")
            if "location" in resource and len(resource["location"]) > 0:
                loc0 = resource["location"][0]
                loc_disp = loc0.get("location", {}).get("display", "")
                sdoc["encounterLocation"] = loc_disp
            if "serviceProvider" in resource:
                sp_ = resource["serviceProvider"]
                sdoc["encounterServiceProvider"] = sp_.get("reference")
            if "participant" in resource and len(resource["participant"]) > 0:
                parts = []
                for pp in resource["participant"]:
                    ind_ = pp.get("individual", {})
                    dsp_ = ind_.get("display", "")
                    parts.append(dsp_)
                if parts:
                    sdoc["encounterParticipant"] = " | ".join(parts)
            if "note" in resource:
                e_notes = []
                for note_ in resource["note"]:
                    e_txt = note_.get("text", "").strip()
                    if e_txt:
                        e_notes.append(e_txt)
                if e_notes:
                    sdoc["encounterNote"] = " | ".join(e_notes)
                    unstructured_pieces.extend(e_notes)

        # Parse MedicationRequest
        elif rtype == "MedicationRequest":
            sdoc["medRequestId"] = rid
            sdoc["medRequestIntent"] = resource.get("intent")
            sdoc["medRequestStatus"] = resource.get("status")
            sdoc["medRequestPriority"] = resource.get("priority")
            sdoc["medRequestAuthoredOn"] = resource.get("authoredOn")
            med_code = resource.get("medicationCodeableConcept", {})
            med_text = med_code.get("text", "")
            if (not med_text) and "coding" in med_code and len(med_code["coding"]) > 0:
                med_text = med_code["coding"][0].get("display", "")
            sdoc["medRequestMedicationDisplay"] = med_text
            if (
                "dosageInstruction" in resource
                and len(resource["dosageInstruction"]) > 0
            ):
                dosage_list = []
                for d_i in resource["dosageInstruction"]:
                    d_txt = d_i.get("text", "")
                    dosage_list.append(d_txt)
                if dosage_list:
                    sdoc["medRequestDosageInstruction"] = " | ".join(dosage_list)
            if "dispenseRequest" in resource:
                dr_ = resource["dispenseRequest"]
                sdoc["medRequestDispenseRequest"] = json.dumps(dr_)
            if "note" in resource:
                mr_notes = []
                for note_ in resource["note"]:
                    n_txt = note_.get("text", "").strip()
                    if n_txt:
                        mr_notes.append(n_txt)
                if mr_notes:
                    sdoc["medRequestNote"] = " | ".join(mr_notes)
                    unstructured_pieces.extend(mr_notes)

        # Parse Procedure
        elif rtype == "Procedure":
            sdoc["procedureId"] = rid
            sdoc["procedureStatus"] = resource.get("status")
            c_ = resource.get("code", {})
            c_text = c_.get("text") or c_.get("coding", [{}])[0].get("display")
            sdoc["procedureCodeText"] = c_text
            if "performedDateTime" in resource:
                sdoc["procedurePerformedDateTime"] = resource["performedDateTime"]
            if "followUp" in resource and len(resource["followUp"]) > 0:
                f_u_arr = []
                for f_item in resource["followUp"]:
                    fu_txt = f_item.get("text", "")
                    f_u_arr.append(fu_txt)
                if f_u_arr:
                    sdoc["procedureFollowUp"] = " | ".join(f_u_arr)
            if "note" in resource:
                proc_notes = []
                for pn in resource["note"]:
                    t_ = pn.get("text", "").strip()
                    if t_:
                        proc_notes.append(t_)
                if proc_notes:
                    sdoc["procedureNote"] = " | ".join(proc_notes)
                    unstructured_pieces.extend(proc_notes)

        # Parse AllergyIntolerance
        elif rtype == "AllergyIntolerance":
            sdoc["allergyId"] = rid
            acs = resource.get("clinicalStatus", {})
            sdoc["allergyClinicalStatus"] = acs.get("text")
            avs = resource.get("verificationStatus", {})
            sdoc["allergyVerificationStatus"] = avs.get("text")
            sdoc["allergyType"] = resource.get("type")
            if "category" in resource and len(resource["category"]) > 0:
                sdoc["allergyCategory"] = resource["category"][0]
            sdoc["allergyCriticality"] = resource.get("criticality")
            acode = resource.get("code", {})
            a_txt = acode.get("text", "")
            if not a_txt and "coding" in acode and len(acode["coding"]) > 0:
                a_txt = acode["coding"][0].get("display", "")
            sdoc["allergyCodeText"] = a_txt
            sdoc["allergyOnsetDateTime"] = resource.get("onsetDateTime")
            if "note" in resource:
                a_notes = []
                for an in resource["note"]:
                    a_ntext = an.get("text", "").strip()
                    if a_ntext:
                        a_notes.append(a_ntext)
                if a_notes:
                    sdoc["allergyNote"] = " | ".join(a_notes)
                    unstructured_pieces.extend(a_notes)

        # Parse Practitioner
        elif rtype == "Practitioner":
            sdoc["practitionerId"] = rid
            if "name" in resource and len(resource["name"]) > 0:
                nm_ = resource["name"][0]
                fam_ = nm_.get("family", "")
                gvn_ = " ".join(nm_.get("given", []))
                sdoc["practitionerName"] = (gvn_ + " " + fam_).strip()
            sdoc["practitionerGender"] = resource.get("gender")
            if "qualification" in resource and len(resource["qualification"]) > 0:
                q_ = resource["qualification"][0]
                sdoc["practitionerSpecialty"] = q_.get("code", {}).get("text")
            if "address" in resource and len(resource["address"]) > 0:
                addr_ = resource["address"][0]
                lines_ = addr_.get("line", [])
                city_ = addr_.get("city", "")
                state_ = addr_.get("state", "")
                postal_ = addr_.get("postalCode", "")
                sdoc["practitionerAddress"] = " ".join(
                    lines_ + [city_, state_, postal_]
                ).strip()
            if "telecom" in resource:
                tel_arr = []
                for t__ in resource["telecom"]:
                    use_ = t__.get("use", "")
                    val_ = t__.get("value", "")
                    tel_arr.append(use_ + ":" + val_)
                if tel_arr:
                    sdoc["practitionerTelecom"] = " | ".join(tel_arr)

        # Parse Organization
        elif rtype == "Organization":
            sdoc["organizationId"] = rid
            sdoc["organizationName"] = resource.get("name")
            if "type" in resource and len(resource["type"]) > 0:
                typ0 = resource["type"][0]
                sdoc["organizationType"] = typ0.get("text") or typ0.get("coding", [{}])[
                    0
                ].get("code")
            if "address" in resource and len(resource["address"]) > 0:
                org_adr = resource["address"][0]
                lines_ = org_adr.get("line", [])
                city_ = org_adr.get("city", "")
                state_ = org_adr.get("state", "")
                postal_ = org_adr.get("postalCode", "")
                sdoc["organizationAddress"] = " ".join(
                    lines_ + [city_, state_, postal_]
                ).strip()
            if "telecom" in resource:
                ot_arr = []
                for t__ in resource["telecom"]:
                    use_ = t__.get("use", "")
                    val_ = t__.get("value", "")
                    ot_arr.append(use_ + ":" + val_)
                if ot_arr:
                    sdoc["organizationTelecom"] = " | ".join(ot_arr)

        structured_docs.append(sdoc)
        if unstructured_pieces:
            combined_text = "\n".join(unstructured_pieces).strip()
            if combined_text:
                unstructured_data.append((rtype, combined_text))

    return structured_docs, unstructured_data


###############################################################################
# Endpoint: upload data => handle text (or) Json/FHIR
###############################################################################
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

            structured_docs, unstructured_data = parse_fhir_bundle(bundle_json)
            bulk_index_structured(user_id, doc_id, structured_docs)
            for resType, raw_text in unstructured_data:
                if raw_text.strip():
                    un_chunks = chunk_text(raw_text, CHUNK_SIZE)
                    un_embs = await embed_texts_in_batches(un_chunks)
                    if un_embs.shape[0] != len(un_chunks):
                        raise HTTPException(
                            status_code=500,
                            detail="Mismatch chunk vs embedding count for unstructured text in FHIR.",
                        )

                    sub_doc_id = f"{doc_id}_{resType}"
                    bulk_index_unstructured(
                        user_id, sub_doc_id, un_chunks, un_embs, resourceType=resType
                    )

        else:
            raise HTTPException(
                status_code=400, detail=f"Unsupported file type: {extension}"
            )

    return {"message": f"Uploaded and indexed {len(files)} file(s) for user={user_id}"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9001, reload=False)
