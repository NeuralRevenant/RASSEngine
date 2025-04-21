#!/usr/bin/env python
# train_ner.py  –  Build 10 000 synthetic EHR queries + fine‑tune BioBERT
# -----------------------------------------------------------------------
import os, glob, json, random, itertools, math
from collections import defaultdict
import numpy as np
from dotenv import load_dotenv
import torch
from torch.utils.data import Dataset
from transformers import (
    BertTokenizerFast,
    BertForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from sklearn.metrics import f1_score

load_dotenv()  # ───── read .env once and propagate everywhere
RAND = random.Random(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────── Env knobs ──────────────────────────────
SAMPLE_PATH = os.getenv("SAMPLE_DATASET_PATH", "./sample_dataset")
N_SAMPLES = int(os.getenv("N_SAMPLES", 10_000))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 4))
MODEL_NAME = os.getenv("MODEL_NAME", "dmis-lab/biobert-large-cased-v1.1")
LR = float(os.getenv("LEARNING_RATE", 3e-5))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
USE_FP16 = os.getenv("USE_FP16", "true").lower() == "true"
MAX_LEN = int(os.getenv("MAX_LEN", 128))
OUT_DIR = os.getenv("OUT_DIR", "./ner_model")

# ───────────────────────────── Label schema ─────────────────────────────
NER_LABELS = [
    "O",
    "B-PERSON",
    "I-PERSON",
    "B-CONDITION",
    "I-CONDITION",
    "B-DOCTOR",
    "I-DOCTOR",
    "B-MEDICATION",
    "I-MEDICATION",
    "B-PROCEDURE",
    "I-PROCEDURE",
    "B-LABTEST",
    "I-LABTEST",
    "B-ANATOMY",
    "I-ANATOMY",
    "B-OBS_VALUE",
    "I-OBS_VALUE",
    "B-ICD10_CODE",
    "I-ICD10_CODE",
    "B-CPT_CODE",
    "I-CPT_CODE",
    "B-LOINC_CODE",
    "I-LOINC_CODE",
    "B-DATE",
    "I-DATE",
    "B-GENDER",
    "I-GENDER",
    "B-PHONE",
    "I-PHONE",
    "B-EMAIL",
    "I-EMAIL",
    "B-ADDRESS",
    "I-ADDRESS",
    "B-ORGANIZATION",
    "I-ORGANIZATION",
    "B-SEVERITY",
    "I-SEVERITY",
    "B-ALLERGY",
    "I-ALLERGY",
]
LABEL2ID = {l: i for i, l in enumerate(NER_LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


# ────────────────────  Entity‑pool extraction from Synthea  ─────────────
def load_bundles(path: str):
    """Read every *.json bundle in `path` (Synthea output)."""
    files = glob.glob(os.path.join(path, "*.json"))
    return [json.load(open(f)) for f in files] or [{"entry": []}]


def pools_from_bundles(bundles):
    """
    Build a dict of lists:
        names, doctors, conditions, labtests, …
    Every value is unique & lower‑cased where appropriate.
    """
    p: dict[str, list[str]] = defaultdict(list)

    def _dedup_push(key: str, value: str):
        if value and value not in p[key]:
            p[key].append(value)

    for bundle in bundles:
        for entry in bundle.get("entry", []):
            res = entry.get("resource", {})
            rtype = res.get("resourceType")

            # ───── PATIENT ───────────────────────────────────────────
            if rtype == "Patient":
                for n in res.get("name", []):
                    full = (
                        " ".join(n.get("given", [])) + " " + n.get("family", "")
                    ).strip()
                    _dedup_push("names", full)
                _dedup_push("genders", res.get("gender", ""))
                for t in res.get("telecom", []):
                    if t.get("system") == "phone":
                        _dedup_push("phones", t["value"])
                    elif t.get("system") == "email":
                        _dedup_push("emails", t["value"])
                for adr in res.get("address", []):
                    txt = ", ".join(
                        adr.get("line", [])
                        + [adr.get(k, "") for k in ("city", "state", "postalCode")]
                    ).strip()
                    _dedup_push("addresses", txt)
                _dedup_push("dates", res.get("birthDate", ""))

            # ───── PRACTITIONER  → doctors pool ─────────────────────
            elif rtype == "Practitioner":
                for n in res.get("name", []):
                    full = (
                        "Dr. "
                        + " ".join(n.get("given", []))
                        + " "
                        + n.get("family", "")
                    ).strip()
                    _dedup_push("doctors", full)
                # also harvest direct telecom for practitioners
                for t in res.get("telecom", []):
                    if t.get("system") == "phone":
                        _dedup_push("phones", t["value"])
                    elif t.get("system") == "email":
                        _dedup_push("emails", t["value"])

            # ───── CONDITION ────────────────────────────────────────
            elif rtype == "Condition":
                for c in res.get("code", {}).get("coding", []):
                    _dedup_push("conditions", c.get("display", "").lower())
                    _dedup_push("icd10_codes", c.get("code", ""))
                if res.get("onsetDateTime"):
                    _dedup_push("dates", res["onsetDateTime"].split("T")[0])

            # ───── MEDICATION REQUEST ───────────────────────────────
            elif rtype == "MedicationRequest":
                for c in res.get("medicationCodeableConcept", {}).get("coding", []):
                    _dedup_push("medications", c.get("display", "").lower())
                if res.get("dateWritten"):
                    _dedup_push("dates", res["dateWritten"].split("T")[0])

            # ───── CARE PLAN  → procedures (planned) ────────────────
            elif rtype == "CarePlan":
                for act in res.get("activity", []):
                    for c in act.get("detail", {}).get("code", {}).get("coding", []):
                        _dedup_push("procedures", c.get("display", "").lower())
                if res.get("period", {}).get("start"):
                    _dedup_push("dates", res["period"]["start"].split("T")[0])

            # ───── OBSERVATION ──────────────────────────────────────
            elif rtype == "Observation":
                for c in res.get("code", {}).get("coding", []):
                    _dedup_push("labtests", c.get("display", "").lower())
                    _dedup_push("loinc_codes", c.get("code", ""))
                if res.get("valueQuantity"):
                    v = str(res["valueQuantity"].get("value", ""))
                    u = res["valueQuantity"].get("unit", "")
                    _dedup_push("obs_values", f"{v} {u}".strip())
                if res.get("effectiveDateTime"):
                    _dedup_push("dates", res["effectiveDateTime"].split("T")[0])

            # ───── ENCOUNTER  → CPT pool + doctors via participants ─
            elif rtype == "Encounter":
                for t in res.get("type", []):
                    for c in t.get("coding", []):
                        code = c.get("code", "")
                        if code.isdigit() and len(code) >= 5:
                            _dedup_push("cpt_codes", code)
                for part in res.get("participant", []):
                    disp = part.get("individual", {}).get("display", "")
                    if disp.startswith("Dr"):
                        _dedup_push("doctors", disp)
                if res.get("period", {}).get("start"):
                    _dedup_push("dates", res["period"]["start"].split("T")[0])

            # ───── DIAGNOSTIC REPORT  → organizations & doctors ─────
            elif rtype == "DiagnosticReport":
                for perf in res.get("performer", []):
                    disp = perf.get("display", "")
                    if disp.startswith("Dr"):
                        _dedup_push("doctors", disp)
                    else:
                        _dedup_push("organizations", disp)
                if res.get("issued"):
                    _dedup_push("dates", res["issued"].split("T")[0])

    # ───── fall‑backs so every key is non‑empty ────────────────────
    defaults = {
        "names": ["John Doe"],
        "doctors": [],  # leave empty – we don't fabricate now
        "conditions": ["diabetes"],
        "medications": ["metformin"],
        "procedures": ["x‑ray"],
        "labtests": ["hemoglobin A1c"],
        "anatomies": ["heart"],
        "obs_values": ["120/80 mmHg"],
        "icd10_codes": ["E11.9"],
        "cpt_codes": ["99213"],
        "loinc_codes": ["4548‑4"],
        "dates": ["2024‑01‑15"],
        "genders": ["male"],
        "phones": ["555‑123‑4567"],
        "emails": ["patient@example.com"],
        "addresses": ["123 Main St"],
        "organizations": ["Mercy Hospital"],
        "severities": ["moderate"],
        "allergies": ["penicillin"],
    }
    for k, dflt in defaults.items():
        if not p[k]:
            p[k] = dflt.copy()

    # If still no doctors, derive from existing patient names (prefix “Dr.”)
    if not p["doctors"]:
        p["doctors"] = [f"Dr. {n.split()[0]}" for n in p["names"][:10]]

    return p


# ───────────────────────────── 400 unique templates ─────────────────────
# 200 single‑slot (brief) + 200 multi‑slot (complex).  All **hand‑written**.
TEMPLATES = [
    # ───────────── simple one‑placeholder sentences (first 200) ─────────────
    "Get details for patient {}.",
    "Show clinical summary for {}.",
    "Retrieve chart of {}.",
    "Display all visits for {}.",
    "Provide demographic data for {}.",
    "Find allergies listed for {}.",
    "List medications for {}.",
    "Give me active conditions for {}.",
    "Fetch lab history of {}.",
    "Show procedures performed on {}.",
    "Get emergency contacts for {}.",
    "Display latest note about {}.",
    "Summarize encounters of {}.",
    "Show immunizations for {}.",
    "Retrieve address of {}.",
    "Locate phone number of {}.",
    "Email for {}?",
    "What is the gender of {}?",
    "When was {} born?",
    "Show most recent appointment of {}.",
    "List upcoming visits for {}.",
    "Find insurance info for {}.",
    "Show care plan of {}.",
    "Display vital signs for {}.",
    "Provide BMI value for {}.",
    "Give problem list for {}.",
    "Show social history for {}.",
    "Retrieve smoking status of {}.",
    "Give marital status of {}.",
    "List active orders for {}.",
    "Show pending labs for {}.",
    "Display imaging results for {}.",
    "Provide genetic tests for {}.",
    "Find growth chart for {}.",
    "List adverse events of {}.",
    "Show consent forms of {}.",
    "Retrieve discharge summary for {}.",
    "Display referral letters for {}.",
    "Show medication adherence for {}.",
    "Provide pain scores for {}.",
    "Give depression screening of {}.",
    "Show fall risk for {}.",
    "Display advance directives of {}.",
    "Get functional status of {}.",
    "Retrieve hearing test for {}.",
    "Show vision screening of {}.",
    "Provide travel history for {}.",
    "List vaccination refusals for {}.",
    "Show blood type of {}.",
    # 50 so far …
    "Find patients with {}.",
    "Search records for {}.",
    "Locate cases of {}.",
    "Show prevalence of {}.",
    "List complications of {}.",
    "Provide guidelines for {}.",
    "Explain symptoms of {}.",
    "Display staging for {}.",
    "Give risk factors of {}.",
    "Retrieve treatments for {}.",
    "Show prognosis of {}.",
    "List comorbidities of {}.",
    "Display genetic links to {}.",
    "Provide ICD‑10 code for {}.",
    "Show SNOMED mapping for {}.",
    "Retrieve CPT mapping for {}.",
    "List typical labs for {}.",
    "Give imaging modalities for {}.",
    "Display severity scale for {}.",
    "Provide common drugs for {}.",
    # 70 …
    "Show info for drug {}.",
    "List indications of {}.",
    "Provide dosage of {}.",
    "Display side effects of {}.",
    "Show contraindications of {}.",
    "Give monitoring needs for {}.",
    "Retrieve pregnancy category of {}.",
    "Explain mechanism of {}.",
    "Provide generic names of {}.",
    "Show brand names of {}.",
    "List cost data for {}.",
    "Display storage rules for {}.",
    "Provide taper schedule for {}.",
    "Show renal adjustment for {}.",
    "List hepatic adjustment for {}.",
    "Provide black‑box warning of {}.",
    "Give pharmacokinetics of {}.",
    "Display onset time of {}.",
    "Provide half‑life of {}.",
    "Show drug interactions with {}.",
    # 90 …
    "What does procedure {} involve?",
    "Show CPT code {} details.",
    "Provide prep for {}.",
    "Display after‑care for {}.",
    "List equipment for {}.",
    "Show average duration of {}.",
    "Provide anesthesia needs for {}.",
    "Give recovery time of {}.",
    "Display risk profile for {}.",
    "Show documentation rules for {}.",
    # 100 …
    "Show lab test {} reference range.",
    "Provide specimen for {}.",
    "Display fasting needs for {}.",
    "Give turnaround time for {}.",
    "Explain meaning of {}.",
    "List interfering factors for {}.",
    "Show cost of {}.",
    "Provide CPT for {}.",
    "Display LOINC for {}.",
    "Give critical values for {}.",
    # 110 …
    "Locate ICD‑10 code {}.",
    "Explain ICD‑10 {} title.",
    "Show inclusion terms for {}.",
    "Provide exclusions for {}.",
    "Display coding tips for {}.",
    "Give billable status of {}.",
    "List chapter for {}.",
    "Show code family of {}.",
    "Provide severity axis for {}.",
    "Give laterality info for {}.",
    # 120 …
    "Locate CPT code {}.",
    "Show description of CPT {}.",
    "Provide RVU of {}.",
    "Display global period for {}.",
    "Give status indicator for {}.",
    "Show Medicare fee for {}.",
    "Provide HCPCS crosswalk for {}.",
    "Display procedure type of {}.",
    "Give surgical package for {}.",
    "List modifiers for {}.",
    # 130 …
    "Locate LOINC code {}.",
    "Provide short name for {}.",
    "Show long name of {}.",
    "Display component of {}.",
    "Give property of {}.",
    "Show system for {}.",
    "Provide scale of {}.",
    "Display class of {}.",
    "Give version of {}.",
    "List related names for {}.",
    # 140 …
    "Show data on date {}.",
    "Provide admissions on {}.",
    "Display discharges on {}.",
    "Give lab volume on {}.",
    "List procedures on {}.",
    "Show prescriptions on {}.",
    "Provide immunizations on {}.",
    "Display ED visits on {}.",
    "Give births on {}.",
    "List deaths on {}.",
    # 150 …
    "Show all {} patients.",
    "Provide {}‑specific dosage.",
    "Display {} risk.",
    "Give {} care plan.",
    "List {} support groups.",
    "Show {} diet.",
    "Provide {} exercise advice.",
    "Display {} educational material.",
    "Give {} rehab protocol.",
    "List {} follow‑up schedule.",
    # 160 …
    "Show phone number {} belongs to.",
    "Locate email {}.",
    "Find address {}.",
    "Which patient uses phone {}?",
    "Which record lists email {}?",
    "Who lives at {}?",
    "Show organization {} details.",
    "Display department of {}.",
    "Provide location of {}.",
    "List contact for {}.",
    # 170 …
    "Patient allergic to {}.",
    "Show reactions to {}.",
    "Give severity options for {}.",
    "Provide avoidance tips for {}.",
    "List cross‑reactants to {}.",
    "Display incident count for {}.",
    "Show desensitization for {}.",
    "Provide alternative drugs for {}.",
    "Display exposure routes for {}.",
    "Give first‑aid for {}.",
    # 180 …
    "Show doctor {} schedule.",
    "Provide specialty of {}.",
    "Display contact for {}.",
    "Give clinic for {}.",
    "List patients under {}.",
    "Show referrals from {}.",
    "Provide npi for {}.",
    "Display ratings of {}.",
    "Give availability of {}.",
    "List languages spoken by {}.",
    # 190 …
    "Show organization {} hierarchy.",
    "Provide accreditation of {}.",
    "Display bed count for {}.",
    "Give departments of {}.",
    "List services of {}.",
    "Show payer contracts for {}.",
    "Provide quality scores for {}.",
    "Display location map of {}.",
    "Give visiting hours of {}.",
    "List emergency numbers for {}.",
    # ─────────── 200 complex multi‑placeholder sentences (200‑400) ───────────
    "Compare {} results for {} before and after {}.",
    "List patients named {} who tested positive for {}.",
    "How many {} procedures did {} undergo between {} and {}?",
    "Retrieve encounters where {} was treated with {} on {}.",
    "Show {} readings and LOINC {} for {}.",
    "Which cases of {} have CPT code {} recorded by {}?",
    "Give me {} results for {} since {} at {}.",
    "Has {} experienced {} severity {} in the past year?",
    "Explain why {} prescribed {} to {} on {}.",
    "Record {} as {} for patient {} at {}.",
    "Reschedule {}’s {} appointment from {} to {}.",
    "Contact {} at {} if {} exceeds {}.",
    "Email {} regarding allergen {} exposure for {}.",
    "Order {} for {} and document under LOINC {}.",
    "Flag {} patients taking {} and suffering {}.",
    "Summarize {}'s history of {} including meds {}.",
    "Add procedure code {} to {}’s encounter dated {}.",
    "What ICD‑10 code {} occurrences were logged for {}?",
    "Graph {} and {} trends for patient {}.",
    "Route discharge summary of {} to {} via email {}.",
    "Provide {} observations with value {} for {} between {} and {}.",
    "List {} patients with {} allergy taking {}.",
    "Compare efficacy of {} versus {} in {} cases.",
    "Show {} outcome differences for {} and {} over {} months.",
    "Count encounters for {} at {} between {} and {}.",
    "Display {} dosage changes for {} during {} visit.",
    "Fetch CPT {} and ICD‑10 {} pairs for {}'s procedure.",
    "Show lab {} values above {} for patient {}.",
    "Provide medication {} start date {} for {}.",
    "Explain link between condition {} and procedure {} for {}.",
    "List {} surgeries by {} done on {} at {}.",
    "Find {} admissions for {} with diagnosis {}.",
    "Display {} readings, reference {}, and date {}.",
    "Compare {} severity {} with allergy {} in {}.",
    "Retrieve phone {} and email {} for {}.",
    "Document {} of {} at facility {} on {}.",
    "Schedule {} follow‑up on {} with doctor {} for {}.",
    "List {} refills of {} since {} for {}.",
    "Show correlations between {} and {} in {} population.",
    "Provide last {} note signed by {} on {}.",
    "Get {} units of {} ordered for {} on {}.",
    "Find patients {} years old with {} taking {}.",
    "Count {} cases of {} under organization {}.",
    "Compare cost of procedure {} CPT {} at {}.",
    "List {} with ICD‑10 {} and CPT {} done on {}.",
    "Retrieve address {} and phone {} for doctor {}.",
    "Summarize {} results (LOINC {}) for {} across {} records.",
    "Display {} lab flagged critical for {} on {}.",
    "Explain procedure {} scheduled on {} for {} at {}.",
    "Provide vaccination {} given to {} on {}.",
    "Give imaging {} result for {} performed {}.",
    "Show diet order {} for {} issued {} by {}.",
    "List readmission of {} with diagnosis {} within {} days.",
    "Graph weight {} and BMI {} over time for {}.",
    "Provide allergy {} reaction noted by {} on {}.",
    "Explain refusal of medication {} by {} on {}.",
    "Display prior authorization for {} CPT {} for {}.",
    "Contact {} via phone {} and email {} about {}.",
    "Schedule physical exam {} for {} at {} with {}.",
    "Retrieve encounter {} codes {} and {} for {}.",
    "Provide smoking status {} recorded for {} on {}.",
    "Find caregiver {} for patient {} at {}.",
    "Show follow‑up plan {} created by {} for {}.",
    "Route imaging {} report to {} and CC {}.",
    "List high blood pressure {} (> {}) cases for {}.",
    "Compare cholesterol {} readings pre‑{} and post‑{} for {}.",
    "Document pain score {} for {} during {}.",
    "Provide consent form {} signed by {} on {}.",
    "Fetch prenatal visit {} for patient {} on {}.",
    "Give newborn screening {} result {} for {}.",
    "Summarize emergency visit of {} on {} coded {}.",
    "Explain discrepancy between {} and {} labs for {}.",
    "List comorbidity {} in {} with medication {}.",
    "Provide therapy {} sessions attended by {} this year.",
    "Show rehab progress {} and goals {} for {}.",
    "Display dialysis {} records for {} at {}.",
    "Find mental health note {} for {} authored {}.",
    "List home meds {} and doses {} for {}.",
    "Provide blood type {} and antibody {} for {}.",
    "Show glucose {} trends > {} for {}.",
    "Compare ejection fraction {} before {} and after {} in {}.",
    "List Deeplinking {} to {} for patient {} on {}.",  # made‑up but unique
    "Document telehealth {} encounter for {} on {} with {}.",
    "Provide CPR event {} logged by {} on {}.",
    "Show wound care {} orders for {} dated {}.",
    "List implant {} model {} inserted in {}.",
    "Compare sedation score {} to target {} in {}.",
    "Retrieve code status {} and date {} for {}.",
    "Provide referral {} from {} to {} for {}.",
    "Display transport {} request for {} on {}.",
    "Summarize nutrition {} macro {} for {}.",
    "Graph temperature {} vs heart rate {} for {}.",
    "Explain override of alert {} by {} on {}.",
    "List insurance {} policy {} covering {}.",
    "Show payer auth {} approved on {} for {}.",
    "Provide coverage {} termination {} for {}.",
    "Display caregiver note {} length {} for {}.",
    "List social risk {} flagged for {} on {}.",
    "Provide language {} preference for {}.",
    "Show portal access {} granted to {} on {}.",
    "Document lab error {} resolved by {} on {}.",
    "List inventory {} consumed during {} for {}.",
    "Provide capillary refill {} recorded for {} at {}.",
    "Display ankle‑brachial {} index for {}.",
    "Explain high INR {} measurement for {} on {}.",
    "Show anticoagulation {} plan dose {} for {}.",
    "Provide fall score {} calculation for {} on {}.",
    "List home oxygen flow {} for {} dated {}.",
    "Display spirometry {} FEV1/FVC {} for {}.",
    "Provide wheelchair {} dimensions {} for {}.",
    "Show ostomy care {} product {} for {}.",
    "Explain pregnancy test {} result {} for {}.",
    "Show IV line {} gauge {} inserted for {}.",
    "Provide blood culture {} organism {} for {}.",
    "List hearing aid {} battery {} for {}.",
    "Document trach care {} frequency {} for {}.",
    "Give pain medication {} PRN {} to {}.",
    "Show sedation vacation {} start {} for {}.",
    "Provide breaks in feeding {} of {} for {}.",
    "List ICU mobility {} level {} for {}.",
    "Document pressure ulcer {} stage {} for {}.",
    "Provide neurosurgery {} consent {} for {}.",
    "Show pacemaker {} interrogation {} for {}.",
    "List skin graft {} donor site {} for {}.",
    "Explain chemo regimen {} cycle {} for {}.",
    "Provide radiation {} dose {} for {}.",
    "Show transplant {} match score {} for {}.",
    "List genetic panel {} variant {} for {}.",
    "Provide endocrine {} TSH {} for {}.",
    "Show antibody {} titer {} for {}.",
    "Provide cerebrospinal fluid {} protein {} for {}.",
    "Show ophthalmology {} acuity {} for {}.",
    "Provide dental {} caries {} in {}.",
    "List podiatry {} ulcer {} in {}.",
    "Show dermatology {} lesion {} for {}.",
    "Provide psychiatry {} score {} for {}.",
    "List speech therapy {} goal {} for {}.",
    "Provide occupational therapy {} ADL {} for {}.",
    "Show physical therapy {} gait {} for {}.",
    "List school note {} excuse {} for {}.",
    "Provide travel clinic {} vaccine {} for {}.",
    "Show hyperbaric {} session {} for {}.",
    "Provide sleep study {} apnea‑index {} for {}.",
    "List fertility {} AMH {} for {}.",
    "Provide bariatric {} weight‑loss {} for {}.",
    "Show nephrology {} GFR {} for {}.",
    "Provide cardiology {} echo‑EF {} for {}.",
    "Show pulmonology {} FVC {} for {}.",
    "Provide gastroenterology {} colonoscopy {} for {}.",
    "Show hematology {} Hb {} for {}.",
    "Provide oncology {} tumor‑marker {} for {}.",
    "Show rheumatology {} DAS‑score {} for {}.",
    "Provide neurology {} NIHSS {} for {}.",
    "Show urology {} PSA {} for {}.",
    "Provide endocrinology {} HbA1c {} for {}.",
    "Show immunology {} IgE {} for {}.",
    "Provide infectious‑disease {} PCR {} for {}.",
    "Show gerontology {} frailty {} for {}.",
    "Provide pediatrics {} growth‑percentile {} for {}.",
    "Show obstetrics {} fundal‑height {} for {}.",
    "Provide gynecology {} pap‑result {} for {}.",
    "Show orthopedics {} ROM {} for {}.",
    "Provide psychiatry {} suicide‑risk {} for {}.",
    "Show nutrition {} kcal‑goal {} for {}.",
    "Provide nephrology {} creatinine {} for {}.",
    "Show hepatology {} ALT {} for {}.",
    "Provide dermatology {} PASI {} for {}.",
    "Show ENT {} audiogram {} for {}.",
    "Provide oncology {} stage {} for {}.",
    "Show endocrinology {} insulin‑dose {} for {}.",
    "Provide cardiology {} stent type {} for {}.",
    "Show pulmonology {} oxygen‑sat {} for {}.",
    "Provide gastroenterology {} polyp {} for {}.",
    "Show radiology {} impression {} for {}.",
    "Provide pathology {} margin {} for {}.",
    "Show ophthalmology {} pressure {} for {}.",
    "Provide neurology {} seizure‑count {} for {}.",
    "Show urology {} stone‑size {} for {}.",
    "Provide rheumatology {} ANA {} for {}.",
    "Show immunology {} allergen‑panel {} for {}.",
    "Provide infectious‑disease {} viral‑load {} for {}.",
    "Show gerontology {} MMSE {} for {}.",
    "Provide pediatrics {} vaccine‑schedule {} for {}.",
    "Show obstetrics {} dilation {} for {}.",
    "Provide gynecology {} fibroid‑size {} for {}.",
    "Show orthopedics {} hardware {} for {}.",
    "Provide psychiatry {} PHQ‑9 {} for {}.",
    "Show nutrition {} sodium‑intake {} for {}.",
    "Provide nephrology {} fluid‑balance {} for {}.",
    "Show hepatology {} fibrosis‑score {} for {}.",
    "Provide dermatology {} biopsy‑result {} for {}.",
    "Show ENT {} tympanogram {} for {}.",
    "Provide oncology {} chemo‑cycle {} for {}.",
    "Show endocrinology {} C‑peptide {} for {}.",
    "Provide cardiology {} troponin {} for {}.",
    "Show pulmonology {} DLCO {} for {}.",
    "Provide gastroenterology {} stool‑test {} for {}.",
    "Show radiology {} dose‑length {} for {}.",
    "Provide pathology {} immunostain {} for {}.",
    "Show ophthalmology {} field‑loss {} for {}.",
    "Provide neurology {} EMG {} for {}.",
    "Show urology {} uroflow {} for {}.",
    "Provide rheumatology {} CRP {} for {}.",
    "Show immunology {} complement {} for {}.",
    "Provide infectious‑disease {} genotype {} for {}.",
    "Show gerontology {} fall‑risk {} for {}.",
    "Provide pediatrics {} bilirubin {} for {}.",
    "Show obstetrics {} NST {} for {}.",
    "Provide gynecology {} HPV‑type {} for {}.",
    "Show orthopedics {} bone‑density {} for {}.",
    "Provide psychiatry {} GAD‑7 {} for {}.",
    "Show nutrition {} carb‑ratio {} for {}.",
    "Provide nephrology {} dialysis‑dose {} for {}.",
    "Show hepatology {} MELD {} for {}.",
    "Provide dermatology {} dermoscopy‑score {} for {}.",
    "Show ENT {} speech‑discrimination {} for {}.",
    "Provide oncology {} neutrophil‑count {} for {}.",
    "Show endocrinology {} cortisol {} for {}.",
    "Provide cardiology {} BNP {} for {}.",
    "Show pulmonology {} peak‑flow {} for {}.",
    "Provide gastroenterology {} esophageal‑pH {} for {}.",
    "Show radiology {} Hounsfield {} for {}.",
    "Provide pathology {} Ki‑67 {} for {}.",
]  # ← end of 400 templates (each unique)

# ------------------------------------------------------------------------
# mapping from placeholder entity to pool key
ENTITY2POOL = {
    "PERSON": "names",
    "CONDITION": "conditions",
    "MEDICATION": "medications",
    "PROCEDURE": "procedures",
    "LABTEST": "labtests",
    "ANATOMY": "anatomies",
    "OBS_VALUE": "obs_values",
    "ICD10_CODE": "icd10_codes",
    "CPT_CODE": "cpt_codes",
    "LOINC_CODE": "loinc_codes",
    "DATE": "dates",
    "GENDER": "genders",
    "PHONE": "phones",
    "EMAIL": "emails",
    "ADDRESS": "addresses",
    "ORGANIZATION": "organizations",
    "SEVERITY": "severities",
    "ALLERGY": "allergies",
    "DOCTOR": "doctors",
}

COMPLEX_LABELS = [
    (
        "Compare {} results for {} before and after {}.",
        ["LABTEST", "PERSON", "DATE"],
    ),
    (
        "List patients named {} who tested positive for {}.",
        ["PERSON", "LABTEST"],
    ),
    (
        "How many {} procedures did {} undergo between {} and {}?",
        ["PROCEDURE", "PERSON", "DATE", "DATE"],
    ),
    (
        "Retrieve encounters where {} was treated with {} on {}.",
        ["CONDITION", "MEDICATION", "DATE"],
    ),
    (
        "Show {} readings and LOINC {} for {}.",
        ["LABTEST", "LOINC_CODE", "PERSON"],
    ),
    (
        "Which cases of {} have CPT code {} recorded by {}?",
        ["CONDITION", "CPT_CODE", "DOCTOR"],
    ),
    (
        "Give me {} results for {} since {} at {}.",
        ["LABTEST", "PERSON", "DATE", "ORGANIZATION"],
    ),
    (
        "Has {} experienced {} severity {} in the past year?",
        ["PERSON", "CONDITION", "SEVERITY"],
    ),
    (
        "Explain why {} prescribed {} to {} on {}.",
        ["DOCTOR", "MEDICATION", "PERSON", "DATE"],
    ),
    (
        "Record {} as {} for patient {} at {}.",
        ["OBS_VALUE", "LABTEST", "PERSON", "DATE"],
    ),
    (
        "Reschedule {}’s {} appointment from {} to {}.",
        ["PERSON", "PROCEDURE", "DATE", "DATE"],
    ),
    (
        "Contact {} at {} if {} exceeds {}.",
        ["DOCTOR", "PHONE", "OBS_VALUE", "OBS_VALUE"],
    ),
    (
        "Email {} regarding allergen {} exposure for {}.",
        ["EMAIL", "ALLERGY", "PERSON"],
    ),
    (
        "Order {} for {} and document under LOINC {}.",
        ["LABTEST", "PERSON", "LOINC_CODE"],
    ),
    (
        "Flag {} patients taking {} and suffering {}.",
        ["GENDER", "MEDICATION", "CONDITION"],
    ),
    (
        "Summarize {}'s history of {} including meds {}.",
        ["PERSON", "CONDITION", "MEDICATION"],
    ),
    (
        "Add procedure code {} to {}’s encounter dated {}.",
        ["CPT_CODE", "PERSON", "DATE"],
    ),
    (
        "What ICD‑10 code {} occurrences were logged for {}?",
        ["ICD10_CODE", "PERSON"],
    ),
    (
        "Graph {} and {} trends for patient {}.",
        ["LABTEST", "LABTEST", "PERSON"],
    ),
    (
        "Route discharge summary of {} to {} via email {}.",
        ["PERSON", "ORGANIZATION", "EMAIL"],
    ),
]

COMPLEX_MAP = {tpl: labels for tpl, labels in COMPLEX_LABELS}

GENERIC_COMPLEX_LABELS = ["LABTEST", "CONDITION", "PERSON"]  # fallback


def fill(template: str, pools: dict[str, list[str]], RNG=random) -> dict:
    """
    Replace each '{}' in `template` with a random entity value from `pools`
    and return spans + labels for NER training.
    """
    # ───── determine slot labels ─────
    if template in COMPLEX_MAP:
        slot_labels = COMPLEX_MAP[template]
    elif template.count("{}") > 1:  # unseen complex sentence
        slot_labels = GENERIC_COMPLEX_LABELS
    else:  # simple one‑placeholder templates
        idx = TEMPLATES.index(template)  # TEMPLATES is your master list
        if idx < 50:
            slot_labels = ["PERSON"]
        elif idx < 70:
            slot_labels = ["CONDITION"]
        elif idx < 90:
            slot_labels = ["MEDICATION"]
        elif idx < 100:
            slot_labels = ["PROCEDURE"]
        elif idx < 110:
            slot_labels = ["LABTEST"]
        elif idx < 120:
            slot_labels = ["ICD10_CODE"]
        elif idx < 130:
            slot_labels = ["CPT_CODE"]
        elif idx < 140:
            slot_labels = ["LOINC_CODE"]
        elif idx < 150:
            slot_labels = ["DATE"]
        elif idx < 160:
            slot_labels = ["CONDITION"]
        elif idx < 170:
            low = template.lower()
            if "phone" in low:
                slot_labels = ["PHONE"]
            elif "address" in low:
                slot_labels = ["ADDRESS"]
            elif "email" in low:
                slot_labels = ["EMAIL"]
            else:
                slot_labels = ["ORGANIZATION"]
        elif idx < 180:
            slot_labels = ["ALLERGY"]
        elif idx < 190:
            slot_labels = ["DOCTOR"]
        else:
            slot_labels = ["ORGANIZATION"]

    # ───── choose concrete values ─────
    values = [
        RNG.choice(pools["doctors"] if lab == "DOCTOR" else pools[ENTITY2POOL[lab]])
        for lab in slot_labels
    ]

    # ───── build labelled text + spans ─────
    text = template.format(*values)
    spans = []
    for val, lab in zip(values, slot_labels):
        start = text.index(val)
        spans.append((start, start + len(val), lab))

    return {"text": text, "entities": spans}


# ------------------------------------------------------------------------
# Dataset class
class QueryDataset(Dataset):
    def __init__(self, data, tok):
        self.data = data
        self.tok = tok

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        enc = self.tok(
            item["text"],
            return_offsets_mapping=True,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
        )
        labels = ["O"] * len(enc["input_ids"])
        for s, e, lab in item["entities"]:
            for i, (ts, te) in enumerate(enc["offset_mapping"]):
                if ts >= s and te <= e and te != 0:
                    labels[i] = "B-" + lab if ts == s else "I-" + lab
        labels = [LABEL2ID[l] for l in labels]
        return {
            "input_ids": torch.tensor(enc["input_ids"]),
            "attention_mask": torch.tensor(enc["attention_mask"]),
            "labels": torch.tensor(labels),
        }


def metric(p):
    pred = p.predictions.argmax(-1)
    lab = p.label_ids
    mask = lab != -100
    return {"f1": f1_score(lab[mask], pred[mask], average="weighted")}


# ------------------------------------------------------------------------
def main():
    print("▶ Loading Synthea pool …")
    pools = pools_from_bundles(load_bundles(SAMPLE_PATH))

    print("▶ Generating", N_SAMPLES, "synthetic queries …")
    data = [fill(RAND.choice(TEMPLATES), pools) for _ in range(N_SAMPLES)]
    RAND.shuffle(data)
    split = int(0.8 * N_SAMPLES)
    train_data, eval_data = data[:split], data[split:]

    tok = BertTokenizerFast.from_pretrained(MODEL_NAME)
    train_ds, eval_ds = QueryDataset(train_data, tok), QueryDataset(eval_data, tok)

    model = BertForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(NER_LABELS)
    ).to(DEVICE)

    args = TrainingArguments(
        output_dir=OUT_DIR,
        evaluation_strategy="epoch",
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        fp16=USE_FP16 and DEVICE.type == "cuda",
        save_total_limit=2,
        logging_steps=100,
        metric_for_best_model="f1",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tok,
        data_collator=DataCollatorForTokenClassification(tok),
        compute_metrics=metric,
    )

    print("▶ Fine‑tuning …")
    trainer.train()
    model.save_pretrained(os.path.join(OUT_DIR, "final"))
    tok.save_pretrained(os.path.join(OUT_DIR, "final"))
    print("✔ Model stored at", os.path.join(OUT_DIR, "final"))


if __name__ == "__main__":
    main()
