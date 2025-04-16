import random
import json
import torch
from torch.utils.data import Dataset
from transformers import (
    BertTokenizer,
    BertForTokenClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import f1_score
import numpy as np
import os
from dotenv import load_dotenv
import glob

# Load environment variables
load_dotenv()
SAMPLE_DATASET_PATH = os.getenv("SAMPLE_DATASET_PATH", "./sample_dataset")

# Define Expanded NER Labels
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
LABEL2ID = {label: idx for idx, label in enumerate(NER_LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}


# Load Multiple Synthea Data Files
def load_synthea_data(directory_path):
    all_data = []
    json_files = glob.glob(os.path.join(directory_path, "*.json"))
    if not json_files:
        print(f"No JSON files found in {directory_path}. Using fallback data.")
        return [{"entry": []}]  # Fallback empty data
    
    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
                all_data.append(data)
        except Exception as e:
            print(f"Error reading {json_file}: {e}")

    return all_data


# Extract Entities from Synthea JSON
def extract_entities_from_synthea(data_list):
    entities = {
        "names": [],
        "conditions": [],
        "medications": [],
        "procedures": [],
        "labtests": [],
        "anatomies": [],
        "obs_values": [],
        "icd10_codes": [],
        "cpt_codes": [],
        "loinc_codes": [],
        "dates": [],
        "genders": [],
        "phones": [],
        "emails": [],
        "addresses": [],
        "organizations": [],
        "severities": [],
        "allergies": [],
    }

    for data in data_list:
        for entry in data.get("entry", []):
            resource = entry.get("resource", {})
            resource_type = resource.get("resourceType")

            # Extract Names and Genders
            if resource_type == "Patient":
                for name_entry in resource.get("name", []):
                    given = name_entry.get("given", [])
                    family = name_entry.get("family", "")
                    for given_name in given:
                        full_name = f"{given_name} {family}".strip()
                        if full_name and full_name not in entities["names"]:
                            entities["names"].append(full_name)
                gender = resource.get("gender")
                if gender and gender not in entities["genders"]:
                    entities["genders"].append(gender)
                for telecom in resource.get("telecom", []):
                    if telecom.get("system") == "phone":
                        phone = telecom.get("value")
                        if phone and phone not in entities["phones"]:
                            entities["phones"].append(phone)
                    elif telecom.get("system") == "email":
                        email = telecom.get("value")
                        if email and email not in entities["emails"]:
                            entities["emails"].append(email)
                for address in resource.get("address", []):
                    lines = address.get("line", [])
                    city = address.get("city", "")
                    state = address.get("state", "")
                    postal = address.get("postalCode", "")
                    addr = ", ".join(
                        filter(None, lines + [city, state, postal])
                    ).strip()
                    if addr and addr not in entities["addresses"]:
                        entities["addresses"].append(addr)
                birth_date = resource.get("birthDate")
                if birth_date and birth_date not in entities["dates"]:
                    entities["dates"].append(birth_date)

            # Extract Conditions
            elif resource_type == "Condition":
                code = resource.get("code", {})
                for coding in code.get("coding", []):
                    display = coding.get("display", "").lower()
                    code_val = coding.get("code", "")
                    if display and display not in entities["conditions"]:
                        entities["conditions"].append(display)
                    if (
                        code_val
                        and code_val.startswith(("E", "I", "J", "M"))
                        and code_val not in entities["icd10_codes"]
                    ):
                        entities["icd10_codes"].append(code_val)
                onset = resource.get("onsetDateTime")
                if onset:
                    date_part = onset.split("T")[0]
                    if date_part and date_part not in entities["dates"]:
                        entities["dates"].append(date_part)

            # Extract Medications
            elif resource_type == "MedicationRequest":
                med = resource.get("medicationCodeableConcept", {})
                for coding in med.get("coding", []):
                    display = coding.get("display", "").lower()
                    if display and display not in entities["medications"]:
                        entities["medications"].append(display)
                date_written = resource.get("dateWritten")
                if date_written:
                    date_part = date_written.split("T")[0]
                    if date_part and date_part not in entities["dates"]:
                        entities["dates"].append(date_part)

            # Extract Procedures
            elif resource_type == "CarePlan":
                for activity in resource.get("activity", []):
                    detail = activity.get("detail", {})
                    code = detail.get("code", {})
                    for coding in code.get("coding", []):
                        display = coding.get("display", "").lower()
                        if display and display not in entities["procedures"]:
                            entities["procedures"].append(display)
                period = resource.get("period", {})
                start_date = period.get("start")
                if start_date:
                    date_part = start_date.split("T")[0]
                    if date_part and date_part not in entities["dates"]:
                        entities["dates"].append(date_part)

            # Extract Lab Tests and Observation Values
            elif resource_type == "Observation":
                code = resource.get("code", {})
                for coding in code.get("coding", []):
                    display = coding.get("display", "").lower()
                    loinc_code = coding.get("code", "")
                    if display and display not in entities["labtests"]:
                        entities["labtests"].append(display)
                    if loinc_code and loinc_code not in entities["loinc_codes"]:
                        entities["loinc_codes"].append(loinc_code)
                value_qty = resource.get("valueQuantity", {})
                if value_qty:
                    value = str(value_qty.get("value", ""))
                    unit = value_qty.get("unit", "")
                    obs_val = f"{value} {unit}".strip()
                    if obs_val and obs_val not in entities["obs_values"]:
                        entities["obs_values"].append(obs_val)
                components = resource.get("component", [])
                for comp in components:
                    value_qty = comp.get("valueQuantity", {})
                    if value_qty:
                        value = str(value_qty.get("value", ""))
                        unit = value_qty.get("unit", "")
                        obs_val = f"{value} {unit}".strip()
                        if obs_val and obs_val not in entities["obs_values"]:
                            entities["obs_values"].append(obs_val)
                effective = resource.get("effectiveDateTime")
                if effective:
                    date_part = effective.split("T")[0]
                    if date_part and date_part not in entities["dates"]:
                        entities["dates"].append(date_part)

            # Extract CPT Codes from Encounters or Procedures
            elif resource_type == "Encounter":
                type_list = resource.get("type", [])
                for type_entry in type_list:
                    for coding in type_entry.get("coding", []):
                        code = coding.get("code", "")
                        if (
                            code.isdigit()
                            and len(code) >= 5
                            and code not in entities["cpt_codes"]
                        ):
                            entities["cpt_codes"].append(code)
                period = resource.get("period", {})
                start_date = period.get("start")
                if start_date:
                    date_part = start_date.split("T")[0]
                    if date_part and date_part not in entities["dates"]:
                        entities["dates"].append(date_part)

            # Extract Organizations
            elif resource_type == "DiagnosticReport":
                for performer in resource.get("performer", []):
                    org = performer.get("display", "")
                    if org and org not in entities["organizations"]:
                        entities["organizations"].append(org)
                issued = resource.get("issued")
                if issued:
                    date_part = issued.split("T")[0]
                    if date_part and date_part not in entities["dates"]:
                        entities["dates"].append(date_part)

    # Fallbacks to ensure lists aren't empty
    if not entities["names"]:
        entities["names"] = ["Unknown Patient"]
    if not entities["conditions"]:
        entities["conditions"] = ["migraine"]
    if not entities["medications"]:
        entities["medications"] = ["metformin"]
    if not entities["procedures"]:
        entities["procedures"] = ["x-ray"]
    if not entities["labtests"]:
        entities["labtests"] = ["blood glucose"]
    if not entities["anatomies"]:
        entities["anatomies"] = ["heart"]
    if not entities["obs_values"]:
        entities["obs_values"] = ["120/80 mmHg"]
    if not entities["icd10_codes"]:
        entities["icd10_codes"] = ["E11.9"]
    if not entities["cpt_codes"]:
        entities["cpt_codes"] = ["99213"]
    if not entities["loinc_codes"]:
        entities["loinc_codes"] = ["4548-4"]
    if not entities["dates"]:
        entities["dates"] = ["2024-01-15"]
    if not entities["genders"]:
        entities["genders"] = ["male"]
    if not entities["phones"]:
        entities["phones"] = ["555-123-4567"]
    if not entities["emails"]:
        entities["emails"] = ["patient@example.com"]
    if not entities["addresses"]:
        entities["addresses"] = ["123 Main St"]
    if not entities["organizations"]:
        entities["organizations"] = ["Mercy Hospital"]
    if not entities["severities"]:
        entities["severities"] = ["mild"]
    if not entities["allergies"]:
        entities["allergies"] = ["penicillin"]

    return entities


# Generate NER Data Using Synthea Entities
def generate_ner_data(directory_path, n_samples=2000):
    synthea_data_list = load_synthea_data(directory_path)
    entities = extract_entities_from_synthea(synthea_data_list)
    doctors = generate_doctor_names(500)

    templates = [
        "Get details for patient {}.",
        "Find patients with {}.",
        "Patient {} had {} on {}.",
        "Consult with {} about {}.",
        "Prescribe {} for {}.",
        "Perform {} on {}.",
        "Order {} for {}.",
        "Check {} function in {}.",
        "Record {} as {}.",
        "Diagnose {} with code {}.",
        "Bill procedure code {} for {}.",
        "Lab test code {} ordered for {}.",
        "Schedule appointment on {} for {}.",
        "Patient is {}.",
        "Contact {} at {}.",
        "Email {} about appointment.",
        "Patient lives at {}.",
        "Refer {} to {}.",
        "{} condition is {}.",
        "Patient allergic to {}.",
    ]
    data = []
    for _ in range(n_samples):
        template = random.choice(templates)
        ner_entities = []
        if "Get details for patient {}" in template:
            name = random.choice(entities["names"])
            text = template.format(name)
            ner_entities.append(
                (text.index(name), text.index(name) + len(name), "PERSON")
            )
        elif "Find patients with {}" in template:
            cond = random.choice(entities["conditions"])
            text = template.format(cond)
            ner_entities.append(
                (text.index(cond), text.index(cond) + len(cond), "CONDITION")
            )
        elif "Patient {} had {} on {}" in template:
            name = random.choice(entities["names"])
            cond = random.choice(entities["conditions"])
            date = random.choice(entities["dates"])
            text = template.format(name, cond, date)
            ner_entities.extend(
                [
                    (text.index(name), text.index(name) + len(name), "PERSON"),
                    (text.index(cond), text.index(cond) + len(cond), "CONDITION"),
                    (text.index(date), text.index(date) + len(date), "DATE"),
                ]
            )
        elif "Consult with {} about {}" in template:
            doctor = random.choice(doctors)
            cond = random.choice(entities["conditions"])
            text = template.format(doctor, cond)
            ner_entities.extend(
                [
                    (text.index(doctor), text.index(doctor) + len(doctor), "DOCTOR"),
                    (text.index(cond), text.index(cond) + len(cond), "CONDITION"),
                ]
            )
        elif "Prescribe {} for {}" in template:
            med = random.choice(entities["medications"])
            name = random.choice(entities["names"])
            text = template.format(med, name)
            ner_entities.extend(
                [
                    (text.index(med), text.index(med) + len(med), "MEDICATION"),
                    (text.index(name), text.index(name) + len(name), "PERSON"),
                ]
            )
        elif "Perform {} on {}" in template:
            proc = random.choice(entities["procedures"])
            name = random.choice(entities["names"])
            text = template.format(proc, name)
            ner_entities.extend(
                [
                    (text.index(proc), text.index(proc) + len(proc), "PROCEDURE"),
                    (text.index(name), text.index(name) + len(name), "PERSON"),
                ]
            )
        elif "Order {} for {}" in template:
            test = random.choice(entities["labtests"])
            name = random.choice(entities["names"])
            text = template.format(test, name)
            ner_entities.extend(
                [
                    (text.index(test), text.index(test) + len(test), "LABTEST"),
                    (text.index(name), text.index(name) + len(name), "PERSON"),
                ]
            )
        elif "Check {} function in {}" in template:
            anatomy = random.choice(entities["anatomies"])
            name = random.choice(entities["names"])
            text = template.format(anatomy, name)
            ner_entities.extend(
                [
                    (
                        text.index(anatomy),
                        text.index(anatomy) + len(anatomy),
                        "ANATOMY",
                    ),
                    (text.index(name), text.index(name) + len(name), "PERSON"),
                ]
            )
        elif "Record {} as {}" in template:
            value = random.choice(entities["obs_values"])
            name = random.choice(entities["names"])
            text = template.format(name, value)
            ner_entities.extend(
                [
                    (text.index(name), text.index(name) + len(name), "PERSON"),
                    (text.index(value), text.index(value) + len(value), "OBS_VALUE"),
                ]
            )
        elif "Diagnose {} with code {}" in template:
            name = random.choice(entities["names"])
            code = random.choice(entities["icd10_codes"])
            text = template.format(name, code)
            ner_entities.extend(
                [
                    (text.index(name), text.index(name) + len(name), "PERSON"),
                    (text.index(code), text.index(code) + len(code), "ICD10_CODE"),
                ]
            )
        elif "Bill procedure code {} for {}" in template:
            code = random.choice(entities["cpt_codes"])
            name = random.choice(entities["names"])
            text = template.format(code, name)
            ner_entities.extend(
                [
                    (text.index(code), text.index(code) + len(code), "CPT_CODE"),
                    (text.index(name), text.index(name) + len(name), "PERSON"),
                ]
            )
        elif "Lab test code {} ordered for {}" in template:
            code = random.choice(entities["loinc_codes"])
            name = random.choice(entities["names"])
            text = template.format(code, name)
            ner_entities.extend(
                [
                    (text.index(code), text.index(code) + len(code), "LOINC_CODE"),
                    (text.index(name), text.index(name) + len(name), "PERSON"),
                ]
            )
        elif "Schedule appointment on {} for {}" in template:
            date = random.choice(entities["dates"])
            name = random.choice(entities["names"])
            text = template.format(date, name)
            ner_entities.extend(
                [
                    (text.index(date), text.index(date) + len(date), "DATE"),
                    (text.index(name), text.index(name) + len(name), "PERSON"),
                ]
            )
        elif "Patient is {}" in template:
            gender = random.choice(entities["genders"])
            text = template.format(gender)
            ner_entities.append(
                (text.index(gender), text.index(gender) + len(gender), "GENDER")
            )
        elif "Contact {} at {}" in template:
            name = random.choice(entities["names"])
            phone = random.choice(entities["phones"])
            text = template.format(name, phone)
            ner_entities.extend(
                [
                    (text.index(name), text.index(name) + len(name), "PERSON"),
                    (text.index(phone), text.index(phone) + len(phone), "PHONE"),
                ]
            )
        elif "Email {} about appointment" in template:
            email = random.choice(entities["emails"])
            text = template.format(email)
            ner_entities.append(
                (text.index(email), text.index(email) + len(email), "EMAIL")
            )
        elif "Patient lives at {}" in template:
            address = random.choice(entities["addresses"])
            text = template.format(address)
            ner_entities.append(
                (text.index(address), text.index(address) + len(address), "ADDRESS")
            )
        elif "Refer {} to {}" in template:
            name = random.choice(entities["names"])
            org = random.choice(entities["organizations"])
            text = template.format(name, org)
            ner_entities.extend(
                [
                    (text.index(name), text.index(name) + len(name), "PERSON"),
                    (text.index(org), text.index(org) + len(org), "ORGANIZATION"),
                ]
            )
        elif "{} condition is {}" in template:
            name = random.choice(entities["names"])
            severity = random.choice(entities["severities"])
            text = template.format(name, severity)
            ner_entities.extend(
                [
                    (text.index(name), text.index(name) + len(name), "PERSON"),
                    (
                        text.index(severity),
                        text.index(severity) + len(severity),
                        "SEVERITY",
                    ),
                ]
            )
        elif "Patient allergic to {}" in template:
            allergy = random.choice(entities["allergies"])
            text = template.format(allergy)
            ner_entities.append(
                (text.index(allergy), text.index(allergy) + len(allergy), "ALLERGY")
            )
        data.append({"text": text, "entities": ner_entities})
    return data


# Generate Doctor Names
def generate_doctor_names(n=500):
    prefixes = ["Dr.", "Doctor"]
    names = ["Smith", "Johnson", "Lee", "Patel", "Brown"]
    return [f"{random.choice(prefixes)} {random.choice(names)}" for _ in range(n)]


# Align Entities to Tokens
def align_entities_to_tokens(tokenizer, text, entities, max_length=128):
    tokens = tokenizer(
        text, truncation=True, padding=False, return_offsets_mapping=True
    )
    offset_mapping = tokens["offset_mapping"]
    labels = ["O"] * len(offset_mapping)
    for start, end, entity_type in entities:
        for i, (token_start, token_end) in enumerate(offset_mapping):
            if token_end == 0:
                continue
            if token_start >= start and token_start < end:
                labels[i] = f"B-{entity_type}"
            elif token_start >= start and token_end <= end:
                labels[i] = f"I-{entity_type}"
    labels = [LABEL2ID[label] for label in labels]
    if len(labels) < max_length:
        labels += [LABEL2ID["O"]] * (max_length - len(labels))
    else:
        labels = labels[:max_length]
    return labels, tokens


# Custom Dataset
class NERDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        entities = item["entities"]
        labels, tokens = align_entities_to_tokens(
            self.tokenizer, text, entities, self.max_length
        )
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# Compute Metrics
def compute_metrics(pred):
    labels = pred.label_ids.flatten()
    preds = pred.predictions.argmax(-1).flatten()
    valid_idx = labels != -100
    f1 = f1_score(labels[valid_idx], preds[valid_idx], average="weighted")
    return {"f1": f1}


# Custom Trainer for Single Checkpoint
class CustomNERTrainer(Trainer):
    def training_step(self, model, inputs):
        step_result = super().training_step(model, inputs)
        self.global_step = self.global_step + 1 if hasattr(self, "global_step") else 1
        if self.global_step % 25 == 0:  # Save every k=25 samples
            checkpoint_dir = "./ner_model/checkpoint"
            self.model.save_pretrained(checkpoint_dir)
            self.tokenizer.save_pretrained(checkpoint_dir)
            print(
                f"Overwrote checkpoint at {checkpoint_dir} at step {self.global_step}"
            )
        return step_result


# Main Training
def main():
    # Initialize
    model_name = "dmis-lab/biobert-large-cased-v1.1"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForTokenClassification.from_pretrained(
        model_name, num_labels=len(NER_LABELS)
    )

    # Set id2label for model config
    model.config.id2label = ID2LABEL
    model.config.label2id = LABEL2ID

    # Generate data
    data = generate_ner_data(SAMPLE_DATASET_PATH, 2000)
    train_data = data[:1600]
    eval_data = data[1600:]

    # Create datasets
    train_dataset = NERDataset(train_data, tokenizer)
    eval_dataset = NERDataset(eval_data, tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./ner_model",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_dir="./logs_ner",
        logging_steps=10,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    # Trainer
    trainer = CustomNERTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Train
    trainer.train()

    # Save final model
    model.save_pretrained("./ner_model/final")
    tokenizer.save_pretrained("./ner_model/final")

    # Test
    model.eval()
    test_queries = [
        "Get details for patient Julian140 Stamm395.",
        "Find patients with migraine taking metformin.",
        "Consult with Dr. Smith about diabetes on 2024-01-15.",
        "Order hemoglobin A1c for Emma567 Brown123.",
        "Patient allergic to penicillin.",
    ]
    for query in test_queries:
        inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)

        preds = outputs.logits.argmax(-1)[0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        print(f"\nQuery: {query}")
        for token, label_id in zip(tokens, preds):
            if token not in ["[CLS]", "[SEP]", "[PAD]"]:
                print(f"Token: {token}, Entity: {ID2LABEL[label_id]}")


if __name__ == "__main__":
    main()
