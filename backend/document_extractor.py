"""
Structured health data extraction from uploaded clinical documents.

When a new document is ingested, this module asks the LLM to read the
document text and return all vitals, lab results, medications, allergies,
and conditions as structured JSON.  The calling code saves each item to
the appropriate UserStore collection.

Design principles:
- Only newly uploaded documents are processed (not re-processed on reload).
- Content-based deduplication: vitals are skipped if the same type + value +
  date combination already exists in the patient record.
- Medications and allergies are already deduplicated by name in UserStore.
- All extracted data includes a note marking it as auto-extracted from the
  source filename so users can distinguish it from manually entered data.
- No hardcoded clinical values, units, or ranges anywhere in this module.
"""

import json
import os
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_EXTRACT_PROMPT = """\
You are a clinical data extraction assistant. Read the clinical document below and extract ALL structured health data present.

Return ONLY a valid JSON object with these four keys:

"vitals": array of objects, each with:
  - "type": standardized snake_case key from the list below
  - "value": the numeric or formatted value as a string (e.g. "120/80" for BP)
  - "unit": unit of measurement as a string
  - "recorded_on": date in YYYY-MM-DD format if present, else empty string
  - "notes": any clinical context (e.g. "fasting", "post-exercise"), else empty string

"medications": array of objects, each with:
  - "name": medication/drug name
  - "dose": dose with unit (e.g. "500 mg")
  - "schedule": frequency or route (e.g. "twice daily", "oral")
  - "reason": condition or indication if stated, else empty string
  - "started_on": start date YYYY-MM-DD if stated, else empty string
  - "notes": any additional context, else empty string

"allergies": array of objects, each with:
  - "name": allergen name
  - "reaction": reaction description
  - "severity": one of "mild" / "moderate" / "severe" / "unknown"
  - "allergy_type": one of "drug" / "food" / "environmental" / "other"
  - "confirmed": true or false

"conditions": array of objects describing diagnosed conditions, past medical history items, and active problems

For "conditions", prefer objects with keys "name", "status", "recorded_on", and "notes".
Use status "active", "past", "resolved", or "unknown".

Preferred vital/lab type keys (use these when they match; otherwise create a concise snake_case key from the document's measurement name):
blood_pressure, heart_rate, temperature, weight, height, bmi,
oxygen_saturation, respiratory_rate, blood_glucose, haemoglobin,
white_blood_cells, neutrophils, lymphocytes, monocytes, eosinophils,
basophils, platelets, haematocrit, mcv, mch, mchc, reticulocytes,
hba1c, egfr, creatinine, urea, sodium, potassium, chloride, bicarbonate,
calcium, phosphate, magnesium, albumin, total_protein, bilirubin_total,
bilirubin_direct, alt, ast, alp, ggt, ldh, crp, esr, ferritin, iron,
transferrin_saturation, b12, folate, cholesterol_total, cholesterol_ldl,
cholesterol_hdl, triglycerides, tsh, free_t4, free_t3, cortisol,
psa, peak_flow, inr, aptt, d_dimer, fibrinogen, troponin, bnp, nt_probnp

Rules:
- Extract ONLY values explicitly stated in the document — do not estimate or infer.
- Do NOT include patient names, addresses, ID numbers, or any identifiers.
- Do NOT include normal reference ranges as values — only the actual measured result.
- If a section is absent, use an empty array.
- Keep the JSON compact — no extra commentary outside the JSON object.

Document text:
{text}
"""


def _chunk_document_text(text: str, max_chars: int = 5000, overlap: int = 350) -> List[str]:
    cleaned = (text or "").strip()
    if not cleaned:
        return []

    chunks = []
    start = 0
    while start < len(cleaned):
        end = min(start + max_chars, len(cleaned))
        if end < len(cleaned):
            split_floor = start + int(max_chars * 0.65)
            newline_split = cleaned.rfind("\n", split_floor, end)
            space_split = cleaned.rfind(" ", split_floor, end)
            split_at = max(newline_split, space_split)
            if split_at > start:
                end = split_at

        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(cleaned):
            break
        start = max(end - overlap, start + 1)

    return chunks


def _normalize_extraction_payload(parsed: Dict) -> Dict[str, List]:
    conditions = []
    for condition in parsed.get("conditions") or []:
        if isinstance(condition, dict):
            name = str(condition.get("name") or "").strip()
            if name:
                conditions.append(condition)
        elif str(condition or "").strip():
            conditions.append(
                {
                    "name": str(condition).strip(),
                    "status": "unknown",
                    "recorded_on": "",
                    "notes": "",
                }
            )

    return {
        "vitals": [v for v in (parsed.get("vitals") or []) if isinstance(v, dict)],
        "medications": [m for m in (parsed.get("medications") or []) if isinstance(m, dict)],
        "allergies": [a for a in (parsed.get("allergies") or []) if isinstance(a, dict)],
        "conditions": conditions,
    }


def _dedupe_items(items: List[Dict], fields: List[str]) -> List[Dict]:
    deduped = []
    seen = set()
    for item in items:
        key = tuple(str(item.get(field, "")).strip().lower() for field in fields)
        if not any(key) or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _merge_extraction_payloads(payloads: List[Dict[str, List]]) -> Dict[str, List]:
    merged: Dict[str, List] = {"vitals": [], "medications": [], "allergies": [], "conditions": []}
    for payload in payloads:
        for key in merged:
            merged[key].extend(payload.get(key, []))

    merged["vitals"] = _dedupe_items(merged["vitals"], ["type", "value", "unit", "recorded_on"])
    merged["medications"] = _dedupe_items(merged["medications"], ["name", "dose", "schedule", "reason"])
    merged["allergies"] = _dedupe_items(merged["allergies"], ["name", "reaction"])
    merged["conditions"] = _dedupe_items(merged["conditions"], ["name", "status"])
    return merged


def extract_health_data_from_document(text: str, filename: str = "") -> Dict[str, List]:
    """
    Use the LLM to extract structured health data from a clinical document.

    Returns a dict:
      {"vitals": [...], "medications": [...], "allergies": [...], "conditions": [...]}
    Returns empty lists on any failure (silently logged).
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    empty: Dict[str, List] = {"vitals": [], "medications": [], "allergies": [], "conditions": []}

    if not api_key:
        empty["extraction_errors"] = ["OPENAI_API_KEY is not configured, so structured extraction could not run."]
        return empty

    if not text.strip():
        empty["extraction_errors"] = ["No readable text was found in the uploaded PDF."]
        return empty

    chunks = _chunk_document_text(text)
    if not chunks:
        empty["extraction_errors"] = ["No readable text was found in the uploaded PDF."]
        return empty

    payloads = []
    errors = []
    client = OpenAI(api_key=api_key)
    for index, chunk in enumerate(chunks, start=1):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": _EXTRACT_PROMPT.format(text=chunk),
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=2500,
            )
            raw = response.choices[0].message.content or "{}"
            payloads.append(_normalize_extraction_payload(json.loads(raw)))
        except Exception as exc:
            errors.append(f"Structured extraction failed for section {index}: {exc}")

    if not payloads:
        print(f"[document_extractor] Extraction failed for '{filename}': {' | '.join(errors)}")
        empty["extraction_errors"] = errors or ["Structured extraction failed."]
        return empty

    merged = _merge_extraction_payloads(payloads)
    if errors:
        merged["extraction_errors"] = errors
    return merged
