"""
Clinical trial search and matching.

Search strategy:
  1. LLM reads the full patient context (memory, triage notes, symptoms,
     medications) and returns individual medical condition/symptom terms
     and individual drug names as separate lists.
  2. A separate ClinicalTrials.gov API call is made for EACH condition term
     (query.cond) and each drug name (query.intr), plus country (query.locn).
  3. Results are merged by NCT ID. Each trial records which of the patient's
     condition searches returned it.
  4. Trials are scored client-side against the full patient profile.
  5. Ranked by multi-condition coverage first, then overall match score.

No raw patient text is ever sent to the API.
No clinical content is hardcoded.
"""

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

import requests
from dotenv import load_dotenv
from openai import OpenAI

from backend.user_store import compute_current_age

load_dotenv()

API_BASE_URL = "https://clinicaltrials.gov/api/v2"
RECRUITING_STATUS = "RECRUITING"

STOPWORDS = {
    "about", "after", "again", "against", "also", "and", "are", "because",
    "been", "before", "being", "between", "but", "can", "clinical",
    "condition", "could", "does", "from", "have", "health", "into", "list",
    "medicine", "medication", "not", "patient", "record", "saved", "should",
    "study", "symptom", "that", "the", "their", "there", "this", "trial",
    "used", "using", "with", "your",
}


@dataclass
class TrialSearchProfile:
    conditions: List[str]           # raw conditions for client-side scoring
    symptoms: List[str]             # raw symptoms for client-side scoring
    medications: List[str]          # raw medication names for client-side scoring
    age: Optional[int]
    biological_sex: str
    raw_context: str                # full patient context → LLM extracts search terms from this


def _clean(value: object) -> str:
    return str(value or "").strip()


def _clean_markup(value: object) -> str:
    text = _clean(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _unique(values: Iterable[str]) -> List[str]:
    seen: set = set()
    result = []
    for value in values:
        cleaned = _clean(value)
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def _tokens(text: str) -> set:
    return {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9+-]{2,}", text.lower())
        if token not in STOPWORDS
    }


def _age_from_api(value: str) -> Optional[int]:
    if not value or value.upper() == "N/A":
        return None
    match = re.search(r"(\d+(?:\.\d+)?)\s*(year|month|week|day)", value.lower())
    if not match:
        return None
    amount = float(match.group(1))
    unit = match.group(2)
    if unit == "year":
        return int(amount)
    if unit == "month":
        return int(amount / 12)
    return 0


def _module(study: Dict, name: str) -> Dict:
    return study.get("protocolSection", {}).get(name, {}) or {}


def _study_text(study: Dict) -> str:
    identification = _module(study, "identificationModule")
    conditions = _module(study, "conditionsModule")
    description = _module(study, "descriptionModule")
    eligibility = _module(study, "eligibilityModule")
    arms = _module(study, "armsInterventionsModule")
    interventions = arms.get("interventions", []) or []
    intervention_text = " ".join(
        " ".join(_clean(p) for p in (
            i.get("name"), i.get("type"), i.get("description"),
        ))
        for i in interventions
    )
    return " ".join(
        _clean_markup(part)
        for part in (
            identification.get("briefTitle"),
            identification.get("officialTitle"),
            " ".join(conditions.get("conditions", []) or []),
            " ".join(conditions.get("keywords", []) or []),
            description.get("briefSummary"),
            eligibility.get("eligibilityCriteria"),
            intervention_text,
        )
        if part
    )


def _extract_locations(study: Dict) -> List[Dict]:
    contacts_locations = _module(study, "contactsLocationsModule")
    extracted = []
    for loc in contacts_locations.get("locations", []) or []:
        extracted.append({
            "facility": _clean(loc.get("facility")),
            "city": _clean(loc.get("city")),
            "state": _clean(loc.get("state")),
            "country": _clean(loc.get("country")),
            "status": _clean(loc.get("status")),
            "contacts": [
                {
                    "name": _clean(c.get("name")),
                    "role": _clean(c.get("role")),
                    "phone": _clean(c.get("phone")),
                    "email": _clean(c.get("email")),
                }
                for c in (loc.get("contacts") or [])
            ],
        })
    return extracted


def _extract_contacts(study: Dict, locations: List[Dict]) -> List[Dict]:
    contacts_locations = _module(study, "contactsLocationsModule")
    contacts = []
    for c in contacts_locations.get("centralContacts", []) or []:
        contacts.append({
            "name": _clean(c.get("name")),
            "role": _clean(c.get("role")),
            "phone": _clean(c.get("phone")),
            "email": _clean(c.get("email")),
            "source": "Central contact",
        })
    for loc in locations:
        for c in loc.get("contacts", []):
            enriched = dict(c)
            enriched["source"] = location_label(loc)
            contacts.append(enriched)
    return [c for c in contacts if c.get("name") or c.get("phone") or c.get("email")]


def _extract_officials(study: Dict) -> List[Dict]:
    contacts_locations = _module(study, "contactsLocationsModule")
    officials = []
    for o in contacts_locations.get("overallOfficials", []) or []:
        officials.append({
            "name": _clean(o.get("name")),
            "role": _clean(o.get("role")),
            "affiliation": _clean(o.get("affiliation")),
        })
    return [o for o in officials if o.get("name") or o.get("affiliation")]


def location_label(location: Dict) -> str:
    pieces = [
        location.get("facility", ""),
        location.get("city", ""),
        location.get("state", ""),
        location.get("country", ""),
    ]
    return ", ".join(_unique(pieces)) or "Location not listed"


# ---------------------------------------------------------------------------
# LLM extraction — individual terms from the full patient context
# ---------------------------------------------------------------------------

def _llm_extract_search_terms(raw_context: str) -> Dict[str, List[str]]:
    """
    Ask the LLM to read the patient's full health context and extract:
      - "conditions": individual medical condition / symptom terms (each 1–5 words)
      - "medications": individual drug / medication names

    Each term will be used in a SEPARATE ClinicalTrials.gov API call so that
    trials can be matched against multiple patient conditions independently.

    Returns {"conditions": [...], "medications": [...]} or empty lists on failure.
    """
    if not raw_context.strip():
        return {"conditions": [], "medications": []}

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return {"conditions": [], "medications": []}

    try:
        client = OpenAI(api_key=api_key)
        prompt = (
            "You are a clinical research coordinator preparing a multi-term ClinicalTrials.gov search.\n\n"
            "Read the patient data below. Extract every distinct medical condition, symptom, "
            "sign, or diagnosis the patient has — as a JSON list of short individual terms.\n"
            "Also extract any drug or medication names as a separate JSON list.\n\n"
            "Return ONLY this JSON object (no explanation):\n"
            "{\n"
            '  "conditions": ["term1", "term2", ...],\n'
            '  "medications": ["drug1", "drug2", ...]\n'
            "}\n\n"
            "Rules for conditions/symptoms:\n"
            "- Each entry must be a real medical condition, symptom, sign, or diagnosis (1–5 words).\n"
            "- List them individually — do NOT concatenate multiple conditions into one string.\n"
            "- Include the most specific terms you can extract "
            "(e.g. \"right iliac fossa pain\" not just \"pain\").\n"
            "- Do NOT include: administrative labels (general triage, routine, self-care, GP), "
            "normal findings (normal FBC, normal results), patient names, or demographics.\n"
            "- Max 8 condition terms.\n\n"
            "Rules for medications:\n"
            "- List drug/medication names only — no doses, no routes, no frequencies.\n"
            "- Max 5 medication names.\n\n"
            f"Patient data:\n{raw_context[:3000]}"
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=250,
        )
        raw = response.choices[0].message.content or "{}"
        parsed = json.loads(raw)

        conditions = [
            _clean(t) for t in (parsed.get("conditions") or [])
            if _clean(t) and len(_clean(t)) <= 80
        ][:8]
        medications = [
            _clean(t) for t in (parsed.get("medications") or [])
            if _clean(t) and len(_clean(t)) <= 60
        ][:5]

        return {"conditions": conditions, "medications": medications}

    except Exception as exc:
        print(f"[clinical_trials] LLM extraction failed: {exc}")
        return {"conditions": [], "medications": []}


# ---------------------------------------------------------------------------
# Profile builder
# ---------------------------------------------------------------------------

def build_trial_search_profile(
    profile: Dict,
    memory: Dict,
    symptom_logs: List[Dict],
    medications: List[Dict],
    allergies: List[Dict],
    vitals: List[Dict],
    triage_summaries: List[Dict],
) -> TrialSearchProfile:
    """
    Gathers all patient data into:
      - Raw lists (conditions, symptoms, medications) used for client-side scoring.
      - raw_context: a compact, structured text the LLM will read to extract
        individual search terms when the user triggers a search.
    """
    conditions = _unique(
        [m.get("reason", "") for m in medications if m.get("reason")]
        + [s.get("pathway_label", "") for s in triage_summaries]
        + [s.get("decision_summary", "") for s in triage_summaries[:3]]
    )
    symptoms = _unique(e.get("symptom", "") for e in symptom_logs)
    medication_names = _unique(m.get("name", "") for m in medications)

    # Build the rich context the LLM will read
    parts: List[str] = []

    memory_summary = _clean(memory.get("summary", ""))
    if memory_summary:
        parts.append(f"Patient longitudinal history:\n{memory_summary[:1800]}")

    for s in triage_summaries[:5]:
        decision = _clean(s.get("decision_summary", ""))
        escalation = s.get("escalation_triggers") or []
        pathway = _clean(s.get("pathway_label", ""))
        if decision:
            parts.append(f"Triage note: {decision}")
        if escalation:
            parts.append("Escalation triggers: " + "; ".join(str(e) for e in escalation[:5]))
        if pathway and pathway.lower() not in ("general triage", "general"):
            parts.append(f"Clinical pathway: {pathway}")

    for e in symptom_logs[:15]:
        sym = _clean(e.get("symptom", ""))
        if sym:
            parts.append(f"Logged symptom: {sym}")

    for m in medications[:10]:
        name = _clean(m.get("name", ""))
        reason = _clean(m.get("reason", ""))
        if name:
            parts.append(f"Medication: {name}" + (f" (prescribed for {reason})" if reason else ""))

    raw_context = "\n".join(parts)

    return TrialSearchProfile(
        conditions=conditions,
        symptoms=symptoms,
        medications=medication_names,
        age=compute_current_age(profile.get("date_of_birth", "")),
        biological_sex=_clean(profile.get("biological_sex")),
        raw_context=raw_context,
    )


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _best_location(locations: List[Dict], location_query: str) -> Dict:
    if not locations:
        return {}
    query_tokens = _tokens(location_query)
    if not query_tokens:
        return locations[0]

    def score(loc: Dict) -> int:
        return len(query_tokens & _tokens(location_label(loc).lower()))

    return max(locations, key=score)


def _location_score(locations: List[Dict], location_query: str) -> int:
    if not location_query:
        return 10
    if not locations:
        return 0
    query_tokens = _tokens(location_query)
    if not query_tokens:
        return 10
    best_overlap = 0
    best_contains = False
    lowered = location_query.lower()
    for loc in locations:
        label = location_label(loc).lower()
        best_overlap = max(best_overlap, len(query_tokens & _tokens(label)))
        if lowered in label:
            best_contains = True
    if best_contains:
        return 25
    if best_overlap:
        return min(24, 12 + best_overlap * 4)
    return 5


def _demographic_score(study: Dict, profile: TrialSearchProfile) -> tuple:
    eligibility = _module(study, "eligibilityModule")
    flags = []
    score = 0
    sex = _clean(eligibility.get("sex")).upper()
    profile_sex = profile.biological_sex.upper()
    if not sex or sex == "ALL" or not profile_sex or profile_sex == "OTHER":
        score += 7
    elif sex.startswith(profile_sex):
        score += 8
    else:
        flags.append(f"Trial sex criterion is {sex.title()}; profile has {profile.biological_sex}.")
    min_age = _age_from_api(_clean(eligibility.get("minimumAge")))
    max_age = _age_from_api(_clean(eligibility.get("maximumAge")))
    if profile.age is None:
        score += 5
    else:
        age_ok = True
        if min_age is not None and profile.age < min_age:
            age_ok = False
            flags.append(f"Profile age appears below minimum age ({eligibility.get('minimumAge')}).")
        if max_age is not None and profile.age > max_age:
            age_ok = False
            flags.append(f"Profile age appears above maximum age ({eligibility.get('maximumAge')}).")
        score += 7 if age_ok else 0
    return min(score, 15), flags


def _health_score(study: Dict, profile: TrialSearchProfile) -> tuple:
    """
    Align the study's full text against the patient's raw conditions, symptoms,
    and medications entirely client-side.
    """
    text = _study_text(study).lower()
    study_tokens = _tokens(text)
    patient_tokens = _tokens(
        " ".join(profile.conditions + profile.symptoms + profile.medications)
    )
    matched = []
    phrase_score = 0
    for cond in profile.conditions[:8]:
        ct = _tokens(cond)
        if ct and ct <= study_tokens:
            phrase_score += 10
            matched.append(cond)
    for sym in profile.symptoms[:8]:
        st = _tokens(sym)
        if st and st <= study_tokens:
            phrase_score += 4
            matched.append(sym)
    for med in profile.medications[:6]:
        mt = _tokens(med)
        if mt and mt <= study_tokens:
            phrase_score += 5
            matched.append(med)
    overlap_score = min(20, len(patient_tokens & study_tokens) * 2)
    return min(55, phrase_score + overlap_score), _unique(matched)


def score_trial(
    study: Dict,
    profile: TrialSearchProfile,
    location_query: str,
    found_for: List[str],
    total_search_terms: int,
) -> Dict:
    identification = _module(study, "identificationModule")
    status = _module(study, "statusModule")
    conditions_mod = _module(study, "conditionsModule")
    description = _module(study, "descriptionModule")
    design = _module(study, "designModule")
    eligibility = _module(study, "eligibilityModule")
    arms = _module(study, "armsInterventionsModule")

    locations = _extract_locations(study)
    contacts = _extract_contacts(study, locations)
    officials = _extract_officials(study)
    best_location = _best_location(locations, location_query)

    health_score, matched_terms = _health_score(study, profile)
    demographic_score, demographic_flags = _demographic_score(study, profile)
    location_score = _location_score(locations, location_query)
    contact_score = 5 if contacts else 2 if officials else 0
    status_score = 5 if status.get("overallStatus") == RECRUITING_STATUS else 0

    # Coverage bonus: trials found across multiple condition searches rank higher
    coverage_count = len(found_for)
    coverage_score = min(20, coverage_count * 5) if total_search_terms > 0 else 0

    score = min(100, 20 + health_score + demographic_score + location_score
                + contact_score + status_score + coverage_score)

    interventions = _unique([
        _clean(i.get("name"))
        for i in (arms.get("interventions", []) or [])
        if _clean(i.get("name"))
    ])[:6]
    nct_id = identification.get("nctId", "")

    return {
        "nct_id": nct_id,
        "title": _clean(identification.get("briefTitle") or identification.get("officialTitle")),
        "official_title": _clean(identification.get("officialTitle")),
        "status": _clean(status.get("overallStatus")),
        "conditions": conditions_mod.get("conditions", []) or [],
        "interventions": interventions,
        "phase": ", ".join(design.get("phases", []) or []),
        "study_type": _clean(design.get("studyType")),
        "summary": _clean_markup(description.get("briefSummary")),
        "eligibility": _clean_markup(eligibility.get("eligibilityCriteria")),
        "sex": _clean(eligibility.get("sex")),
        "minimum_age": _clean(eligibility.get("minimumAge")),
        "maximum_age": _clean(eligibility.get("maximumAge")),
        "locations": locations,
        "best_location": best_location,
        "contacts": contacts,
        "officials": officials,
        "matched_terms": matched_terms,
        "demographic_flags": demographic_flags,
        "found_for_conditions": found_for,
        "condition_coverage": coverage_count,
        "total_conditions_searched": total_search_terms,
        "match_score": round(score),
        "health_score": health_score,
        "coverage_score": coverage_score,
        "location_score": location_score,
        "demographic_score": demographic_score,
        "contact_score": contact_score,
        "url": f"https://clinicaltrials.gov/study/{nct_id}" if nct_id else "",
    }


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _request_studies(params: Dict[str, str], timeout: int = 20) -> List[Dict]:
    response = requests.get(f"{API_BASE_URL}/studies", params=params, timeout=timeout)
    response.raise_for_status()
    return response.json().get("studies", []) or []


def _search_one_term(
    term: str,
    param_key: str,
    location_query: str,
    page_size: int = 25,
) -> List[Dict]:
    """Run one condition or intervention search and return raw studies."""
    params: Dict[str, str] = {
        "format": "json",
        "pageSize": str(page_size),
        "countTotal": "true",
        "filter.overallStatus": RECRUITING_STATUS,
        param_key: term,
    }
    if location_query.strip():
        params["query.locn"] = location_query.strip()

    studies = _request_studies(params)

    # Fallback: if few results with location, broaden globally
    if len(studies) < 5 and location_query.strip():
        global_params = {k: v for k, v in params.items() if k != "query.locn"}
        global_params["pageSize"] = str(page_size)
        known = {_module(s, "identificationModule").get("nctId") for s in studies}
        for s in _request_studies(global_params):
            if _module(s, "identificationModule").get("nctId") not in known:
                studies.append(s)

    return studies


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def find_matching_trials(
    profile: TrialSearchProfile,
    location_query: str,
    max_results: int = 10,
) -> Dict:
    """
    Three-phase clinical trial matching:

    Phase 1 — Extract: LLM reads the full patient context and returns
      individual medical condition terms and drug names as separate lists.

    Phase 2 — Search: each condition term gets its own query.cond call;
      each drug name gets its own query.intr call. Results are merged by
      NCT ID. Every trial records which patient conditions found it.

    Phase 3 — Score & rank: trials scored client-side against the full
      patient profile. Ranked by multi-condition coverage first, then
      overall match score.
    """
    # Phase 1: LLM extracts individual terms
    extracted = _llm_extract_search_terms(profile.raw_context)
    condition_terms = extracted.get("conditions", [])
    medication_terms = extracted.get("medications", [])

    if not condition_terms and not medication_terms:
        return {
            "searched_at": datetime.now(timezone.utc).isoformat(),
            "trials": [],
            "condition_terms": [],
            "medication_terms": [],
            "location": location_query,
            "error": (
                "No specific medical conditions or medications could be found in your saved data. "
                "Chat with Dr. Charlotte about your health concerns — the assistant builds a "
                "condition record the trial finder uses to search."
            ),
        }

    # Phase 2: one API call per term, merge by NCT ID
    # merged[nct_id] = {"study": <raw study dict>, "found_for": [<term>, ...]}
    merged: Dict[str, Dict] = {}

    total_terms = len(condition_terms) + len(medication_terms)

    for term in condition_terms:
        try:
            for study in _search_one_term(term, "query.cond", location_query):
                nct_id = _module(study, "identificationModule").get("nctId", "")
                if not nct_id:
                    continue
                if nct_id not in merged:
                    merged[nct_id] = {"study": study, "found_for": []}
                if term not in merged[nct_id]["found_for"]:
                    merged[nct_id]["found_for"].append(term)
            time.sleep(0.15)  # courtesy rate limit
        except Exception as exc:
            print(f"[clinical_trials] Search failed for cond '{term}': {exc}")

    for drug in medication_terms:
        try:
            for study in _search_one_term(drug, "query.intr", location_query):
                nct_id = _module(study, "identificationModule").get("nctId", "")
                if not nct_id:
                    continue
                if nct_id not in merged:
                    merged[nct_id] = {"study": study, "found_for": []}
                label = f"medication: {drug}"
                if label not in merged[nct_id]["found_for"]:
                    merged[nct_id]["found_for"].append(label)
            time.sleep(0.15)
        except Exception as exc:
            print(f"[clinical_trials] Search failed for intr '{drug}': {exc}")

    # Phase 3: score each unique trial against the full patient profile
    ranked = [
        score_trial(
            data["study"],
            profile,
            location_query,
            found_for=data["found_for"],
            total_search_terms=total_terms,
        )
        for data in merged.values()
    ]

    # Primary sort: how many patient condition searches returned this trial
    # Secondary: overall match score, then location proximity
    ranked.sort(
        key=lambda t: (
            t["condition_coverage"],
            t["match_score"],
            t["location_score"],
            t["health_score"],
        ),
        reverse=True,
    )

    return {
        "searched_at": datetime.now(timezone.utc).isoformat(),
        "trials": ranked[:max_results],
        "condition_terms": condition_terms,
        "medication_terms": medication_terms,
        "location": location_query,
        "error": "",
    }
