from __future__ import annotations

from typing import Dict, List


PRODUCT_NAME = "Dr. Charlotte"
PRODUCT_TAGLINE = "Private, structured health support for individuals, caregivers, and care teams."
PRODUCT_SUBTITLE = "Secure account access, role-aware terms, and continuity across visits."
FOUNDER_NAME = "Francis Osei"
SUPPORT_EMAIL = "oseifrancis633@gmail.com"
TERMS_VERSION = "2026-04-05"

ROLE_OPTIONS = [
    "Patient / Individual",
    "Caregiver",
    "Doctor / Physician",
    "Nurse",
    "Midwife",
    "Physiotherapist",
    "Other Clinician",
]

CLINICIAN_ROLES = {
    "doctor / physician",
    "nurse",
    "midwife",
    "physiotherapist",
    "other clinician",
}

DEFAULT_CARE_CONTEXT_BY_ROLE = {
    "patient / individual": "Personal health guidance",
    "caregiver": "Caregiver support",
    "doctor / physician": "Clinical decision support",
    "nurse": "Clinical decision support",
    "midwife": "Maternity decision support",
    "physiotherapist": "MSK and rehabilitation support",
    "other clinician": "Clinical decision support",
}

ROLE_TERMS: Dict[str, Dict[str, List[str] | str]] = {
    "Patient / Individual": {
        "title": "Patient and Individual Terms",
        "summary": "This account type is designed for personal health education and question support.",
        "bullets": [
            "The service provides informational guidance only and does not diagnose conditions, prescribe medicines, or replace a licensed clinician.",
            "Do not rely on the service for emergencies. If symptoms are severe, worsening, or urgent, contact emergency services or a qualified clinician immediately.",
            "Only upload or enter information that belongs to you, or that you are permitted to use.",
        ],
        "acknowledgement": "I understand this patient account is for personal health information and education, not diagnosis or emergency care.",
    },
    "Caregiver": {
        "title": "Caregiver Terms",
        "summary": "This account type supports people helping a family member, friend, or person in their care.",
        "bullets": [
            "Use the service to support communication, preparation, and understanding, not as a substitute for professional clinical advice.",
            "You are responsible for ensuring you are permitted to share or review another person's information before using it here.",
            "If the person you support has urgent symptoms or immediate safety concerns, contact emergency or clinical services without delay.",
        ],
        "acknowledgement": "I understand this caregiver account supports care coordination and education and does not replace professional advice or emergency services.",
    },
    "Doctor / Physician": {
        "title": "Doctor and Physician Terms",
        "summary": "This account type is intended for licensed physicians using Dr. Charlotte as a professional support tool.",
        "bullets": [
            "Outputs are supportive and must not be used as the sole basis for diagnosis, treatment, prescribing, referral, or discharge decisions.",
            "You remain responsible for clinical judgement, patient-specific assessment, and compliance with local policy, supervision, and documentation requirements.",
            "Only enter patient information when you have a lawful basis and organisational authorisation to access and process it.",
        ],
        "acknowledgement": "I will use this physician account as decision support only and will verify outputs using my own clinical judgement and local policy.",
    },
    "Nurse": {
        "title": "Nursing Terms",
        "summary": "This account type is intended for registered nurses using Dr. Charlotte for evidence review and practice support.",
        "bullets": [
            "Outputs support nursing interpretation, communication, and escalation planning, but do not replace clinical assessment or local nursing policy.",
            "You remain responsible for confirming medicines, escalation pathways, and patient-specific actions with the relevant clinical team and approved workflows.",
            "Only use patient information that you are authorised to access, discuss, and document in your professional role.",
        ],
        "acknowledgement": "I will use this nursing account within my scope of practice and will verify recommendations against clinical assessment and local protocol.",
    },
    "Midwife": {
        "title": "Midwifery Terms",
        "summary": "This account type is intended for maternity professionals using Dr. Charlotte for educational and decision-support purposes.",
        "bullets": [
            "Outputs do not replace maternity assessment, safeguarding duties, escalation to obstetric teams, or emergency referral pathways.",
            "Pregnancy, postpartum, and newborn care require heightened caution, and all recommendations must be reviewed within the full clinical context.",
            "You are responsible for ensuring any patient information entered is used lawfully and in line with maternity governance requirements.",
        ],
        "acknowledgement": "I will use this midwifery account as a support tool only and will apply maternity-specific safety judgement and escalation standards.",
    },
    "Physiotherapist": {
        "title": "Physiotherapy Terms",
        "summary": "This account type is intended for physiotherapists and rehabilitation professionals using Dr. Charlotte in practice support.",
        "bullets": [
            "Outputs support rehabilitation planning and MSK education, but do not replace full assessment, contraindication screening, or referral judgement.",
            "You remain responsible for checking red flags, neurovascular issues, exercise tolerance, and suitability for any intervention before acting.",
            "Only use patient information that you are authorised to review and record within your clinical role.",
        ],
        "acknowledgement": "I will use this physiotherapy account as clinical support only and will verify all exercise and management decisions against my assessment.",
    },
    "Other Clinician": {
        "title": "Clinician Terms",
        "summary": "This account type is intended for authorised healthcare professionals who need structured evidence support in practice.",
        "bullets": [
            "Outputs are for professional support only and must not replace regulated judgement, supervision, or approved organisational policy.",
            "You remain responsible for validating accuracy, relevance, and applicability before using any output in clinical work.",
            "Only process patient information when you have permission and a lawful basis to do so.",
        ],
        "acknowledgement": "I will use this clinician account as support only and will verify any output before applying it in practice.",
    },
}

PRIVACY_NOTICE_POINTS = [
    "We store your account profile, password hash, conversation history, uploads, and audit events so your account can persist across sessions.",
    "Passwords are stored as one-way cryptographic hashes and are never written in plain text.",
    "You should only upload information you are entitled to use and share.",
    f"Support, privacy, and account questions can be sent to {SUPPORT_EMAIL}.",
]


def normalize_role_label(role_label: str) -> str:
    return (role_label or "").strip().lower()


def is_clinician_role(role_label: str) -> bool:
    return normalize_role_label(role_label) in CLINICIAN_ROLES


def get_terms_for_role(role_label: str) -> Dict[str, List[str] | str]:
    return ROLE_TERMS.get(role_label, ROLE_TERMS["Patient / Individual"])


def default_care_context_for_role(role_label: str) -> str:
    normalized_role = normalize_role_label(role_label)
    return DEFAULT_CARE_CONTEXT_BY_ROLE.get(normalized_role, "Personal health guidance")
