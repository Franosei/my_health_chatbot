"""Minimal FHIR R4 resource shapes + mapping to/from the internal Patient model.

Not exhaustive -- only the fields FlynnMed's own features would plausibly
read or write are modeled. These types and mapping functions are unused in
this phase; they exist to prove the internal schema (backend/models/patient.py)
won't need reshaping when a later phase wires up a real EHR connection.

Reference: https://hl7.org/fhir/R4/patient.html,
https://hl7.org/fhir/R4/encounter.html, https://hl7.org/fhir/R4/observation.html
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from backend.models.patient import Patient

# The identifier system FlynnMed's own MRN is namespaced under when expressed
# as a FHIR Identifier -- lets a real EHR distinguish "FlynnMed's ID for this
# patient" from its own MRN or any other system's identifier.
FLYNNMED_MRN_SYSTEM = "urn:flynnmed:mrn"


class FHIRIdentifier(BaseModel):
    system: str
    value: str


class FHIRHumanName(BaseModel):
    family: Optional[str] = None
    given: list[str] = Field(default_factory=list)


class FHIRPatient(BaseModel):
    resourceType: str = "Patient"
    id: Optional[str] = None
    identifier: list[FHIRIdentifier] = Field(default_factory=list)
    name: list[FHIRHumanName] = Field(default_factory=list)
    birthDate: Optional[str] = None  # noqa: N815 -- FHIR field name, not our naming convention
    gender: Optional[str] = None  # FHIR's fixed set: male | female | other | unknown


class FHIREncounter(BaseModel):
    resourceType: str = "Encounter"
    id: Optional[str] = None
    status: str = "unknown"
    subjectReference: Optional[str] = None  # noqa: N815 -- FHIR field name
    periodStart: Optional[str] = None  # noqa: N815 -- FHIR field name
    periodEnd: Optional[str] = None  # noqa: N815 -- FHIR field name


class FHIRObservation(BaseModel):
    resourceType: str = "Observation"
    id: Optional[str] = None
    status: str = "unknown"
    code_text: Optional[str] = None
    value_text: Optional[str] = None
    effectiveDateTime: Optional[str] = None  # noqa: N815 -- FHIR field name


_FHIR_GENDER_BY_BIOLOGICAL_SEX = {
    "male": "male",
    "female": "female",
    "other": "other",
    "prefer not to say": "unknown",
}


def to_fhir_patient(patient: "Patient") -> FHIRPatient:
    """Map an internal Patient row to a FHIR Patient resource, keyed by MRN."""
    display_name = (patient.account.display_name if patient.account else "").strip()
    given, _, family = display_name.rpartition(" ") if " " in display_name else ("", "", display_name)

    return FHIRPatient(
        id=str(patient.id),
        identifier=[FHIRIdentifier(system=FLYNNMED_MRN_SYSTEM, value=patient.patient_id)],
        name=[FHIRHumanName(family=family or None, given=[given] if given else [])],
        birthDate=patient.date_of_birth.isoformat() if patient.date_of_birth else None,
        gender=_FHIR_GENDER_BY_BIOLOGICAL_SEX.get((patient.biological_sex or "").strip().lower()),
    )


def from_fhir_patient(fhir_patient: FHIRPatient) -> dict:
    """Map an inbound FHIR Patient resource to the subset of internal Patient
    fields it can populate. Returns a plain dict (not an ORM instance) since
    a Patient row also requires an account_id this resource doesn't carry."""
    mrn = next(
        (ident.value for ident in fhir_patient.identifier if ident.system == FLYNNMED_MRN_SYSTEM),
        None,
    )
    name = fhir_patient.name[0] if fhir_patient.name else FHIRHumanName()
    display_name = " ".join(part for part in [*name.given, name.family] if part).strip()

    return {
        "patient_id": mrn,
        "display_name": display_name or None,
        "date_of_birth": fhir_patient.birthDate,
        "biological_sex": fhir_patient.gender,
    }
