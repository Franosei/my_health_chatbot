from __future__ import annotations

import os
from typing import Optional

from backend.fhir.interface import EHRClient, EHRNotConfiguredError
from backend.fhir.resources import FHIREncounter, FHIRObservation, FHIRPatient


class NullEHRClient:
    """The only EHRClient implementation in this phase. Every method raises
    EHRNotConfiguredError -- an explicit, catchable signal -- rather than
    crashing or silently returning empty data, so callers in a live request
    path can show "EHR integration not connected" instead of a 500 or a
    misleadingly empty result."""

    def get_patient(self, fhir_patient_ref: str) -> FHIRPatient:
        raise EHRNotConfiguredError("No EHR connection is configured (EHR_PROVIDER=none).")

    def get_encounters(self, fhir_patient_ref: str, since: Optional[str] = None) -> list[FHIREncounter]:
        raise EHRNotConfiguredError("No EHR connection is configured (EHR_PROVIDER=none).")

    def get_observations(self, fhir_patient_ref: str, category: Optional[str] = None) -> list[FHIRObservation]:
        raise EHRNotConfiguredError("No EHR connection is configured (EHR_PROVIDER=none).")

    def create_document_reference(self, fhir_patient_ref: str, document: dict) -> str:
        raise EHRNotConfiguredError("No EHR connection is configured (EHR_PROVIDER=none).")


_PROVIDERS: dict[str, type[EHRClient]] = {
    "none": NullEHRClient,
}


def get_ehr_client() -> EHRClient:
    """Reads EHR_PROVIDER (default "none") and returns the matching client.
    Only "none" -> NullEHRClient exists today; a later phase registers real
    vendor clients here without callers needing to change."""
    provider = os.getenv("EHR_PROVIDER", "none").strip().lower()
    client_cls = _PROVIDERS.get(provider, NullEHRClient)
    return client_cls()


def fhir_integration_status() -> dict:
    provider = os.getenv("EHR_PROVIDER", "none").strip().lower()
    return {"connected": provider != "none" and provider in _PROVIDERS, "provider": provider}
