"""EHR integration boundary -- interface only, no live implementation.

This package exists so the internal schema (backend/models/patient.py) never
needs reshaping when a later phase actually wires up a real EHR: every shape
an `EHRClient` would need to produce or consume is already defined here and
exercised by backend/fhir/resources.py's mapping functions, even though
nothing in this phase calls a real EHR over the network.

Not implemented in this phase (deliberately, per the foundation-phase scope
decision -- no EHR vendor sandbox credentials are available yet):

- The actual SMART App Launch OAuth2 + PKCE handshake, in either of its two
  documented flows:
    1. EHR launch: the EHR itself opens FlynnMed with a `launch` token and
       `iss` (issuer/FHIR base URL) query param; the app exchanges those at
       the EHR's authorization endpoint for an access token scoped to the
       launch context (which patient/encounter, if any).
    2. Standalone launch: FlynnMed initiates the OAuth2 authorization-code
       flow directly against a known `iss`, without an EHR-provided launch
       context.
  See https://hl7.org/fhir/smart-app-launch/ for the full spec this should
  follow when implemented.
- Any network calls to a real FHIR server.
- Token refresh/storage for a live EHR connection.

`SmartLaunchContext` below documents the parameters that handshake produces,
so `ConsentGrant`/`Patient` fields can be designed today knowing what a real
integration will eventually need to carry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol

from backend.fhir.resources import FHIREncounter, FHIRObservation, FHIRPatient


class EHRNotConfiguredError(RuntimeError):
    """Raised by EHRClient implementations when no EHR connection exists yet.

    Deliberately a normal exception, not a bare crash/NotImplementedError --
    callers in a live request path should catch this and degrade gracefully
    (e.g. "EHR integration not connected" in the UI) rather than 500.
    """


@dataclass(frozen=True)
class SmartLaunchContext:
    """The parameters a completed SMART App Launch handshake produces.

    Not populated or consumed anywhere yet -- this documents the shape a
    later phase's OAuth2 callback handler would construct and persist per
    EHR connection.
    """

    iss: str  # FHIR server base URL (the "issuer") -- identifies which EHR/tenant.
    launch: Optional[str] = None  # Present for EHR-initiated launch only.
    client_id: str = ""
    scopes: tuple[str, ...] = field(default_factory=tuple)  # e.g. ("patient/*.read", "launch")
    patient_fhir_id: Optional[str] = None  # Patient in launch context, if the EHR provided one.
    encounter_fhir_id: Optional[str] = None


class EHRClient(Protocol):
    """What a real EHR integration must be able to do. Implement against a
    specific vendor's FHIR API once a sandbox/production connection exists;
    `backend.fhir.stub_client.NullEHRClient` is the only implementation
    today."""

    def get_patient(self, fhir_patient_ref: str) -> FHIRPatient:
        ...

    def get_encounters(self, fhir_patient_ref: str, since: Optional[str] = None) -> list[FHIREncounter]:
        ...

    def get_observations(self, fhir_patient_ref: str, category: Optional[str] = None) -> list[FHIRObservation]:
        ...

    def create_document_reference(self, fhir_patient_ref: str, document: dict) -> str:
        """Push a generated document (e.g. a GP-prep summary) back to the EHR.
        Returns the created DocumentReference's FHIR id."""
        ...
