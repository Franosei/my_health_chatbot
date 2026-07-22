import datetime
import uuid

import pytest

from backend.fhir.interface import EHRNotConfiguredError
from backend.fhir.resources import FHIRPatient, from_fhir_patient, to_fhir_patient
from backend.fhir.stub_client import NullEHRClient, fhir_integration_status, get_ehr_client
from backend.models.account import Account, AccountKind
from backend.models.patient import Patient


def _make_patient(display_name: str = "Jane Doe") -> Patient:
    account = Account(
        id=uuid.uuid4(),
        username="jane",
        email="jane@example.com",
        display_name=display_name,
        password_hash="x",
        password_algo="argon2id",
        account_kind=AccountKind.patient,
    )
    patient = Patient(
        id=uuid.uuid4(),
        account_id=account.id,
        patient_id="FM-7K2Q-9XHD",
        date_of_birth=datetime.date(1990, 1, 1),
        biological_sex="female",
        longitudinal_memory={},
    )
    patient.account = account
    return patient


def test_to_fhir_patient_maps_mrn_and_name():
    patient = _make_patient("Jane Doe")
    fhir_patient = to_fhir_patient(patient)

    assert fhir_patient.identifier[0].system == "urn:flynnmed:mrn"
    assert fhir_patient.identifier[0].value == "FM-7K2Q-9XHD"
    assert fhir_patient.name[0].family == "Doe"
    assert fhir_patient.name[0].given == ["Jane"]
    assert fhir_patient.birthDate == "1990-01-01"
    assert fhir_patient.gender == "female"


def test_from_fhir_patient_recovers_mrn():
    fhir_patient = FHIRPatient.model_validate(
        {
            "identifier": [{"system": "urn:flynnmed:mrn", "value": "FM-7K2Q-9XHD"}],
            "name": [{"family": "Doe", "given": ["Jane"]}],
            "birthDate": "1990-01-01",
            "gender": "female",
        }
    )
    mapped = from_fhir_patient(fhir_patient)

    assert mapped["patient_id"] == "FM-7K2Q-9XHD"
    assert mapped["display_name"] == "Jane Doe"
    assert mapped["date_of_birth"] == "1990-01-01"
    assert mapped["biological_sex"] == "female"


def test_null_client_is_the_default_and_raises_explicitly():
    client = get_ehr_client()
    assert isinstance(client, NullEHRClient)
    with pytest.raises(EHRNotConfiguredError):
        client.get_patient("Patient/123")
    with pytest.raises(EHRNotConfiguredError):
        client.get_encounters("Patient/123")
    with pytest.raises(EHRNotConfiguredError):
        client.get_observations("Patient/123")
    with pytest.raises(EHRNotConfiguredError):
        client.create_document_reference("Patient/123", {})


def test_fhir_integration_status_reports_disconnected_by_default(monkeypatch):
    monkeypatch.delenv("EHR_PROVIDER", raising=False)
    assert fhir_integration_status() == {"connected": False, "provider": "none"}
