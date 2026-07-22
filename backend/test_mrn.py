from backend.mrn import generate_mrn, is_valid_mrn, normalize_mrn


def test_generated_mrn_has_expected_shape():
    mrn = generate_mrn()
    assert mrn.startswith("FM-")
    body = mrn.split("-", 1)[1]
    groups = body.split("-")
    assert len(groups) == 2
    assert all(len(g) == 4 for g in groups)


def test_generated_mrn_is_valid():
    for _ in range(200):
        assert is_valid_mrn(generate_mrn())


def test_generated_mrns_are_unique():
    mrns = {generate_mrn() for _ in range(500)}
    assert len(mrns) == 500


def test_mrns_never_contain_ambiguous_characters():
    ambiguous = set("ILOU")
    for _ in range(200):
        mrn = generate_mrn()
        assert not (ambiguous & set(mrn))


def test_lowercase_and_unformatted_input_still_validates():
    mrn = generate_mrn()
    assert is_valid_mrn(mrn.lower())
    assert is_valid_mrn(mrn.replace("-", ""))


def test_tampered_check_digit_is_rejected():
    mrn = generate_mrn()
    last_char = mrn[-1]
    replacement = "1" if last_char != "1" else "2"
    tampered = mrn[:-1] + replacement
    assert not is_valid_mrn(tampered)


def test_garbage_input_is_rejected():
    assert not is_valid_mrn("")
    assert not is_valid_mrn("not-an-mrn")
    assert not is_valid_mrn("FM-0000-000")  # wrong length


def test_normalize_mrn_strips_formatting_and_uppercases():
    assert normalize_mrn(" fm-7k2q-9xhd ") == "FM7K2Q9XHD"
