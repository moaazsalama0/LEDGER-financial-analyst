from app.schemas import Evidence


def test_page_number_mirrors_into_page():
    ev = Evidence(document_id="doc_001", page_number=5)
    assert ev.page == 5
    assert ev.page_number == 5


def test_page_mirrors_into_page_number():
    ev = Evidence(document_id="doc_001", page=5)
    assert ev.page == 5
    assert ev.page_number == 5


def test_as_contract_dict_includes_both_keys():
    ev = Evidence(document_id="doc_017", page=1, section="Income Statement")
    d = ev.as_contract_dict()
    assert d == {
        "document_id": "doc_017",
        "page": 1,
        "page_number": 1,
        "section": "Income Statement",
    }


def test_as_contract_dict_omits_section_when_absent():
    ev = Evidence(document_id="doc_017", page=1)
    d = ev.as_contract_dict()
    assert "section" not in d
