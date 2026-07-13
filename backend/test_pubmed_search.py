from backend.pubmed_search import PubMedCentralSearcher


class _Response:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "resultList": {
                "result": [
                    {
                        "pmcid": "PMC123",
                        "title": "Dexamethasone adverse effects in older adults",
                        "journalTitle": "Example Journal",
                        "pubYear": "2025",
                        "authorString": "Example A",
                        "abstractText": "A structured abstract.",
                    }
                ]
            }
        }


def test_pubmed_search(monkeypatch):
    monkeypatch.setattr("backend.pubmed_search.requests.get", lambda *args, **kwargs: _Response())
    searcher = PubMedCentralSearcher()
    query = "dexamethasone elderly adverse effects"

    pmc_ids = searcher.search_articles(query, max_results=3)

    assert pmc_ids == ["PMC123"]
    assert searcher.search_cache[f"{query}::3"][0]["year"] == "2025"
