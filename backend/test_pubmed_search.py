from pubmed_search import PubMedCentralSearcher

def test_pubmed_search():
    print("Initializing search...")
    searcher = PubMedCentralSearcher()

    # Use a medically relevant query
    query = "dexamethasone elderly adverse effects"
    print(f"Searching for: {query}")

    pmc_ids = searcher.search_articles(query, max_results=3)
    if not pmc_ids:
        print("No articles found.")
        return

    print(f"Found PMC IDs: {pmc_ids}\n")

    for pmc_id in pmc_ids:
        print(f"Fetching full text for {pmc_id}...")
        sections = searcher.fetch_article_sections(pmc_id)

        print(f"\n--- {pmc_id} ---")
        print("Abstract:\n", sections.get("abstract", "N/A")[:500])
        print("\nIntroduction:\n", sections.get("introduction", "N/A")[:500])
        print("\nDiscussion:\n", sections.get("discussion", "N/A")[:500])
        print("\nConclusion:\n", sections.get("conclusion", "N/A")[:500])
        print("-" * 50)

if __name__ == "__main__":
    test_pubmed_search()
