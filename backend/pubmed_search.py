import requests
from typing import List, Dict
from xml.etree import ElementTree as ET


class PubMedCentralSearcher:
    """
    Searches and retrieves open-access full-text biomedical articles from PubMed Central (PMC).
    Extracts structured sections such as Abstract, Introduction, Discussion, and Conclusion.
    """

    SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    FULLTEXT_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"

    def search_articles(self, query: str, max_results: int = 3) -> List[str]:
        """
        Searches for PMC open-access article IDs using a query string.
        Returns an empty list on failure instead of raising errors.
        """
        params = {
            "query": query + " OPEN_ACCESS:Y",
            "format": "json",
            "pageSize": max_results
        }

        try:
            response = requests.get(self.SEARCH_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            pmc_ids = [
                item["pmcid"]
                for item in data.get("resultList", {}).get("result", [])
                if "pmcid" in item
            ]
            print("PubMed Query:", query)
            print("PMC IDs:", pmc_ids)
            return pmc_ids

        except requests.exceptions.RequestException as e:
            print(f"PubMed API request failed: {e}")
            return []

        except Exception as e:
            print(f"PubMed JSON parse error: {e}")
            return []

    def fetch_article_sections(self, pmcid: str) -> Dict[str, str]:
        """
        Retrieves full-text XML from PMC and extracts relevant sections.
        """
        url = self.FULLTEXT_URL.format(pmcid=pmcid)
        sections = {
            "abstract": "", "introduction": "", "discussion": "", "conclusion": ""
        }

        section_keywords = {
            "abstract": ["abstract"],
            "introduction": ["introduction", "background"],
            "discussion": ["discussion"],
            "conclusion": ["conclusion", "summary", "concluding remarks"]
        }

        def extract_text(elem) -> str:
            return "".join(elem.itertext()).strip()

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            root = ET.fromstring(response.content)

            # Try section titles
            for sec in root.findall(".//sec"):
                title_elem = sec.find("title")
                if title_elem is not None and title_elem.text:
                    title = title_elem.text.lower()
                    for key, keywords in section_keywords.items():
                        if any(kw in title for kw in keywords) and not sections[key]:
                            sections[key] = extract_text(sec)
                            print(f"Found section [{key}] in {pmcid}")

            # Backup: use <abstract>
            abstract_elem = root.find(".//abstract")
            if abstract_elem is not None and not sections["abstract"]:
                sections["abstract"] = extract_text(abstract_elem)
                print(f"Fallback abstract in {pmcid}")

        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch full text for {pmcid}: {e}")

        except ET.ParseError as e:
            print(f"XML parsing error for {pmcid}: {e}")

        except Exception as e:
            print(f"Unexpected error processing {pmcid}: {e}")

        return sections
