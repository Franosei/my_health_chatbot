#!/usr/bin/env python3
# fda_drug_event_demo.py

import argparse
import datetime as dt
import sys
from typing import Dict, List, Optional
import requests

BASE_URL = "https://api.fda.gov/drug/event.json"


# -------------------- HTTP helper --------------------

def _get(url: str, params: Dict[str, str]) -> Dict:
    """GET with basic error handling; tolerate openFDA's 404 for 'no matches'."""
    try:
        r = requests.get(url, params=params, timeout=20)
        if r.status_code == 404:
            # openFDA returns 404 when the query has zero matches
            return {"meta": {"results": {"total": 0}}, "results": []}
        r.raise_for_status()
        return r.json()
    except requests.HTTPError:
        sys.stderr.write(f"[HTTP {r.status_code}] {r.text}\n")
        raise
    except Exception as e:
        sys.stderr.write(f"[ERROR] {e}\n")
        raise


# -------------------- Query builders --------------------

def _build_drug_query(drug: str) -> str:
    """
    Build a robust search string across common FAERS fields.
    Use plain spaces; 'requests' will URL-encode correctly.
    """
    drug_q = drug.strip()
    return (
        f'(patient.drug.medicinalproduct:"{drug_q}") '
        f'OR patient.drug.openfda.brand_name.exact:"{drug_q}" '
        f'OR patient.drug.openfda.generic_name.exact:"{drug_q}"'
    )

def _date_range_clause(start: Optional[str], end: Optional[str]) -> Optional[str]:
    """Return a receivedate clause like: receivedate:[YYYYMMDD TO YYYYMMDD]."""
    if not start and not end:
        return None
    if not start:
        start = "19000101"
    if not end:
        end = dt.date.today().strftime("%Y%m%d")
    return f"receivedate:[{start} TO {end}]"

def _and_join(*clauses: Optional[str]) -> str:
    """Join non-empty search clauses with ' AND '."""
    parts = [c for c in clauses if c]
    return " AND ".join(parts) if parts else ""


# -------------------- API query functions --------------------

def query_meta(drug: str, start: Optional[str], end: Optional[str]) -> Dict:
    q_drug = _build_drug_query(drug)
    q_date = _date_range_clause(start, end)
    search = _and_join(q_drug, q_date)
    params = {"search": search, "limit": "1"}
    return _get(BASE_URL, params)

def query_top_reactions(drug: str, start: Optional[str], end: Optional[str], top_n: int = 15) -> List[Dict]:
    q_drug = _build_drug_query(drug)
    q_date = _date_range_clause(start, end)
    search = _and_join(q_drug, q_date)
    params = {
        "search": search,
        "count": "patient.reaction.reactionmeddrapt.exact",
        "limit": str(top_n),
    }
    data = _get(BASE_URL, params)
    return data.get("results", [])

def query_seriousness_breakdown(drug: str, start: Optional[str], end: Optional[str]) -> Dict[str, int]:
    """
    Count FAERS seriousness flags where value == "1".
    Fields: serious, seriousnessdeath, seriousnesslifethreatening,
            seriousnesshospitalization, seriousnessdisabling,
            seriousnesscongenitalanomali, seriousnessother.
    """
    q_drug = _build_drug_query(drug)
    q_date = _date_range_clause(start, end)
    search = _and_join(q_drug, q_date)

    breakdown: Dict[str, int] = {}
    fields = [
        "serious",
        "seriousnessdeath",
        "seriousnesslifethreatening",
        "seriousnesshospitalization",
        "seriousnessdisabling",
        "seriousnesscongenitalanomali",
        "seriousnessother",
    ]
    for f in fields:
        params = {"search": search, "count": f"{f}.exact", "limit": "10"}
        data = _get(BASE_URL, params)
        total_ones = 0
        for row in data.get("results", []):
            if str(row.get("term")) == "1":
                total_ones += int(row.get("count", 0))
        breakdown[f] = total_ones
    return breakdown

def query_recent_events(drug: str, start: Optional[str], end: Optional[str], limit: int = 5) -> List[Dict]:
    """Fetch a few recent event records for inspection."""
    q_drug = _build_drug_query(drug)
    q_date = _date_range_clause(start, end)
    search = _and_join(q_drug, q_date)
    params = {"search": search, "limit": str(limit), "sort": "receivedate:desc"}
    data = _get(BASE_URL, params)
    return data.get("results", [])


# -------------------- CLI & main --------------------

def main():
    ap = argparse.ArgumentParser(description="Query openFDA drug adverse events (FAERS).")
    ap.add_argument("--drug", required=True, help='Drug name, e.g., "dexamethasone" or "ibuprofen"')
    ap.add_argument("--start", help="Start date (YYYYMMDD), optional")
    ap.add_argument("--end", help="End date (YYYYMMDD), optional")
    ap.add_argument("--limit", type=int, default=5, help="How many recent events to fetch (default 5)")
    ap.add_argument("--top", type=int, default=15, help="How many top reactions to list (default 15)")
    args = ap.parse_args()

    print(f"\n=== openFDA FAERS: {args.drug} ===")
    if args.start or args.end:
        print(f"Date window: {args.start or '19000101'} → {args.end or dt.date.today().strftime('%Y%m%d')}")

    # 1) Meta
    meta = query_meta(args.drug, args.start, args.end)
    total = meta.get("meta", {}).get("results", {}).get("total", 0)
    print(f"\n[Meta] Total matching reports: {total}")

    if total == 0:
        print("\nNo matches found for this query. Try a brand name (e.g., 'Decadron') or adjust the date window.\n")
        return

    # 2) Top reactions
    print(f"\n[Top Reactions] (MedDRA PT, count) — top {args.top}")
    reactions = query_top_reactions(args.drug, args.start, args.end, top_n=args.top)
    for r in reactions:
        term = r.get("term", "—")
        count = r.get("count", 0)
        print(f" - {term}: {count}")

    # 3) Seriousness breakdown
    print("\n[Seriousness Breakdown] (count where field == 1)")
    serious = query_seriousness_breakdown(args.drug, args.start, args.end)
    for k in [
        "serious",
        "seriousnessdeath",
        "seriousnesslifethreatening",
        "seriousnesshospitalization",
        "seriousnessdisabling",
        "seriousnesscongenitalanomali",
        "seriousnessother",
    ]:
        print(f" - {k}: {serious.get(k, 0)}")

    # 4) Recent events
    print(f"\n[Recent Events] latest {args.limit}")
    events = query_recent_events(args.drug, args.start, args.end, limit=args.limit)
    for i, ev in enumerate(events, 1):
        recv = ev.get("receivedate", "—")
        sid = ev.get("safetyreportid", "—")
        rx = [rx.get("reactionmeddrapt") for rx in ev.get("patient", {}).get("reaction", []) if rx.get("reactionmeddrapt")]
        prods = [d.get("medicinalproduct") for d in ev.get("patient", {}).get("drug", []) if d.get("medicinalproduct")]
        print(f"\n  #{i} receivedate={recv}  safetyreportid={sid}")
        print(f"     reactions: {', '.join(rx[:10]) if rx else '—'}")
        print(f"     products : {', '.join(prods[:6]) if prods else '—'}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
