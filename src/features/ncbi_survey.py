"""
NCBI availability survey for ESKAPE species.

Uses NCBI E-utilities (Entrez) API  -  stable, well-documented, no extra installs.
Queries complete genome counts and samples metadata (country, year, isolation
source) for sampling strategy decisions in Phase 2.

Runs on system Python 3.9+  -  only stdlib (urllib, xml, json, collections).

Usage:
    cd eskape-defence-ml
    python src/features/ncbi_survey.py

Output:
    Console: summary tables per species.
    File: data/interim/ncbi_survey.json (raw results for reference).

NCBI E-utilities rate limit: 3 requests/sec without API key.
Set NCBI_API_KEY env var to get 10 req/sec if you have one.
"""

import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter

# ── ESKAPE species: NCBI taxonomy IDs ─────────────────────────────────────────
SPECIES = {
    "E_faecium":    {"taxid": "1352",  "name": "Enterococcus faecium"},
    "S_aureus":     {"taxid": "1280",  "name": "Staphylococcus aureus"},
    "K_pneumoniae": {"taxid": "573",   "name": "Klebsiella pneumoniae"},
    "A_baumannii":  {"taxid": "470",   "name": "Acinetobacter baumannii"},
    "P_aeruginosa": {"taxid": "287",   "name": "Pseudomonas aeruginosa"},
    "E_cloacae":    {"taxid": "550",   "name": "Enterobacter cloacae"},
}

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
API_KEY = os.environ.get("NCBI_API_KEY", "")
# Sleep between requests: 0.34s without key (~3/s), 0.11s with key (~9/s)
SLEEP = 0.12 if API_KEY else 0.36
# How many assemblies to sample per species for metadata estimation
METADATA_SAMPLE_SIZE = 300


def eutils_url(endpoint: str, params: dict) -> str:
    """Build an E-utilities URL with optional API key."""
    if API_KEY:
        params["api_key"] = API_KEY
    return f"{EUTILS_BASE}/{endpoint}?{urllib.parse.urlencode(params)}"


def fetch(url: str, as_json: bool = False):
    """Fetch URL content with retries. Returns parsed JSON or raw bytes."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "eskape-defence-ml/1.0 (muthuramanvigneshwaran@gmail.com)"}
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read()
            return json.loads(raw) if as_json else raw
        except urllib.error.HTTPError as e:
            print(f"    HTTP {e.code} (attempt {attempt+1}/3): {url[:80]}...")
            time.sleep(2 ** attempt)
        except Exception as e:
            print(f"    Error (attempt {attempt+1}/3): {e}")
            time.sleep(2 ** attempt)
    return None


def esearch_count(taxid: str) -> tuple[int, str, str]:
    """
    Search assembly DB for complete genomes. Returns (count, webenv, query_key).
    webenv + query_key let us reuse the search result in subsequent calls.
    """
    url = eutils_url("esearch.fcgi", {
        "db": "assembly",
        # txid[Organism:exp] = exact taxon subtree match, not fuzzy name search
        "term": f'txid{taxid}[Organism:exp] AND "complete genome"[Assembly Level] AND latest[filter]',
        "retmax": 10000,   # fetch all UIDs, not just count
        "usehistory": "y",
        "retmode": "json",
    })
    time.sleep(SLEEP)
    result = fetch(url, as_json=True)
    if not result:
        return 0, "", ""
    er = result.get("esearchresult", {})
    count = int(er.get("count", 0))
    webenv = er.get("webenv", "")
    query_key = er.get("querykey", "")
    ids = er.get("idlist", [])
    return count, webenv, query_key, ids


def esummary_batch(uids: list[str]) -> dict:
    """Fetch assembly esummary for a batch of UIDs. Returns dict keyed by UID."""
    if not uids:
        return {}
    url = eutils_url("esummary.fcgi", {
        "db": "assembly",
        "id": ",".join(uids),
        "retmode": "json",
    })
    time.sleep(SLEEP)
    result = fetch(url, as_json=True)
    if not result:
        return {}
    return result.get("result", {})


def efetch_biosample_batch(biosample_ids: list[str]) -> list[ET.Element]:
    """
    Fetch BioSample XML records for a list of BioSample accessions (SAMN...).
    Returns list of parsed XML root elements (one per biosample).
    """
    if not biosample_ids:
        return []
    url = eutils_url("efetch.fcgi", {
        "db": "biosample",
        "id": ",".join(biosample_ids),
        "rettype": "xml",
        "retmode": "xml",
    })
    time.sleep(SLEEP)
    raw = fetch(url, as_json=False)
    if not raw:
        return []
    try:
        root = ET.fromstring(raw)
        # BioSampleSet contains multiple BioSample elements
        return root.findall("BioSample")
    except ET.ParseError:
        return []


def parse_biosample(bs: ET.Element) -> dict:
    """Extract country, collection_date, isolation_source from a BioSample element."""
    attrs = {}
    for attr in bs.findall(".//Attribute"):
        name = (attr.get("attribute_name") or attr.get("harmonized_name") or "").lower()
        val = (attr.text or "").strip()
        attrs[name] = val

    # Controlled vocabulary mappings
    fields_country = ["geo_loc_name", "geographic location", "country"]
    fields_date    = ["collection_date", "collection date"]
    fields_iso     = ["isolation_source", "isolation source"]

    country_raw = next((attrs[f] for f in fields_country if f in attrs), "")
    date_raw    = next((attrs[f] for f in fields_date    if f in attrs), "")
    iso_raw     = next((attrs[f] for f in fields_iso     if f in attrs), "")

    return {
        "country": country_raw.split(":")[0].strip() if country_raw else "unknown",
        "year_bin": _bin_year(date_raw),
        "iso_class": _classify_iso(iso_raw),
    }


def _bin_year(date_str: str) -> str:
    if not date_str:
        return "unknown"
    for part in date_str.replace("-", " ").replace("/", " ").split():
        if part.isdigit() and len(part) == 4:
            y = int(part)
            if y < 2010:   return "pre-2010"
            if y < 2016:   return "2010-2015"
            if y < 2021:   return "2016-2020"
            return "2021-present"
    return "unknown"


def _classify_iso(s: str) -> str:
    s = s.lower()
    if not s or s in ("unknown", "not collected", "not applicable",
                      "missing", "na", "n/a", "none", ""):
        return "unknown"
    if any(t in s for t in ["blood", "urine", "sputum", "wound", "csf",
                             "swab", "respiratory", "bronch", "trachea",
                             "lung", "pleural", "abscess", "pus", "clinical",
                             "patient", "hospital", "icu", "catheter", "burn",
                             "human", "infection", "biopsy", "tonsil"]):
        return "clinical_human"
    if any(t in s for t in ["veterinary", "bovine", "cow", "pig", "swine",
                             "poultry", "chicken", "sheep", "horse", "equine",
                             "canine", "feline", "animal", "livestock"]):
        return "clinical_animal"
    if any(t in s for t in ["water", "river", "lake", "sea", "marine",
                             "ocean", "effluent", "wastewater", "sewage",
                             "drain", "aquatic"]):
        return "environmental_water"
    if any(t in s for t in ["soil", "sediment", "rhizosphere", "plant",
                             "root", "compost", "land", "dirt"]):
        return "environmental_soil"
    if any(t in s for t in ["food", "meat", "milk", "cheese", "vegetable",
                             "retail", "produce"]):
        return "food"
    return "unknown"


def survey_species(key: str, info: dict) -> dict:
    taxid = info["taxid"]
    name  = info["name"]

    print(f"\n{'─'*60}")
    print(f"  {name}  (taxid={taxid})")

    # ── Step 1: count + get all UIDs ──────────────────────────────────────────
    count, webenv, query_key, all_uids = esearch_count(taxid)
    print(f"  Complete genomes available: {count:,}")

    if count == 0 or not all_uids:
        return {"name": name, "taxid": taxid, "total": 0}

    # ── Step 2: sample UIDs for metadata (avoid fetching thousands) ───────────
    random.seed(42)
    sample_uids = (
        random.sample(all_uids, METADATA_SAMPLE_SIZE)
        if len(all_uids) > METADATA_SAMPLE_SIZE
        else all_uids
    )
    print(f"  Sampling {len(sample_uids)} assemblies for metadata...", flush=True)

    # ── Step 3: assembly esummary → get BioSample accessions ─────────────────
    biosample_ids = []
    batch_size = 100
    for i in range(0, len(sample_uids), batch_size):
        batch = sample_uids[i:i + batch_size]
        summary = esummary_batch(batch)
        for uid, rec in summary.items():
            if uid == "uids":
                continue
            bs_acc = rec.get("biosampleaccn", "")
            if bs_acc:
                biosample_ids.append(bs_acc)

    print(f"  BioSample accessions found: {len(biosample_ids)}", flush=True)

    # ── Step 4: fetch BioSample records in batches of 100 ────────────────────
    countries   = Counter()
    year_bins   = Counter()
    iso_sources = Counter()

    bs_batch = 100
    for i in range(0, len(biosample_ids), bs_batch):
        batch = biosample_ids[i:i + bs_batch]
        elements = efetch_biosample_batch(batch)
        for bs_elem in elements:
            parsed = parse_biosample(bs_elem)
            countries[parsed["country"]]  += 1
            year_bins[parsed["year_bin"]] += 1
            iso_sources[parsed["iso_class"]] += 1

    sampled = sum(iso_sources.values())
    print(f"  Metadata parsed for {sampled} genomes")

    return {
        "name":              name,
        "taxid":             taxid,
        "total":             count,
        "sampled":           sampled,
        "countries":         dict(countries.most_common(20)),
        "year_bins":         dict(year_bins),
        "isolation_sources": dict(iso_sources),
    }


def print_summary(r: dict) -> None:
    total    = r["total"]
    sampled  = r.get("sampled", 0)
    if total == 0:
        print(f"  No complete genomes found.")
        return

    print(f"\n  Countries (from {sampled}-genome sample):")
    for country, n in list(r.get("countries", {}).items())[:12]:
        bar = "█" * min(n // 3, 40)
        print(f"    {country:<32} {n:>4}  {bar}")

    print(f"\n  Collection year bins:")
    for yr in ["pre-2010", "2010-2015", "2016-2020", "2021-present", "unknown"]:
        n = r.get("year_bins", {}).get(yr, 0)
        bar = "█" * min(n // 3, 40)
        print(f"    {yr:<22} {n:>4}  {bar}")

    print(f"\n  Isolation source (classified):")
    for src, n in sorted(r.get("isolation_sources", {}).items(), key=lambda x: -x[1]):
        pct = 100 * n / sampled if sampled else 0
        print(f"    {src:<30} {n:>4}  ({pct:.0f}%)")


def main():
    os.makedirs("data/interim", exist_ok=True)

    print("NCBI ESKAPE Genome Availability Survey")
    print("=" * 60)
    print(f"API key: {'set' if API_KEY else 'not set (3 req/s limit)'}")
    print(f"Metadata sample per species: {METADATA_SAMPLE_SIZE}")

    results = {}
    for key, info in SPECIES.items():
        results[key] = survey_species(key, info)
        print_summary(results[key])

    # ── Save raw results ──────────────────────────────────────────────────────
    out = "data/interim/ncbi_survey.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nRaw data → {out}")

    # ── Decision-relevant summary ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SAMPLING DECISION TABLE")
    print(f"{'Species':<30} {'Total':>7}  {'Clinical%':>10}  {'Unknown%':>10}  {'#Countries':>11}")
    print("-" * 75)
    for key, r in results.items():
        if r["total"] == 0:
            print(f"  {r['name']:<28}  NO COMPLETE GENOMES")
            continue
        iso  = r.get("isolation_sources", {})
        samp = r.get("sampled", 1) or 1
        clin = iso.get("clinical_human", 0) + iso.get("clinical_animal", 0)
        unk  = iso.get("unknown", 0)
        nc   = len(r.get("countries", {}))
        print(f"  {r['name']:<28}  {r['total']:>7,}  "
              f"{100*clin/samp:>9.0f}%  {100*unk/samp:>9.0f}%  {nc:>11}")


if __name__ == "__main__":
    main()
