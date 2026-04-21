"""
Build stratified accession lists for all ESKAPE species.

Queries NCBI E-utilities for complete genome assemblies, applies:
  - Per-country cap (max 30 per species, configurable)
  - Year-bin stratification (best effort)
  - Per-member caps for E. cloacae complex
  - Random seed 42 throughout

Writes final GCF_ accession lists back into config/species.yaml.

Usage:
    cd eskape-defence-ml
    python src/features/build_accession_lists.py

Runtime: ~10-15 min (NCBI rate limits). Set NCBI_API_KEY for 3x speedup.
"""

import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict

import yaml  # requires pyyaml — run after: conda activate eskape-ml

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
API_KEY = os.environ.get("NCBI_API_KEY", "")
SLEEP = 0.12 if API_KEY else 0.36
RANDOM_SEED = 42

CONFIG_PATH = "config/species.yaml"
INTERIM_DIR = "data/interim"


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def fetch(url: str, as_json: bool = False):
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
            print(f"    HTTP {e.code} (attempt {attempt+1}/3)")
            time.sleep(2 ** attempt)
        except Exception as e:
            print(f"    Error (attempt {attempt+1}/3): {e}")
            time.sleep(2 ** attempt)
    return None


def eurl(endpoint: str, params: dict) -> str:
    if API_KEY:
        params["api_key"] = API_KEY
    return f"{EUTILS}/{endpoint}?{urllib.parse.urlencode(params)}"


# ── NCBI queries ──────────────────────────────────────────────────────────────

def fetch_all_uids(taxid: str) -> list[str]:
    """Return all assembly UIDs for complete genomes of a taxon."""
    url = eurl("esearch.fcgi", {
        "db": "assembly",
        "term": (f'txid{taxid}[Organism:exp] AND '
                 f'"complete genome"[Assembly Level] AND latest[filter]'),
        "retmax": 10000,
        "usehistory": "y",
        "retmode": "json",
    })
    time.sleep(SLEEP)
    result = fetch(url, as_json=True)
    if not result:
        return []
    return result.get("esearchresult", {}).get("idlist", [])


def fetch_assembly_summary(uids: list[str]) -> list[dict]:
    """
    Fetch esummary records for a list of assembly UIDs.
    Returns list of dicts with accession, biosample_acc, release_date.
    Processes in batches of 200.
    """
    records = []
    batch_size = 200
    for i in range(0, len(uids), batch_size):
        batch = uids[i:i + batch_size]
        url = eurl("esummary.fcgi", {
            "db": "assembly",
            "id": ",".join(batch),
            "retmode": "json",
        })
        time.sleep(SLEEP)
        result = fetch(url, as_json=True)
        if not result:
            continue
        data = result.get("result", {})
        for uid in batch:
            rec = data.get(uid, {})
            if not rec:
                continue
            acc = rec.get("assemblyaccession", "")
            # Only keep RefSeq accessions (GCF_), not GenBank (GCA_)
            if not acc.startswith("GCF_"):
                continue
            records.append({
                "uid":          uid,
                "accession":    acc,
                "biosample":    rec.get("biosampleaccn", ""),
                "organism":     rec.get("speciesname", ""),
                "release_date": rec.get("seqreleasedate", ""),
                "country":      "unknown",   # enriched from biosample below
                "year_bin":     _bin_year(rec.get("seqreleasedate", "")),
            })
        if i % 1000 == 0 and i > 0:
            print(f"    Fetched summaries for {i}/{len(uids)} records...", flush=True)
    return records


def enrich_country(records: list[dict], sample_n: int = 500) -> list[dict]:
    """
    Fetch BioSample records for a random sample of assemblies to get country.
    Country is used to apply the per-country cap in stratified sampling.
    We sample rather than fetch all to stay within rate limits.
    """
    random.seed(RANDOM_SEED)
    to_enrich = (
        random.sample(records, min(sample_n, len(records)))
        if len(records) > sample_n else records
    )
    bs_to_rec = {r["biosample"]: r for r in to_enrich if r["biosample"]}

    import xml.etree.ElementTree as ET

    batch_size = 100
    bs_ids = list(bs_to_rec.keys())
    for i in range(0, len(bs_ids), batch_size):
        batch = bs_ids[i:i + batch_size]
        url = eurl("efetch.fcgi", {
            "db": "biosample",
            "id": ",".join(batch),
            "rettype": "xml",
            "retmode": "xml",
        })
        time.sleep(SLEEP)
        raw = fetch(url, as_json=False)
        if not raw:
            continue
        try:
            root = ET.fromstring(raw)
        except ET.ParseError:
            continue
        for bs_elem in root.findall("BioSample"):
            acc = bs_elem.get("accession", "")
            if acc not in bs_to_rec:
                continue
            for attr in bs_elem.findall(".//Attribute"):
                name = (attr.get("attribute_name") or
                        attr.get("harmonized_name") or "").lower()
                if name in ("geo_loc_name", "geographic location", "country"):
                    val = (attr.text or "").strip()
                    country = val.split(":")[0].strip() if val else "unknown"
                    bs_to_rec[acc]["country"] = country
                    break
    return records


def _bin_year(date_str: str) -> str:
    if not date_str:
        return "unknown"
    for part in date_str.replace("-", " ").replace("/", " ").split():
        if part.isdigit() and len(part) == 4:
            y = int(part)
            if y < 2010:  return "pre-2010"
            if y < 2016:  return "2010-2015"
            if y < 2021:  return "2016-2020"
            return "2021-present"
    return "unknown"


# ── Stratified sampler ────────────────────────────────────────────────────────

def stratified_sample(
    records: list[dict],
    target_n: int,
    country_cap: int,
    seed: int = RANDOM_SEED,
) -> list[dict]:
    """
    Sample `target_n` records with:
      1. Per-country cap: no more than `country_cap` from any single country.
      2. Year-bin balance: within the country-capped pool, prefer even
         representation across year bins (best effort — not enforced as hard cap).
      3. Random tie-breaking with fixed seed.

    Why this order: country cap is the hard constraint (geographic bias
    protection). Year balance is a soft preference because year metadata is
    often missing.
    """
    random.seed(seed)
    random.shuffle(records)   # randomise before any selection

    # Step 1 — apply country cap
    country_counts: Counter = Counter()
    country_capped: list[dict] = []
    for r in records:
        c = r.get("country", "unknown") or "unknown"
        if country_counts[c] < country_cap:
            country_capped.append(r)
            country_counts[c] += 1

    if len(country_capped) <= target_n:
        return country_capped   # can't reach target even without further filtering

    # Step 2 — year-bin stratification within country-capped pool
    # Group by year bin, then interleave to spread years evenly
    by_year: dict = defaultdict(list)
    for r in country_capped:
        by_year[r.get("year_bin", "unknown")].append(r)

    # Round-robin across year bins until we have target_n
    selected: list[dict] = []
    bins = list(by_year.values())
    idx = 0
    while len(selected) < target_n and any(bins):
        bucket = bins[idx % len(bins)]
        if bucket:
            selected.append(bucket.pop(0))
        idx += 1

    return selected[:target_n]


# ── Per-species orchestration ─────────────────────────────────────────────────

def build_for_single_taxid(
    taxid: str,
    member_name: str,
    sample_target: int,
    country_cap: int,
) -> list[dict]:
    """Fetch and sample records for one taxid."""
    print(f"    [{member_name}] fetching UIDs...", flush=True)
    uids = fetch_all_uids(taxid)
    print(f"    [{member_name}] {len(uids)} UIDs found", flush=True)
    if not uids:
        return []

    records = fetch_assembly_summary(uids)
    print(f"    [{member_name}] {len(records)} GCF_ accessions", flush=True)

    # Enrich a sample with country info
    records = enrich_country(records, sample_n=min(len(records), 500))

    sampled = stratified_sample(records, sample_target, country_cap)
    # Tag each record with its complex member name (useful for E. cloacae)
    for r in sampled:
        r["complex_member"] = member_name
    return sampled


def build_for_species(sp_key: str, sp_cfg: dict) -> list[str]:
    """
    Build the final accession list for one ESKAPE species entry.
    Handles both single-taxid species and the multi-taxid E. cloacae complex.
    Returns list of GCF_ accession strings.
    """
    country_cap = sp_cfg.get("country_cap", 30)
    taxid_entries = sp_cfg.get("ncbi_taxids", [])
    all_sampled: list[dict] = []

    for entry in taxid_entries:
        taxid  = entry["taxid"]
        member = entry.get("member", sp_cfg["name"])
        target = entry.get("sample_target", sp_cfg["target_n"])
        sampled = build_for_single_taxid(taxid, member, target, country_cap)
        all_sampled.extend(sampled)

    accessions = [r["accession"] for r in all_sampled]

    # Save per-species metadata for QC and downstream use
    os.makedirs(INTERIM_DIR, exist_ok=True)
    meta_path = os.path.join(INTERIM_DIR, f"{sp_key}_accession_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(all_sampled, f, indent=2)

    return accessions


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    print("Building stratified accession lists for ESKAPE species")
    print("=" * 60)
    print(f"Country cap: 30 per species  |  Seed: {RANDOM_SEED}")
    print(f"API key: {'set' if API_KEY else 'not set (3 req/s)'}\n")

    for sp_key, sp_cfg in config["species"].items():
        print(f"\n{'─'*60}")
        print(f"Species: {sp_cfg['name']}")
        accessions = build_for_species(sp_key, sp_cfg)
        config["species"][sp_key]["accessions"] = accessions
        print(f"  Final accession count: {len(accessions)}")

    # Write updated config with accession lists
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True,
                  sort_keys=False)

    print(f"\n\nAccession lists written to {CONFIG_PATH}")
    print("Per-species metadata written to data/interim/*_accession_metadata.json")

    # Summary table
    print("\n" + "=" * 50)
    print("FINAL SAMPLE SIZES")
    print(f"{'Species':<35} {'N':>6}")
    print("-" * 42)
    total = 0
    for sp_key, sp_cfg in config["species"].items():
        n = len(sp_cfg.get("accessions", []))
        total += n
        print(f"  {sp_cfg['name']:<33} {n:>6}")
    print(f"  {'TOTAL':<33} {total:>6}")


if __name__ == "__main__":
    main()
