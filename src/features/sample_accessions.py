"""
sample_accessions.py — Build 600/species superset accession lists for ESKAPE 4× study.

Strategy:
  1. Load original 150 accessions from eskape-defence-ml metadata JSONs (pre-seeding).
  2. Seed country counters with originals BEFORE sampling new genomes, so the
     per-country cap (80/species, 40/EC member) applies across the full 600, not
     just the 450 new additions.
  3. Fetch all current NCBI UIDs for each taxid; remove originals from the pool.
  4. Enrich the new pool with BioSample country metadata (best-effort, 500 sample).
  5. Apply remaining per-country budget to new pool; year-bin stratify; fill to 600.
  6. Assemble: original 150 accessions + up to 450 new = 600 per species.
  7. Write config/species.yaml and per-species metadata JSONs.

EC complex: same logic per-member (country cap 40/member, targets from decisions.md).

Taxids (verified 2026-06-01, see decisions.md for audit trail):
  hormaechei=158836, cloacae_ss=550, asburiae=61645,
  kobei=208224, ludwigii=299767, roggenkampii=1812935

Usage:
    cd eskape-defence-ml_2
    conda run -n eskape-ml python src/features/sample_accessions.py

Runtime: ~20-30 min (NCBI rate limits on 6 species × large pools).
Set NCBI_API_KEY env var for 3× speedup.
"""

import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

import yaml

# ── Constants ─────────────────────────────────────────────────────────────────

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
API_KEY = os.environ.get("NCBI_API_KEY", "")
SLEEP = 0.12 if API_KEY else 0.36
RANDOM_SEED = 42

ORIGINAL_PROJECT = "/Users/Vicky/Acinetobacter_ML_2/eskape-defence-ml"
ORIGINAL_INTERIM = os.path.join(ORIGINAL_PROJECT, "data/interim")
ORIGINAL_CONFIG  = os.path.join(ORIGINAL_PROJECT, "config/species.yaml")

OUTPUT_CONFIG  = "config/species.yaml"
OUTPUT_INTERIM = "data/interim"

# Species config: target_n=600, country_cap=80 per species (40 per EC member).
# Taxids verified 2026-06-01 against live NCBI taxonomy (see decisions.md).
SPECIES_CONFIG = {
    "efaecium": {
        "name": "Enterococcus faecium",
        "short": "Efm",
        "gram": "positive",
        "genome_size_mb_approx": 2.8,
        "target_n": 600,
        "country_cap": 80,
        "ncbi_taxids": [
            {"taxid": "1352", "member": "E. faecium", "sample_target": 600},
        ],
    },
    "saureus": {
        "name": "Staphylococcus aureus",
        "short": "Sa",
        "gram": "positive",
        "genome_size_mb_approx": 2.8,
        "target_n": 600,
        "country_cap": 80,
        "ncbi_taxids": [
            {"taxid": "1280", "member": "S. aureus", "sample_target": 600},
        ],
    },
    "kpneumoniae": {
        "name": "Klebsiella pneumoniae",
        "short": "Kp",
        "gram": "negative",
        "genome_size_mb_approx": 5.4,
        "target_n": 600,
        "country_cap": 80,
        "ncbi_taxids": [
            {"taxid": "573", "member": "K. pneumoniae", "sample_target": 600},
        ],
    },
    "abaumannii": {
        "name": "Acinetobacter baumannii",
        "short": "Ab",
        "gram": "negative",
        "genome_size_mb_approx": 3.9,
        "target_n": 600,
        "country_cap": 80,
        "ncbi_taxids": [
            {"taxid": "470", "member": "A. baumannii", "sample_target": 600},
        ],
    },
    "paeruginosa": {
        "name": "Pseudomonas aeruginosa",
        "short": "Pa",
        "gram": "negative",
        "genome_size_mb_approx": 6.6,
        "target_n": 600,
        "country_cap": 80,
        "ncbi_taxids": [
            {"taxid": "287", "member": "P. aeruginosa", "sample_target": 600},
        ],
    },
    "ecloaceae": {
        "name": "Enterobacter cloacae complex",
        "short": "Ec",
        "gram": "negative",
        "genome_size_mb_approx": 5.0,
        "target_n": 600,
        "country_cap": 40,   # per-member cap (decisions.md)
        "note": (
            "Taxids verified 2026-06-01 (decisions.md). "
            "Cloacae ss target reduced 200→130 (142 available); "
            "roggenkampii increased 2→70 (1812935 has 75 available). "
            "Hormaechei taxid 158836 preserves original-study taxonomic scope."
        ),
        "ncbi_taxids": [
            {"taxid": "158836", "member": "E. hormaechei",    "sample_target": 265},
            {"taxid": "550",    "member": "E. cloacae",       "sample_target": 130},
            {"taxid": "61645",  "member": "E. asburiae",      "sample_target": 60},
            {"taxid": "208224", "member": "E. kobei",         "sample_target": 40},
            {"taxid": "299767", "member": "E. ludwigii",      "sample_target": 35},
            {"taxid": "1812935","member": "E. roggenkampii",  "sample_target": 70},
        ],
    },
}


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def fetch(url: str, as_json: bool = False):
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "eskape-defence-ml/2.0 (muthuramanvigneshwaran@gmail.com)"},
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
        "term": (
            f"txid{taxid}[Organism:exp] AND "
            f'"complete genome"[Assembly Level] AND latest[filter]'
        ),
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
    Fetch esummary for assembly UIDs; return GCF_ records with release_date.
    Processes in batches of 200.
    """
    records = []
    for i in range(0, len(uids), 200):
        batch = uids[i : i + 200]
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
            if not acc.startswith("GCF_"):
                continue
            records.append({
                "uid":          uid,
                "accession":    acc,
                "biosample":    rec.get("biosampleaccn", ""),
                "organism":     rec.get("speciesname", ""),
                "release_date": rec.get("seqreleasedate", ""),
                "country":      "unknown",
                "year_bin":     _bin_year(rec.get("seqreleasedate", "")),
                "complex_member": "",
            })
        if i % 1000 == 0 and i > 0:
            print(f"    Fetched summaries for {i}/{len(uids)}...", flush=True)
    return records


def enrich_country(records: list[dict], sample_n: int = 500) -> list[dict]:
    """
    Fetch BioSample XML for up to sample_n records to get geo_loc_name / country.
    Best-effort: records without BioSample IDs remain 'unknown'.
    """
    random.seed(RANDOM_SEED)
    to_enrich = (
        random.sample(records, min(sample_n, len(records)))
        if len(records) > sample_n else list(records)
    )
    bs_to_rec = {r["biosample"]: r for r in to_enrich if r.get("biosample")}

    bs_ids = list(bs_to_rec.keys())
    for i in range(0, len(bs_ids), 100):
        batch = bs_ids[i : i + 100]
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
                name = (
                    attr.get("attribute_name") or attr.get("harmonized_name") or ""
                ).lower()
                if name in ("geo_loc_name", "geographic location", "country"):
                    val = (attr.text or "").strip()
                    country = val.split(":")[0].strip() if val else "unknown"
                    bs_to_rec[acc]["country"] = country
                    break
    return records


def _bin_year(date_str: str) -> str:
    for part in date_str.replace("-", " ").replace("/", " ").split():
        if part.isdigit() and len(part) == 4:
            y = int(part)
            if y < 2010:   return "pre-2010"
            if y < 2016:   return "2010-2015"
            if y < 2021:   return "2016-2020"
            return "2021-present"
    return "unknown"


# ── Stratified sampler ────────────────────────────────────────────────────────

def stratified_sample_new(
    new_records: list[dict],
    n_new_needed: int,
    country_cap: int,
    seeded_country_counts: Counter,
) -> list[dict]:
    """
    Sample up to n_new_needed records from new_records, respecting country_cap.

    seeded_country_counts: Counter pre-populated with the original accessions'
    country data. The cap applies to (originals + new) combined, so we only
    have room for (country_cap - seeded_count[c]) more from country c.

    Year-bin stratification: round-robin across bins within the country-capped pool.
    """
    random.seed(RANDOM_SEED)
    random.shuffle(new_records)

    # Apply remaining country budget (cap minus already-spent slots from originals)
    country_counts = Counter(seeded_country_counts)   # copy, not mutate original
    country_capped: list[dict] = []
    for r in new_records:
        c = r.get("country", "unknown") or "unknown"
        if country_counts[c] < country_cap:
            country_capped.append(r)
            country_counts[c] += 1

    if len(country_capped) <= n_new_needed:
        return country_capped

    # Year-bin round-robin within country-capped pool
    by_year: dict = defaultdict(list)
    for r in country_capped:
        by_year[r.get("year_bin", "unknown")].append(r)

    selected: list[dict] = []
    bins = list(by_year.values())
    idx = 0
    while len(selected) < n_new_needed and any(bins):
        bucket = bins[idx % len(bins)]
        if bucket:
            selected.append(bucket.pop(0))
        idx += 1

    return selected[:n_new_needed]


# ── Original accession loading ─────────────────────────────────────────────────

def load_original_metadata(sp_key: str) -> list[dict]:
    """
    Load the original study's per-species accession metadata JSON.
    Returns list of dicts with accession, country, year_bin, complex_member.
    Warns if file is missing (should not happen — all 6 files confirmed present).
    """
    path = os.path.join(ORIGINAL_INTERIM, f"{sp_key}_accession_metadata.json")
    if not os.path.exists(path):
        print(f"  WARNING: original metadata not found at {path}")
        return []
    with open(path) as f:
        return json.load(f)


def seed_country_counter(original_records: list[dict]) -> Counter:
    """Build a Counter of country occurrences from the original records."""
    counts: Counter = Counter()
    for r in original_records:
        c = r.get("country", "unknown") or "unknown"
        counts[c] += 1
    return counts


# ── Per-taxid orchestration ───────────────────────────────────────────────────

def build_new_for_taxid(
    taxid: str,
    member_name: str,
    original_accessions: set[str],
    n_new_needed: int,
    country_cap: int,
    seeded_country_counts: Counter,
    member_original_accs: set[str] | None = None,
) -> list[dict]:
    """
    Fetch NCBI pool for one taxid, strip originals, enrich country,
    sample n_new_needed new accessions respecting remaining country budget.

    member_original_accs: subset of original_accessions belonging to this
    specific taxid member — used only for the "suppressed" warning, so it
    doesn't count cross-member accessions as missing.
    """
    if member_original_accs is None:
        member_original_accs = original_accessions

    print(f"    [{member_name}] fetching UIDs...", flush=True)
    uids = fetch_all_uids(taxid)
    print(f"    [{member_name}] {len(uids)} UIDs found", flush=True)
    if not uids:
        return []

    all_records = fetch_assembly_summary(uids)
    print(f"    [{member_name}] {len(all_records)} GCF_ accessions", flush=True)

    # Split into already-selected (originals) and new pool
    new_records = [r for r in all_records if r["accession"] not in original_accessions]
    overlap     = [r for r in all_records if r["accession"] in original_accessions]

    # Check only this member's originals against NCBI (avoids cross-member false alarms)
    found_in_ncbi = {r["accession"] for r in all_records}
    suppressed = member_original_accs - found_in_ncbi
    if suppressed:
        print(f"    [{member_name}] WARNING: {len(suppressed)} original accessions "
              f"not in current NCBI snapshot (suppressed/superseded): "
              f"{sorted(suppressed)[:5]}{'...' if len(suppressed)>5 else ''}")

    print(f"    [{member_name}] pool: {len(new_records)} new + "
          f"{len(overlap)} overlap ({len(suppressed)} originals not in snapshot)",
          flush=True)

    if n_new_needed <= 0:
        return []

    new_records = enrich_country(new_records, sample_n=len(new_records))
    selected = stratified_sample_new(
        new_records, n_new_needed, country_cap, seeded_country_counts
    )

    for r in selected:
        r["complex_member"] = member_name
    return selected


# ── Per-species orchestration ─────────────────────────────────────────────────

def build_for_species(sp_key: str, sp_cfg: dict) -> tuple[list[str], list[dict]]:
    """
    Build final 600-accession list for one species entry.
    Returns (accession_list, all_metadata_records).

    For EC complex: handles multiple taxids with per-member country caps.
    For single-taxid species: one call to build_new_for_taxid.
    """
    country_cap    = sp_cfg["country_cap"]
    taxid_entries  = sp_cfg["ncbi_taxids"]
    target_n       = sp_cfg["target_n"]

    # Load originals and build seeded counter
    original_meta  = load_original_metadata(sp_key)
    original_accs  = {r["accession"] for r in original_meta}
    seeded_counts  = seed_country_counter(original_meta)

    print(f"  Originals: {len(original_accs)} accessions, "
          f"{len(seeded_counts)} distinct countries (pre-seeding country counters)",
          flush=True)

    all_new_meta: list[dict] = []

    for entry in taxid_entries:
        taxid  = entry["taxid"]
        member = entry.get("member", sp_cfg["name"])
        member_target = entry["sample_target"]   # total per-member target (originals + new)

        # How many originals belong to this member?
        orig_this_member = [
            r for r in original_meta
            if r.get("complex_member", member) == member
               or len(taxid_entries) == 1   # single-taxid species
        ]
        n_orig_member  = len(orig_this_member)
        n_new_needed   = max(0, member_target - n_orig_member)

        # For single-taxid species, seeded_counts covers the whole species.
        # For EC complex, re-compute seeded_counts per member (per-member cap).
        if len(taxid_entries) > 1:
            member_seeded = seed_country_counter(orig_this_member)
        else:
            member_seeded = seeded_counts

        print(f"    [{member}] orig={n_orig_member}, target={member_target}, "
              f"need_new={n_new_needed}", flush=True)

        member_orig_accs = {r["accession"] for r in orig_this_member}
        new_meta = build_new_for_taxid(
            taxid, member, original_accs, n_new_needed, country_cap,
            member_seeded, member_original_accs=member_orig_accs,
        )
        all_new_meta.extend(new_meta)

    # Final assembly: originals (tag complex_member if single-taxid) + new
    for r in original_meta:
        if not r.get("complex_member"):
            r["complex_member"] = taxid_entries[0].get("member", sp_cfg["name"])

    all_meta = original_meta + all_new_meta
    accessions = [r["accession"] for r in original_meta] + \
                 [r["accession"] for r in all_new_meta]

    return accessions, all_meta


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_INTERIM, exist_ok=True)
    os.makedirs("config", exist_ok=True)

    print("Building 600/species superset accession lists — ESKAPE 4× study")
    print("=" * 65)
    print(f"Country cap: 80/species (40/EC member)  |  Seed: {RANDOM_SEED}")
    print(f"API key: {'set' if API_KEY else 'not set (3 req/s)'}")
    print(f"Original project: {ORIGINAL_PROJECT}\n")

    output_config = dict(SPECIES_CONFIG)   # deep copy-ish (we'll add accessions)

    summary_rows = []
    for sp_key, sp_cfg in SPECIES_CONFIG.items():
        print(f"\n{'─'*65}")
        print(f"Species: {sp_cfg['name']}")
        accessions, all_meta = build_for_species(sp_key, sp_cfg)

        output_config[sp_key] = dict(sp_cfg)
        output_config[sp_key]["accessions"] = accessions

        # Save per-species metadata JSON
        meta_path = os.path.join(OUTPUT_INTERIM, f"{sp_key}_accession_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(all_meta, f, indent=2)

        # Per-country distribution for sign-off summary
        country_dist = Counter(
            (r.get("country", "unknown") or "unknown") for r in all_meta
        )
        top_countries = country_dist.most_common(5)

        # EC complex member breakdown
        if len(sp_cfg["ncbi_taxids"]) > 1:
            member_counts = Counter(r.get("complex_member", "?") for r in all_meta)
            member_str = ", ".join(f"{m}={n}" for m, n in sorted(member_counts.items()))
        else:
            member_str = ""

        summary_rows.append({
            "sp_key": sp_key,
            "name": sp_cfg["name"],
            "n_total": len(accessions),
            "n_original": sum(1 for r in all_meta if r["accession"] in
                              {rec["accession"] for rec in load_original_metadata(sp_key)}),
            "n_new": len(all_meta) - sum(1 for r in all_meta if r["accession"] in
                         {rec["accession"] for rec in load_original_metadata(sp_key)}),
            "top_countries": top_countries,
            "member_breakdown": member_str,
        })

        print(f"  Final: {len(accessions)} accessions "
              f"(target {sp_cfg['target_n']})", flush=True)

    # Write species.yaml
    yaml_out = {"species": output_config}
    with open(OUTPUT_CONFIG, "w") as f:
        yaml.dump(yaml_out, f, default_flow_style=False,
                  allow_unicode=True, sort_keys=False)

    # Sign-off summary table
    print("\n\n" + "=" * 65)
    print("SIGN-OFF SUMMARY — verify before proceeding to downloads")
    print("=" * 65)
    print(f"{'Species':<35} {'N':>5} {'Orig':>5} {'New':>5}")
    print("-" * 55)
    grand_total = 0
    for row in summary_rows:
        grand_total += row["n_total"]
        print(f"  {row['name']:<33} {row['n_total']:>5} {row['n_original']:>5} {row['n_new']:>5}")
        if row["member_breakdown"]:
            print(f"    Members: {row['member_breakdown']}")
        print(f"    Top countries: " +
              ", ".join(f"{c}={n}" for c, n in row["top_countries"]))
    print("-" * 55)
    print(f"  {'TOTAL':<33} {grand_total:>5}")
    print(f"\nspecies.yaml written to: {OUTPUT_CONFIG}")
    print("Per-species metadata: data/interim/*_accession_metadata.json")
    print("\nReview counts above. Confirm with user before starting downloads.")


if __name__ == "__main__":
    main()
