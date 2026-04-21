# Decision Log

Deviations from the pre-analysis plan and significant design choices made
during the project. Each entry: date, decision, alternatives considered,
reason for choice.

---

## 2026-04-21 — Phase 0/1

**Python version:** 3.11 (not system 3.9.6)
Reason: scikit-learn ≥1.4 and shap ≥0.44 have dropped 3.9 support.

**A. baumannii data:** Fresh NCBI download (~150 genomes) for training.
Published 132 genomes held out for validation.
Reason: independent benchmark; contig-level IC2 assemblies evaluated separately.

**Q2 ARG burden label:** Tertile split (top vs bottom 33%).
Alternative considered: median split (50/50), continuous regression.
Reason: clean class separation maximises signal detectability for primary test.
Middle-tertile genomes retained for Q1 and Q3.

**Q3 clustering:** Species labels hidden during clustering, applied post-hoc.
Reason: prevents species-identity bias from contaminating archetype discovery.

---

## 2026-04-22 — Phase 2 (sampling design)

**E. cloacae taxonomy expansion:** Expanded from taxid 550 (E. cloacae ss, 142
genomes) to the full Enterobacter cloacae complex (6 genomospecies, 980 total
complete genomes). Genomospecies recorded as a metadata column.
Reason: ESKAPE "E" refers to the complex, not E. cloacae sensu stricto. Using
taxid 550 alone would have excluded E. hormaechei (the clinically dominant
member, 576 complete genomes) entirely.
Per-member caps: E. hormaechei 60, E. cloacae ss 50, E. asburiae 15,
E. kobei 10, E. ludwigii 10, E. roggenkampii 5 (total 150).

**Sampling strategy — all species:**
- Country cap: max 30 genomes per country per species.
- Stratification axes: country, collection year (4 bins), isolation source (best effort).
- Clinical/non-clinical ratio: not enforced — reflects NCBI availability.
- Isolation source: captured raw, classified post-download into 6 categories.
- Random seed: 42 throughout.
- E. cloacae complex: country cap applied per member, not across the combined pool.
