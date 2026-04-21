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
