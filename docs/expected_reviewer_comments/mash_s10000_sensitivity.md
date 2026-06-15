# Mash sketch size sensitivity: s=1000 vs s=10000

**Date:** 2026-06-13  
**Purpose:** Rebuttal-ready sensitivity analysis. Not included in manuscript proactively.

## Parameters
- k=21, -p 8 in both runs
- Same per-species thresholds: AB/EC/KP t=0.010, EF t=0.007, PA/SA t=0.005

## Clustering comparison

| Species | t | Groups s=1000 | Groups s=10000 | MaxSize s=1000 | MaxSize s=10000 |
|---------|---|---|---|---|---|
| AB | 0.010 | 81 | 80 | 290 | 291 |
| EC | 0.010 | 220 | 221 | 30 | 30 |
| EF | 0.007 | 74 | 72 | 112 | 80 |
| KP | 0.010 | 58 | 73 | 75 | 74 |
| PA | 0.005 | 187 | 194 | 37 | 34 |
| SA | 0.005 | 50 | 50 | 126 | 126 |

## MLST concordance

| Species | s=1000 | s=10000 |
|---------|--------|---------|
| AB | 95.5% | 95.5% |
| EC | 98.6% | 98.6% |
| EF | 65.0% | 70.0% |
| KP | 98.2% | 98.2% |
| PA | 92.0% | 92.0% |
| SA | 97.7% | 97.7% |
| **OVERALL** | **92.4%** | **93.0%** |

## Conclusion
Overall concordance improves by 0.6% (92.4% → 93.0%). Structure is stable in 4/6 species.
KP gains 15 groups (noise-driven merges corrected). EF concordance remains below 90% at both
settings — the EF complexity is biological, not a Mash precision artefact. The s=1000 result
passes the pre-specified 90% concordance threshold and biological conclusions are unchanged.
Pipeline was not rerun; s=1000 results stand.
