# Benchmark Baseline 2026-04-16

Frozen from `output/human_benchmark_ec3040/summary.json` on 2026-04-16.

## Aggregate Baseline

- `sample_count`: `4`
- `mean_abs_total_delta`: `9.38`
- `mean_abs_part_delta`: `1.47`
- `mean_latency_total`: `373.7s`
- PDF `mean_abs_total_delta`: `13.0`
- DOCX `mean_abs_total_delta`: `5.75`

## Fixed Benchmark Set

- Sadik Ulusow (`.pdf`)
- Federica Alvarez-Ossorio Zaldo (`.pdf`)
- Aya Kalada (`.docx`)
- Apisan Rasiah (`.docx`)

## Targeted Outliers

- Sadik PDF:
  - `Part 2 Q2` human `6.0` vs model `4.5`
  - `Part 2 Q3` human `6.5` vs model `1.5`
- Federica PDF:
  - total absolute delta `22.5`
  - `Part 2 Q3` human `3.5` vs model `0.0`
  - `Part 4` human `9.0` vs model `5.0`
- Aya DOCX:
  - total absolute delta `6.0`
  - `Part 2 Q2` human `4.0` vs model `5.0`
  - `Part 2 Q6` human `7.0` vs model `5.0`
- Apisan DOCX:
  - total absolute delta `5.5`
  - `Part 2 Q2` human `0.0` vs model `5.0`
  - `Part 2 Q3` human `6.5` vs model `1.5`

## Tracker Opened On 2026-04-16

Milestone tracker issues:

- `#1` `M1 Trustworthy Payloads`
- `#2` `M2 Scoring Calibration`
- `#3` `M3 Latency And Moderation Discipline`
- `#4` `M4 Benchmark Productization`
- `#5` `M5 Structure Mismatch Visibility`
- `#6` `M6 Architecture Hardening`
- `#7` `M7 Data Hygiene`

Opened `P0` issues:

- `#8` Stop aggressive truncation of exact question text for deterministic units
- `#9` Detect damaged section wording and fall back to a larger local chunk
- `#10` Export before/after payload inspections for PDF outlier sections
- `#11` Tighten derivation scoring around inequality validity and family-by-family deviation logic
- `#12` Add missing-answer and low-evidence benchmark cases for zero or near-zero credit
- `#13` Re-run the fixed 4-script benchmark after each scoring change

## Artifact Anchors

- Payload and structure damage baseline: `output/assessment_structure_bakeoff/assessment_structure_bakeoff.json`
- Frozen benchmark baseline: `output/human_benchmark_ec3040/summary.json`
- DOCX comparison reference: `output/docx_competitor_smoke_10/summary.json`
