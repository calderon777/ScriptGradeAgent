# VAT Marking Experiment Notes

Date: `2026-04-17`

This folder tree preserves a sequence of marking experiments on EC3040 submissions.

## Folders

- `codex/marking_test/`
  - First constrained test on Sadik and Federica.
  - Important flaw: the local grade workbook already contained the target students' own marks and feedback, so this was not a clean blind setup.

- `codex/blind_marking_test/`
  - Clean blind rerun using:
    - EC3040 solutions / marking scheme
    - Alessandro submission
    - Fahima submission
  - No local marksheet rows used during scoring.
  - Main result: blind scoring overshot human marks badly, especially on Fahima.

- `codex/calibrated_blind_test/`
  - Blind rerun using:
    - EC3040 marking scheme
    - four human exemplar rows only:
      - Sadik
      - Federica
      - Alessandro
      - Fahima
    - Maria submission
    - Alice submission
  - Main result: exemplar calibration corrected the earlier overmarking, but swung too far toward harsh under-crediting, especially where extraction damaged or compressed formal work.

## Main Experimental Lessons

1. Extraction fidelity is a first-order bottleneck.
   - Formal sections were the least stable when PDF or DOCX extraction lost equations, spacing, or structured derivations.
   - The largest misses against the human marker appeared in Part 2 and parts of Part 3, where notation and derivation visibility matter most.

2. Calibration moves the error, not just the level.
   - Scheme-only blind marking tended to overmark polished answers.
   - Scheme plus human exemplars reduced overmarking but increased under-crediting for brief or partially extracted formal answers.

3. Human-style partial credit is the harder target.
   - The remaining problem is not only "what is correct in principle" but "what deserves partial credit under the human standard when the answer is incomplete, compressed, or not fully visible in extraction."

## Key Reports

- `codex/marking_test/report.md`
- `codex/blind_marking_test/blind_report.md`
- `codex/calibrated_blind_test/calibrated_blind_report.md`

## Competition Note

A side-by-side comparison against a `gpt-5.4` sub-agent on the calibrated blind task produced the same totals as Codex for Maria and Alice, though not always through identical part-level marks.
