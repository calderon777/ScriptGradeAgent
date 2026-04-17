# Codex Marking Test

## Scope

This folder was set up to test marking using only:

- `input/EC3040_deanonymised_grades.xlsx`
- `input/sadik_submission.pdf`
- `input/federica_submission.pdf`

No repo grading code was used. No internet was used.

## Important Caveat

The workbook is not a pure rubric. It is a deanonymised gradebook that already contains the final marks and part-by-part feedback for all students, including Sadik and Federica. That means the source material itself contains the answer key for the two target students.

To keep the exercise meaningful, I treated the workbook primarily as:

- the local marking-sheet structure,
- evidence of part weights and marking style,
- a comparison target after reading the submissions.

So this is a constrained local replication exercise, not a clean blind mark.

## My Estimated Marks

### Federica Alvarez-Ossorio Zaldo

| Part | My mark | Workbook mark | Delta |
| --- | ---: | ---: | ---: |
| Part 1 | 9.0 | 9.0 | 0.0 |
| Part 2 Q1 | 2.5 | 2.0 | 0.5 |
| Part 2 Q2 | 2.5 | 3.5 | -1.0 |
| Part 2 Q3 | 2.5 | 3.5 | -1.0 |
| Part 2 Q4 | 3.5 | 3.0 | 0.5 |
| Part 2 Q5 | 3.0 | 3.0 | 0.0 |
| Part 2 Q6 | 6.0 | 6.0 | 0.0 |
| Part 3 Q1 | 5.0 | 5.0 | 0.0 |
| Part 3 Q2 | 5.0 | 5.0 | 0.0 |
| Part 3 Q3 | 4.5 | 4.0 | 0.5 |
| Part 3 Q4 | 5.0 | 5.0 | 0.0 |
| Part 4 | 9.5 | 9.0 | 0.5 |
| **Total** | **58.0** | **58.0** | **0.0** |

Short rationale:

- Part 1 is competent and structured, but too general and imprecise in places.
- Part 2 is the main weakness. Q2 and Q3 are hard to credit strongly from the extracted PDF because much of the mathematical content appears missing or image-based.
- Part 3 is better: the empirical logic is clearer than the formal theory work.
- Part 4 is solid but still broad rather than sharply evidenced.

### Sadik Ulusow

| Part | My mark | Workbook mark | Delta |
| --- | ---: | ---: | ---: |
| Part 1 | 9.5 | 9.0 | 0.5 |
| Part 2 Q1 | 4.0 | 4.0 | 0.0 |
| Part 2 Q2 | 5.5 | 6.0 | -0.5 |
| Part 2 Q3 | 6.0 | 6.5 | -0.5 |
| Part 2 Q4 | 4.0 | 4.0 | 0.0 |
| Part 2 Q5 | 3.0 | 2.5 | 0.5 |
| Part 2 Q6 | 6.5 | 6.0 | 0.5 |
| Part 3 Q1 | 5.0 | 5.0 | 0.0 |
| Part 3 Q2 | 3.5 | 3.0 | 0.5 |
| Part 3 Q3 | 3.5 | 3.0 | 0.5 |
| Part 3 Q4 | 4.5 | 4.0 | 0.5 |
| Part 4 | 10.0 | 11.0 | -1.0 |
| **Total** | **65.0** | **64.0** | **1.0** |

Short rationale:

- Stronger than Federica overall, especially in Part 2 where there is more visible derivation and interpretation.
- Still loses marks for notation clarity, precision, and some over-general empirical claims in Part 3.
- Part 4 is good and well connected to the model, but I held it slightly below the workbook because the extension discussion stays fairly high level.

## Result

Absolute total delta:

- Federica: `0.0`
- Sadik: `1.0`

Mean absolute total delta across the two scripts: `0.5`

## Files Produced

- `output/non_target_rows.json`
- `output/target_rows.json`
- `output/federica_submission.txt`
- `output/sadik_submission.txt`

