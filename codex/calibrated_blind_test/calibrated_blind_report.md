# Calibrated Blind Marking Test

## Inputs Used During Scoring

Only these sources were used during the marking pass:

- `input/ec3040_marking_scheme.pdf`
- `output/human_exemplars.json` containing only the human marks and feedback for:
  - Sadik Ulusow
  - Federica Alvarez-Ossorio Zaldo
  - Alessandro Ungurean
  - Fahima Ali
- `input/maria_submission.pdf`
- `input/alice_submission.docx`

No other marksheet rows were used during scoring. No repo grading code was used.

## Calibrated Blind Marks

### Maria Klenina

| Part | Calibrated blind mark |
| --- | ---: |
| Part 1 | 10.5 / 15 |
| Part 2 Q1 | 2.5 / 5 |
| Part 2 Q2 | 2.5 / 7.5 |
| Part 2 Q3 | 2.5 / 7.5 |
| Part 2 Q4 | 2.5 / 5 |
| Part 2 Q5 | 2.0 / 5 |
| Part 2 Q6 | 4.5 / 10 |
| Part 3 Q1 | 5.5 / 7.5 |
| Part 3 Q2 | 5.0 / 7.5 |
| Part 3 Q3 | 2.0 / 7.5 |
| Part 3 Q4 | 4.5 / 7.5 |
| Part 4 | 9.5 / 15 |
| **Total** | **53.5 / 100** |

Short rationale:

- Part 1 is competent and evidence-based but not especially sharp.
- Part 2 is the weakest section. The formal derivations are incomplete in the visible submission text and the answers stay highly schematic.
- Part 3 has workable predictions and data suggestions, but the regression answer is under-specified.
- Part 4 identifies the basic trade-off from the videos, though the model extensions stay broad.

### Alice Daranijoh

| Part | Calibrated blind mark |
| --- | ---: |
| Part 1 | 12.0 / 15 |
| Part 2 Q1 | 2.5 / 5 |
| Part 2 Q2 | 1.5 / 7.5 |
| Part 2 Q3 | 0.5 / 7.5 |
| Part 2 Q4 | 0.5 / 5 |
| Part 2 Q5 | 2.5 / 5 |
| Part 2 Q6 | 6.0 / 10 |
| Part 3 Q1 | 6.0 / 7.5 |
| Part 3 Q2 | 5.5 / 7.5 |
| Part 3 Q3 | 1.5 / 7.5 |
| Part 3 Q4 | 5.0 / 7.5 |
| Part 4 | 11.5 / 15 |
| **Total** | **55.5 / 100** |

Short rationale:

- Part 1 is strong and clear.
- Part 2 falls sharply because much of the required formal work is absent or only asserted in summary form.
- Part 3 has sensible predictions and measurement discussion, but the regression itself is effectively missing in the extracted submission text.
- Part 4 is the best section after Part 1, with good engagement with omitted mechanisms and possible model extensions.

## Post-Hoc Comparison To Human Marks

The workbook rows for Maria and Alice were opened only after the marks above were fixed.

### Maria Klenina

Human total: `62.0`

| Part | Calibrated blind | Human | Delta |
| --- | ---: | ---: | ---: |
| Part 1 | 10.5 | 11.0 | -0.5 |
| Part 2 Q1 | 2.5 | 3.0 | -0.5 |
| Part 2 Q2 | 2.5 | 3.5 | -1.0 |
| Part 2 Q3 | 2.5 | 6.5 | -4.0 |
| Part 2 Q4 | 2.5 | 2.0 | +0.5 |
| Part 2 Q5 | 2.0 | 3.0 | -1.0 |
| Part 2 Q6 | 4.5 | 6.0 | -1.5 |
| Part 3 Q1 | 5.5 | 4.0 | +1.5 |
| Part 3 Q2 | 5.0 | 5.0 | 0.0 |
| Part 3 Q3 | 2.0 | 5.0 | -3.0 |
| Part 3 Q4 | 4.5 | 4.0 | +0.5 |
| Part 4 | 9.5 | 9.0 | +0.5 |
| **Total** | **53.5** | **62.0** | **-8.5** |

### Alice Daranijoh

Human total: `67.0`

| Part | Calibrated blind | Human | Delta |
| --- | ---: | ---: | ---: |
| Part 1 | 12.0 | 12.0 | 0.0 |
| Part 2 Q1 | 2.5 | 3.0 | -0.5 |
| Part 2 Q2 | 1.5 | 4.0 | -2.5 |
| Part 2 Q3 | 0.5 | 4.0 | -3.5 |
| Part 2 Q4 | 0.5 | 3.0 | -2.5 |
| Part 2 Q5 | 2.5 | 2.0 | +0.5 |
| Part 2 Q6 | 6.0 | 7.0 | -1.0 |
| Part 3 Q1 | 6.0 | 5.0 | +1.0 |
| Part 3 Q2 | 5.5 | 6.0 | -0.5 |
| Part 3 Q3 | 1.5 | 5.0 | -3.5 |
| Part 3 Q4 | 5.0 | 5.0 | 0.0 |
| Part 4 | 11.5 | 11.0 | +0.5 |
| **Total** | **55.5** | **67.0** | **-11.5** |

## Main Lesson

Using the four human exemplars corrected the earlier over-marking problem, but it introduced a new one: I became materially too harsh whenever the extracted script text was sparse, especially in the formal sections.

The largest misses were:

- Maria `Part 2 Q3`: `-4.0`
- Maria `Part 3 Q3`: `-3.0`
- Alice `Part 2 Q3`: `-3.5`
- Alice `Part 3 Q3`: `-3.5`
- Alice `Part 2 Q2`: `-2.5`
- Alice `Part 2 Q4`: `-2.5`

So the next calibration problem is clear: the method now under-credits partially correct formal work when the student’s derivation is brief, compressed, or not fully visible in extraction.
