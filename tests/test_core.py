import json
from pathlib import Path
from shutil import rmtree
from uuid import uuid4
import unittest
from unittest.mock import Mock, patch

from marking_pipeline.core import (
    build_missing_part_analysis,
    _run_part_analysis_with_retry,
    AssessmentMap,
    AssessmentUnit,
    build_marking_context_from_bundle,
    build_assessment_map,
    build_part_messages,
    build_rubric_matrix_markdown,
    build_rubric_verification_messages,
    build_single_assessment_bundle,
    build_submission_diagnostics,
    extract_expected_parts_from_context,
    extract_expected_subparts_from_context,
    extract_structure_guidance,
    calibrate_marks_across_students,
    call_ollama,
    compute_total_mark_from_part_scores,
    discover_assessment_bundles,
    infer_max_mark_from_texts,
    normalize_detected_parts,
    normalize_marking_result,
    normalize_verified_rubric,
    moderate_linked_part_analyses,
    normalize_moderated_part_scores,
    normalize_verification_result,
    parse_json_object,
    prepare_marking_context,
    reconcile_detected_parts,
    regrade_marking_result,
    refine_submission_granularity,
    segment_submission_parts,
    SubmissionPart,
    verify_assessment_rubrics,
    verify_marking_result,
)
from marking_pipeline import cache as cache_module
from marking_pipeline.workflow import suggest_marking_scale

LONG_FEEDBACK = (
    "Question 1 is addressed with clear explanation and relevant evidence, while Question 2 is also handled with a sensible structure and "
    "a credible discussion of the main policy mechanism. The response shows a strong grasp of the topic, uses the supplied material in a "
    "reasonably disciplined way, and presents a coherent line of argument throughout the script. There are still some weaknesses in depth, "
    "precision, and completeness, especially when the answer moves from theory to empirical interpretation, but the overall standard remains "
    "high and the script demonstrates real understanding across the assessment. The best sections explain the economic logic clearly, connect claims "
    "to the evidence in the answer, and maintain focus on the actual question asked. The weaker sections need more detail, more precise terminology, "
    "and a fuller explanation of why the chosen approach is justified, but the script still shows meaningful understanding and sustained engagement "
    "with the material across the full submission."
)


class InferMaxMarkTests(unittest.TestCase):
    def test_infers_max_mark_from_rubric(self) -> None:
        rubric = "Assessment: Short answer exam. Maximum mark: 20."
        self.assertEqual(infer_max_mark_from_texts(rubric), 20)

    def test_rejects_conflicting_max_marks(self) -> None:
        with self.assertRaises(ValueError):
            infer_max_mark_from_texts("Maximum mark: 20", "Maximum mark: 85")


class ParseJsonTests(unittest.TestCase):
    def test_parses_json_inside_wrapping_text(self) -> None:
        content = '```json\n{"total_mark": 17, "max_mark": 20, "overall_feedback": "Clear answer with one weakness."}\n```'
        data = parse_json_object(content)
        self.assertEqual(data["total_mark"], 17)

    def test_parses_json_with_braces_inside_string(self) -> None:
        content = 'prefix {"total_mark": 18, "max_mark": 20, "overall_feedback": "Mentions {elasticity} correctly."} suffix'
        data = parse_json_object(content)
        self.assertEqual(data["max_mark"], 20)


class NormalizationTests(unittest.TestCase):
    def test_rescales_percentage_when_model_returns_100_scale(self) -> None:
        result = normalize_marking_result(
            {
                "total_mark": 95,
                "max_mark": 100,
                "overall_feedback": LONG_FEEDBACK,
            },
            expected_max_mark=20,
        )
        self.assertEqual(result["total_mark"], 19)
        self.assertEqual(result["max_mark"], 20)

    def test_rejects_out_of_range_marks(self) -> None:
        with self.assertRaises(ValueError):
            normalize_marking_result(
                {
                    "total_mark": 30,
                    "max_mark": 20,
                    "overall_feedback": LONG_FEEDBACK,
                },
                expected_max_mark=20,
            )

    def test_rejects_full_marks_with_major_weaknesses(self) -> None:
        diagnostics = build_submission_diagnostics("Question 1\nA full answer.\nQuestion 2\nAnother full answer.")
        parts = normalize_detected_parts(
            {"sections": [{"label": "Question 1", "focus_hint": "First answer", "anchor_text": "Question 1"}, {"label": "Question 2", "focus_hint": "Second answer", "anchor_text": "Question 2"}]}
        )
        with self.assertRaises(ValueError):
            normalize_marking_result(
                {
                    "total_mark": 20,
                    "max_mark": 20,
                    "overall_feedback": (
                        "Question 1 is addressed and Question 2 is addressed, but there is an incorrect calculation and the empirical model is not properly "
                        "specified. The response has some strengths, but it also misses key details and requires significant revision to reach a high standard. "
                        "The answer is clearly written and engages with the topic, yet the central reasoning remains incomplete, several claims are not justified, "
                        "and the evidence is too thin to support a top-band mark even though parts of the explanation are promising."
                    ),
                    "covered_parts": ["Question 1", "Question 2"],
                    "strengths": ["clear writing", "engages with the topic"],
                    "weaknesses": ["incorrect calculation", "model not properly specified"],
                },
                expected_max_mark=20,
                diagnostics=diagnostics,
                parts=parts,
                part_analyses=[],
            )

    def test_uses_math_total_and_keeps_ai_total_for_audit(self) -> None:
        parts = [
            SubmissionPart(label="Part 1", max_mark=20.0),
            SubmissionPart(label="Part 2", max_mark=30.0),
        ]
        result = normalize_marking_result(
            {
                "total_mark": 20.0,
                "max_mark": 50.0,
                "overall_feedback": LONG_FEEDBACK,
                "covered_parts": ["Part 1", "Part 2"],
                "strengths": ["clear explanation", "good structure"],
                "weaknesses": ["limited depth", "some missing precision"],
            },
            expected_max_mark=50.0,
            parts=parts,
            part_analyses=[
                {"provisional_score": 16.0, "provisional_score_0_to_100": None},
                {"provisional_score": 24.0, "provisional_score_0_to_100": None},
            ],
        )
        self.assertEqual(result["total_mark"], 40.0)
        self.assertEqual(result["ai_total_mark"], 20.0)
        self.assertEqual(result["math_total_mark"], 40.0)
        self.assertEqual(result["ai_math_mark_delta"], -20.0)
        self.assertIn("math_total_used", result["validation_notes"])
        self.assertIn("ai_total_disagreement=20.00", result["validation_notes"])

    def test_computes_total_mark_from_part_scores(self) -> None:
        parts = [
            SubmissionPart(label="Part 1", max_mark=20.0),
            SubmissionPart(label="Part 2", max_mark=30.0),
        ]
        total = compute_total_mark_from_part_scores(
            expected_max_mark=50.0,
            parts=parts,
            part_analyses=[
                {"provisional_score": 16.0, "provisional_score_0_to_100": None},
                {"provisional_score": 24.0, "provisional_score_0_to_100": None},
            ],
        )
        self.assertEqual(total, 40.0)

    def test_does_not_rescale_total_when_part_maxima_do_not_cover_expected_total(self) -> None:
        parts = [SubmissionPart(label="Part 1", max_mark=20.0)]
        total = compute_total_mark_from_part_scores(
            expected_max_mark=50.0,
            parts=parts,
            part_analyses=[
                {"provisional_score": 16.0, "provisional_score_0_to_100": None},
            ],
        )
        self.assertEqual(total, 16.0)

    def test_accepts_feedback_only_final_synthesis_when_math_total_is_available(self) -> None:
        parts = [
            SubmissionPart(label="Part 1", max_mark=20.0),
            SubmissionPart(label="Part 2", max_mark=30.0),
        ]
        result = normalize_marking_result(
            {
                "overall_feedback": LONG_FEEDBACK,
                "covered_parts": ["Part 1", "Part 2"],
                "strengths": ["clear explanation", "good structure"],
                "weaknesses": ["limited depth", "some missing precision"],
            },
            expected_max_mark=50.0,
            parts=parts,
            part_analyses=[
                {"provisional_score": 16.0, "provisional_score_0_to_100": None},
                {"provisional_score": 24.0, "provisional_score_0_to_100": None},
            ],
        )
        self.assertEqual(result["total_mark"], 40.0)
        self.assertIsNone(result["ai_total_mark"])
        self.assertEqual(result["math_total_mark"], 40.0)

    def test_accepts_structured_feedback_object_when_math_total_is_available(self) -> None:
        parts = [
            SubmissionPart(label="Part 1", max_mark=20.0),
            SubmissionPart(label="Part 2", max_mark=30.0),
        ]
        result = normalize_marking_result(
            {
                "overall_feedback": {
                    "length": 180,
                    "content": [
                        {
                            "section_label": "Part 1",
                            "analysis": (
                                "The answer covers the main theoretical mechanism with some precision, "
                                "explains the core logic clearly, and gives enough detail to support a "
                                "reasoned judgement about quality and coverage. It also keeps the discussion "
                                "close to the task, uses relevant terminology, and shows enough development "
                                "to support a meaningful evaluation of both strengths and weaknesses."
                            ),
                            "feedback": (
                                "This section is coherent but misses some finer detail, stronger justification, "
                                "and closer engagement with a few of the more demanding elements of the task. "
                                "A stronger answer would sustain the same clarity while tightening the link "
                                "between the argument, the evidence, and the marking criteria."
                            ),
                        },
                        {
                            "section_label": "Part 2",
                            "analysis": (
                                "The empirical discussion is competent and mostly well supported, "
                                "with a coherent explanation of evidence, limits, and interpretation "
                                "that stays close to the assessment task and the student's argument. "
                                "It offers enough substance to comment on method, interpretation, and "
                                "the overall balance between confidence and caution in the conclusions."
                            ),
                            "feedback": (
                                "The section would benefit from tighter identification analysis, clearer "
                                "justification of assumptions, and more precise use of the underlying framework. "
                                "The current version is sensible, but it still leaves room for sharper explanation, "
                                "better prioritization of key points, and more disciplined use of evidence."
                            ),
                        },
                    ],
                },
                "covered_parts": ["Part 1", "Part 2"],
                "strengths": ["clear explanation", "good structure"],
                "weaknesses": ["limited depth", "some missing precision"],
            },
            expected_max_mark=50.0,
            parts=parts,
            part_analyses=[
                {"provisional_score": 16.0, "provisional_score_0_to_100": None},
                {"provisional_score": 24.0, "provisional_score_0_to_100": None},
            ],
        )
        self.assertEqual(result["total_mark"], 40.0)
        self.assertIn("Part 1", result["overall_feedback"])
        self.assertIn("Part 2", result["overall_feedback"])

    def test_rejects_feedback_only_final_synthesis_when_math_total_is_unavailable(self) -> None:
        with self.assertRaises(ValueError):
            normalize_marking_result(
                {
                    "overall_feedback": LONG_FEEDBACK,
                    "covered_parts": ["Part 1", "Part 2"],
                    "strengths": ["clear explanation", "good structure"],
                    "weaknesses": ["limited depth", "some missing precision"],
                },
                expected_max_mark=50.0,
                parts=[SubmissionPart(label="Part 1"), SubmissionPart(label="Part 2")],
                part_analyses=[
                    {"provisional_score": None, "provisional_score_0_to_100": 80.0},
                    {"provisional_score": None, "provisional_score_0_to_100": 70.0},
                ],
            )

    def test_computes_single_part_total_from_expected_max_mark(self) -> None:
        total = compute_total_mark_from_part_scores(
            expected_max_mark=85.0,
            parts=[SubmissionPart(label="Whole Submission")],
            part_analyses=[
                {"provisional_score": None, "provisional_score_0_to_100": 80.0},
            ],
        )
        self.assertEqual(total, 68.0)

    def test_normalizes_moderated_part_scores(self) -> None:
        parts = [
            SubmissionPart(label="Part 1", max_mark=20.0),
            SubmissionPart(label="Part 2", max_mark=30.0),
        ]
        moderated = normalize_moderated_part_scores(
            {
                "adjusted_sections": [
                    {"section_label": "Part 1", "adjusted_provisional_score": 15.0, "adjusted_provisional_score_0_to_100": None, "rationale": "Slightly harsh relative to Part 2."},
                    {"section_label": "Part 2", "adjusted_provisional_score": 23.0, "adjusted_provisional_score_0_to_100": None, "rationale": "Slightly generous relative to Part 1."},
                ]
            },
            parts=parts,
            part_analyses=[
                {"section_label": "Part 1", "provisional_score": 14.0, "provisional_score_0_to_100": None, "strengths": [], "weaknesses": [], "evidence": [], "coverage_comment": ""},
                {"section_label": "Part 2", "provisional_score": 24.0, "provisional_score_0_to_100": None, "strengths": [], "weaknesses": [], "evidence": [], "coverage_comment": ""},
            ],
        )
        self.assertEqual(moderated[0]["provisional_score"], 15.0)
        self.assertEqual(moderated[0]["moderation_delta"], 1.0)
        self.assertEqual(moderated[1]["provisional_score"], 23.0)
        self.assertEqual(moderated[1]["moderation_delta"], -1.0)

    def test_moderation_allows_null_or_missing_adjustments_as_unchanged(self) -> None:
        parts = [
            SubmissionPart(label="Part 1", max_mark=20.0),
            SubmissionPart(label="Part 2", max_mark=30.0),
        ]
        moderated = normalize_moderated_part_scores(
            {
                "adjusted_sections": [
                    {"section_label": "Part 1", "adjusted_provisional_score": None, "adjusted_provisional_score_0_to_100": None, "rationale": "Keep unchanged."}
                ]
            },
            parts=parts,
            part_analyses=[
                {"section_label": "Part 1", "provisional_score": 14.0, "provisional_score_0_to_100": None, "strengths": [], "weaknesses": [], "evidence": [], "coverage_comment": ""},
                {"section_label": "Part 2", "provisional_score": 24.0, "provisional_score_0_to_100": None, "strengths": [], "weaknesses": [], "evidence": [], "coverage_comment": ""},
            ],
        )
        self.assertEqual(moderated[0]["provisional_score"], 14.0)
        self.assertEqual(moderated[0]["moderation_delta"], 0.0)
        self.assertEqual(moderated[1]["provisional_score"], 24.0)
        self.assertEqual(moderated[1]["moderation_delta"], 0.0)

    @patch("marking_pipeline.core.moderate_part_analyses_across_submission")
    def test_moderates_only_linked_groups(self, mock_moderate: Mock) -> None:
        mock_moderate.side_effect = lambda **kwargs: kwargs["part_analyses"]
        assessment_map = type("Map", (), {
            "units": (
                type("Unit", (), {"label": "Part 1", "dependency_group": ""})(),
                type("Unit", (), {"label": "Part 2 Q1", "dependency_group": "part 2"})(),
                type("Unit", (), {"label": "Part 2 Q2", "dependency_group": "part 2"})(),
            )
        })()
        parts = [
            SubmissionPart(label="Part 1", max_mark=15.0),
            SubmissionPart(label="Part 2 Q1", max_mark=5.0),
            SubmissionPart(label="Part 2 Q2", max_mark=7.5),
        ]
        analyses = [
            {"section_label": "Part 1", "provisional_score": 10.0, "provisional_score_0_to_100": None},
            {"section_label": "Part 2 Q1", "provisional_score": 4.0, "provisional_score_0_to_100": None},
            {"section_label": "Part 2 Q2", "provisional_score": 6.0, "provisional_score_0_to_100": None},
        ]
        moderate_linked_part_analyses(
            script_text="Student answer",
            context=prepare_marking_context("Maximum mark: 20", "", "", "", ""),
            filename="student.docx",
            parts=parts,
            part_analyses=analyses,
            assessment_map=assessment_map,
            model_name="qwen2:7b",
        )
        self.assertEqual(mock_moderate.call_count, 1)
        call_kwargs = mock_moderate.call_args.kwargs
        self.assertEqual([part.label for part in call_kwargs["parts"]], ["Part 2 Q1", "Part 2 Q2"])

class SubmissionStructureTests(unittest.TestCase):
    def test_normalizes_detected_parts(self) -> None:
        parts = normalize_detected_parts(
            {"sections": [{"label": "Question 1", "focus_hint": "First answer", "anchor_text": "Question 1"}, {"label": "Question 2", "focus_hint": "Second answer", "anchor_text": "Question 2"}]}
        )
        self.assertEqual([part.label for part in parts], ["Question 1", "Question 2"])
        self.assertEqual(parts[0].anchor_text, "Question 1")

    def test_builds_submission_diagnostics(self) -> None:
        script = "Question 1\n" + ("word " * 80) + "\nQuestion 2\n" + ("word " * 90)
        parts = normalize_detected_parts(
            {"sections": [{"label": "Question 1", "focus_hint": "First answer", "anchor_text": "Question 1"}, {"label": "Question 2", "focus_hint": "Second answer", "anchor_text": "Question 2"}]}
        )
        diagnostics = build_submission_diagnostics(script, parts)
        self.assertEqual(diagnostics.detected_part_count, 2)
        self.assertFalse(diagnostics.low_text)

    def test_segments_submission_parts_by_anchor_text(self) -> None:
        script = "Question 1\nAnswer one.\nQuestion 2\nAnswer two."
        parts = normalize_detected_parts(
            {"sections": [{"label": "Question 1", "focus_hint": "First answer", "anchor_text": "Question 1"}, {"label": "Question 2", "focus_hint": "Second answer", "anchor_text": "Question 2"}]}
        )
        segmented = segment_submission_parts(script, parts)
        self.assertIn("Answer one.", segmented[0].section_text)
        self.assertIn("Answer two.", segmented[1].section_text)

    def test_extracts_only_useful_structure_guidance(self) -> None:
        context = prepare_marking_context(
            rubric_text="Question 1 (25 marks)\nStrong analysis expected.\nGeneral comment about presentation.",
            brief_text="Part 2: Empirical strategy worth 35% of the mark.\nRead the dataset appendix carefully.",
            marking_scheme_text="Section 3 out of 40 marks.\nFeedback should be specific.",
            graded_sample_text="",
            other_context_text="",
        )
        guidance = extract_structure_guidance(context)
        self.assertIn("Question 1 (25 marks)", guidance)
        self.assertIn("Part 2: Empirical strategy worth 35% of the mark.", guidance)
        self.assertIn("Section 3 out of 40 marks.", guidance)
        self.assertNotIn("General comment about presentation.", guidance)
        self.assertNotIn("Read the dataset appendix carefully.", guidance)

    def test_expected_parts_ignore_inline_question_references(self) -> None:
        context = prepare_marking_context(
            rubric_text="Part 1 (15 marks)\nPart 2 (40 marks)\nPart 3 (30 marks)\nPart 4 (15 marks)\nout of 85",
            brief_text="",
            marking_scheme_text=(
                "conditions in question 2 are satisfied, then\n"
                "However, as shown in question 4, an increase in VAT can switch the equilibrium.\n"
            ),
            graded_sample_text="",
            other_context_text="",
        )
        expected = extract_expected_parts_from_context(context)
        self.assertEqual([part.label for part in expected], ["Part 1", "Part 2", "Part 3", "Part 4"])

    def test_extracts_expected_subparts_from_context(self) -> None:
        context = prepare_marking_context(
            rubric_text=(
                "Part 2 (40 marks)\n"
                "1. [5 marks] First item\n"
                "2. [7.5 marks] Second item\n"
                "3. [7.5 marks] Third item\n"
                "Part 3 (30 marks)\n"
                "1. [7.5 marks] First item\n"
                "2. [7.5 marks] Second item\n"
                "out of 100\n"
            ),
            brief_text="",
            marking_scheme_text="",
            graded_sample_text="",
            other_context_text="",
        )
        child_map = extract_expected_subparts_from_context(context)
        self.assertEqual(
            [part.label for part in child_map["part 2"]],
            ["Part 2 Q1", "Part 2 Q2", "Part 2 Q3"],
        )
        self.assertEqual(child_map["part 2"][0].max_mark, 5.0)

    def test_builds_assessment_map_with_subparts_and_modes(self) -> None:
        context = prepare_marking_context(
            rubric_text=(
                "Part 1 (15 marks)\n"
                "Generate a 300-word summary of the VAT policy and discuss likely costs and benefits.\n"
                "Part 2 (40 marks)\n"
                "1. [5 marks] Write out the payoff function.\n"
                "2. [7.5 marks] Derive conditions.\n"
                "3. [7.5 marks] Derive conditions for scenario 2.\n"
                "4. [5 marks] Explain how changing each parameter affects the scenario.\n"
                "Part 3 (30 marks)\n"
                "1. [7.5 marks] List predictions.\n"
                "2. [7.5 marks] Explain how each variable could be measured.\n"
                "3. [7.5 marks] Write down the regression equation.\n"
                "4. [7.5 marks] Explain why the regression might suffer from endogeneity issues using the theoretical model from Part 2.\n"
                "Part 4 (15 marks)\n"
                "Discuss carefully, in your own words, whether you think the introduction of VAT is a good idea.\n"
                "out of 100\n"
            ),
            brief_text="",
            marking_scheme_text="",
            graded_sample_text="",
            other_context_text="",
        )
        assessment_map = build_assessment_map(context)
        labels = [unit.label for unit in assessment_map.units]
        self.assertIn("Part 2 Q1", labels)
        self.assertIn("Part 3 Q4", labels)
        unit_by_label = {unit.label: unit for unit in assessment_map.units}
        self.assertEqual(unit_by_label["Part 2 Q2"].grading_mode, "deterministic")
        self.assertEqual(unit_by_label["Part 4"].grading_mode, "analytical")
        self.assertEqual(unit_by_label["Part 3 Q4"].dependency_group, "part 3")
        self.assertIn("UK-style quality bands", unit_by_label["Part 4"].rubric_text)
        self.assertIn("Task focus:", unit_by_label["Part 4"].rubric_text)

    def test_rubric_generation_avoids_raw_task_prose_as_criterion(self) -> None:
        context = prepare_marking_context(
            rubric_text=(
                "Part 1 (15 marks)\n"
                "We start our analysis of the VAT on private schools with a broad overview of the costs and benefits of the policy using concepts we have learned in the course.\n"
                "out of 15\n"
            ),
            brief_text="",
            marking_scheme_text="",
            graded_sample_text="",
            other_context_text="",
        )
        assessment_map = build_assessment_map(context)
        rubric_text = assessment_map.units[0].rubric_text
        self.assertNotIn("We start our analysis", rubric_text)
        self.assertIn("Evaluate whether the response directly addresses", rubric_text)

    def test_builds_rubric_matrix_markdown(self) -> None:
        context = prepare_marking_context(
            rubric_text=(
                "Part 1 (15 marks)\n"
                "Discuss carefully, in your own words, whether the policy is a good idea.\n"
                "Part 2 (5 marks)\n"
                "Write down the regression equation.\n"
                "out of 20\n"
            ),
            brief_text="",
            marking_scheme_text="",
            graded_sample_text="",
            other_context_text="",
        )
        rubric_md = build_rubric_matrix_markdown(context, assessment_name="EC3040")
        self.assertIn("# EC3040 Rubric Matrix", rubric_md)
        self.assertIn("## Part 1", rubric_md)
        self.assertIn("#### Criteria Sentences", rubric_md)
        self.assertIn("Classification question:", rubric_md)
        self.assertIn("Ranking rule:", rubric_md)
        self.assertIn("| Band | Descriptor | Within-band placement |", rubric_md)
        self.assertIn("First", rubric_md)
        self.assertIn("Excellent", rubric_md)

    def test_builds_rubric_verification_messages(self) -> None:
        context = prepare_marking_context(
            rubric_text="Part 4 (15 marks)\nDiscuss carefully whether the policy is a good idea.\nout of 15",
            brief_text="",
            marking_scheme_text="Use the UK classification language where appropriate.",
            graded_sample_text="",
            other_context_text="",
        )
        unit = AssessmentUnit(
            label="Part 4",
            max_mark=15.0,
            grading_mode="analytical",
            rubric_text="Part 4: use holistic judgement.",
        )
        messages = build_rubric_verification_messages(unit, context)
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn('"rubric_text"', messages[0]["content"])
        self.assertIn("UK classification language", messages[1]["content"])
        self.assertIn("Part 4", messages[1]["content"])

    def test_part_messages_use_local_support_not_full_context_dump(self) -> None:
        context = prepare_marking_context(
            rubric_text="Part 1 (15 marks)\nDiscuss whether the policy is a good idea.\nout of 15",
            brief_text="Use the IFS report and lecture material.",
            marking_scheme_text="Part 1 is worth 15 marks.",
            graded_sample_text="This should not be copied into the part prompt.",
            other_context_text="Additional support text.",
        )
        part = SubmissionPart(
            label="Part 1",
            focus_hint="Evaluate the policy.",
            section_text="Student section text.",
            max_mark=15.0,
            marking_guidance="Evaluate whether the response directly addresses the required evaluative discussion.",
        )
        messages = build_part_messages(part, context, "student.docx")
        user = messages[1]["content"]
        self.assertIn("STRUCTURE AND MARKS HINTS:", user)
        self.assertIn("LOCAL ASSIGNMENT BRIEF SUPPORT:", user)
        self.assertIn("RELEVANT MARKING-SCHEME EXCERPT:", user)
        self.assertNotIn("RUBRIC:\n", user)
        self.assertNotIn("EXAMPLE GRADED SCRIPT", user)

    def test_normalizes_verified_rubric(self) -> None:
        unit = AssessmentUnit(label="Part 2 Q1", max_mark=5.0, grading_mode="deterministic")
        normalized = normalize_verified_rubric(
            {
                "rubric_text": "Award method marks, equation accuracy, and interpretation credit across the full range.",
                "issues": ["Added explicit algebra check.", "Retained original structure."],
                "confidence_0_to_100": 88,
            },
            unit,
        )
        self.assertEqual(normalized["confidence_0_to_100"], 88.0)
        self.assertEqual(normalized["issues"][0], "Added explicit algebra check.")

    @patch("marking_pipeline.core._call_ollama_json")
    def test_verifies_assessment_rubrics(self, mock_call: Mock) -> None:
        context = prepare_marking_context(
            rubric_text="Part 1 (15 marks)\nDiscuss the policy.\nout of 15",
            brief_text="",
            marking_scheme_text="",
            graded_sample_text="",
            other_context_text="",
        )
        assessment_map = AssessmentMap(
            units=(
                AssessmentUnit(
                    label="Part 1",
                    max_mark=15.0,
                    grading_mode="analytical",
                    rubric_text="Use holistic judgement.",
                ),
            ),
            overall_max_mark=15.0,
            scale_confidence_0_to_100=95.0,
        )
        mock_call.return_value = {
            "rubric_text": "Use UK classification bands with explicit distinctions between First, 2:1, 2:2, Third/Pass, Fail, and Missing.",
            "issues": ["Added missing band language."],
            "confidence_0_to_100": 91,
        }

        verified = verify_assessment_rubrics(assessment_map, context, verifier_model_name="gemma3:4b")
        self.assertEqual(verified.units[0].rubric_confidence_0_to_100, 91.0)
        self.assertEqual(verified.units[0].rubric_issues, ("Added missing band language.",))
        self.assertIn("First, 2:1, 2:2", verified.units[0].rubric_text)
        self.assertEqual(mock_call.call_count, 1)

    @patch("marking_pipeline.core._call_ollama_json")
    def test_part_analysis_retries_when_lists_are_too_short(self, mock_call: Mock) -> None:
        context = prepare_marking_context("Maximum mark: 20", "", "", "", "")
        part = SubmissionPart(label="Question 1", section_text="Question 1\nStudent answer text", max_mark=20.0)
        mock_call.side_effect = [
            {
                "section_label": "Question 1",
                "provisional_score": 10,
                "strengths": ["one"],
                "weaknesses": ["weak one", "weak two"],
                "evidence": ["line 1"],
                "coverage_comment": "Covers the section.",
            },
            {
                "section_label": "Question 1",
                "provisional_score": 10,
                "strengths": ["strong one", "strong two"],
                "weaknesses": ["weak one", "weak two"],
                "evidence": ["line 1"],
                "coverage_comment": "Covers the section.",
            },
        ]

        result = _run_part_analysis_with_retry(
            part=part,
            context=context,
            filename="student1.pdf",
            model_name="llama3.1:8b",
            ollama_url="http://localhost:11434/api/chat",
        )
        self.assertEqual(result["provisional_score"], 10.0)
        self.assertEqual(mock_call.call_count, 2)

    @patch("marking_pipeline.core._call_ollama_json")
    def test_rubric_verifier_fails_soft_and_keeps_original_rubric(self, mock_call: Mock) -> None:
        context = prepare_marking_context(
            rubric_text="Part 2 Q3 (7.5 marks)\nDerive the relevant conditions.\nout of 7.5",
            brief_text="",
            marking_scheme_text="",
            graded_sample_text="",
            other_context_text="",
        )
        assessment_map = AssessmentMap(
            units=(
                AssessmentUnit(
                    label="Part 2 Q3",
                    max_mark=7.5,
                    grading_mode="deterministic",
                    rubric_text="Original deterministic rubric text.",
                ),
            ),
            overall_max_mark=7.5,
            scale_confidence_0_to_100=95.0,
        )
        mock_call.side_effect = ValueError("No JSON object found in model response: {bad")

        verified = verify_assessment_rubrics(assessment_map, context, verifier_model_name="qwen2:7b")
        self.assertEqual(verified.units[0].rubric_text, "Original deterministic rubric text.")
        self.assertEqual(verified.units[0].rubric_confidence_0_to_100, 0.0)
        self.assertIn("rubric_verifier_failed:", verified.units[0].rubric_issues[0])
        self.assertEqual(mock_call.call_count, 2)

    def test_refines_segmented_parts_to_subparts(self) -> None:
        context = prepare_marking_context(
            rubric_text=(
                "Part 2 (40 marks)\n"
                "1. [5 marks] First item\n"
                "2. [7.5 marks] Second item\n"
                "3. [7.5 marks] Third item\n"
                "4. [5 marks] Fourth item\n"
                "out of 40\n"
            ),
            brief_text="",
            marking_scheme_text="",
            graded_sample_text="",
            other_context_text="",
        )
        parts = [
            SubmissionPart(
                label="Part 2",
                anchor_text="Part 2",
                section_text="Part 2\nQ1\nAnswer one\nQ2\nAnswer two\nQ3\nAnswer three\nQ4\nAnswer four",
                max_mark=40.0,
            )
        ]
        refined = refine_submission_granularity(parts, context, "student.docx", "qwen2:7b")
        self.assertEqual([part.label for part in refined], ["Part 2 Q1", "Part 2 Q2", "Part 2 Q3", "Part 2 Q4"])
        self.assertTrue(all(part.max_mark is not None for part in refined))

    def test_reconciles_detected_parts_with_expected_parts(self) -> None:
        context = prepare_marking_context(
            rubric_text="Part 1 (15 marks)\nPart 2 (40 marks)\nPart 3 (20 marks)\nPart 4 (10 marks)\nout of 85",
            brief_text="",
            marking_scheme_text="",
            graded_sample_text="",
            other_context_text="",
        )
        detected = normalize_detected_parts(
            {"sections": [{"label": "Part 1", "focus_hint": "First", "anchor_text": "Part 1"}, {"label": "Part 3", "focus_hint": "Third", "anchor_text": "Part 3"}]}
        )
        reconciled = reconcile_detected_parts(detected, context)
        self.assertEqual([part.label for part in reconciled], ["Part 1", "Part 2", "Part 3", "Part 4"])
        self.assertEqual([part.label for part in extract_expected_parts_from_context(context)], ["Part 1", "Part 2", "Part 3", "Part 4"])

    def test_discards_placeholder_and_unexpected_parts_when_context_defines_structure(self) -> None:
        context = prepare_marking_context(
            rubric_text="Part 1 (15 marks)\nPart 2 (40 marks)\nPart 3 (15 marks)\nPart 4 (15 marks)\nout of 85",
            brief_text="",
            marking_scheme_text="",
            graded_sample_text="",
            other_context_text="",
        )
        detected = normalize_detected_parts(
            {
                "sections": [
                    {"label": "Section above", "focus_hint": "junk", "anchor_text": "Section above"},
                    {"label": "Question 1", "focus_hint": "wrong structure", "anchor_text": "Question 1"},
                    {"label": "Part 2", "focus_hint": "matched", "anchor_text": "Part 2"},
                ]
            }
        )
        reconciled = reconcile_detected_parts(detected, context)
        self.assertEqual([part.label for part in reconciled], ["Part 1", "Part 2", "Part 3", "Part 4"])

    def test_refine_submission_granularity_expands_missing_parent_into_expected_subparts(self) -> None:
        context = prepare_marking_context(
            rubric_text="Part 3 (30 marks)\n1. [7.5 marks]\n2. [7.5 marks]\n3. [7.5 marks]\n4. [7.5 marks]\nout of 30",
            brief_text="",
            marking_scheme_text="",
            graded_sample_text="",
            other_context_text="",
        )
        parts = [SubmissionPart(label="Part 3", anchor_text="Part 3", section_text="", max_mark=30.0)]
        refined = refine_submission_granularity(parts, context, "student.docx", "qwen2:7b")
        self.assertEqual([part.label for part in refined], ["Part 3 Q1", "Part 3 Q2", "Part 3 Q3", "Part 3 Q4"])
        self.assertTrue(all(part.section_text == "" for part in refined))

    def test_builds_zero_credit_analysis_for_missing_expected_part(self) -> None:
        part = SubmissionPart(label="Part 3 Q4", max_mark=7.5, marking_guidance="Deterministic derivation section.")
        analysis = build_missing_part_analysis(part)
        self.assertEqual(analysis["section_label"], "Part 3 Q4")
        self.assertEqual(analysis["provisional_score"], 0.0)
        self.assertIsNone(analysis["provisional_score_0_to_100"])
        self.assertIn("zero", analysis["coverage_comment"].lower())


class CalibrationTests(unittest.TestCase):
    @patch("marking_pipeline.core.requests.post")
    def test_calibrates_marks_across_students(self, mock_post: Mock) -> None:
        context = prepare_marking_context("Maximum mark: 20", "", "", "", "")
        response = Mock()
        response.json.return_value = {
            "message": {
                "content": '{"calibrated_results": [{"filename": "a.pdf", "adjusted_mark": 16, "rationale": "Slight downward adjustment for weaker evidence."}, {"filename": "b.pdf", "adjusted_mark": 18, "rationale": "Slight upward adjustment for stronger coverage."}]}'
            }
        }
        response.raise_for_status.return_value = None
        mock_post.return_value = response

        result = calibrate_marks_across_students(
            [
                {"filename": "a.pdf", "total_mark": 17, "detected_part_count": 2, "strengths": ["x", "y"], "weaknesses": ["a", "b"], "validation_notes": []},
                {"filename": "b.pdf", "total_mark": 17, "detected_part_count": 2, "strengths": ["x", "y"], "weaknesses": ["a", "b"], "validation_notes": []},
            ],
            model_name="qwen2:7b",
            model_label="Qwen2_7B",
            context=context,
            assessment_name="Assessment 1",
        )

        self.assertEqual(result["a.pdf"]["adjusted_mark"], 16)
        self.assertEqual(result["b.pdf"]["delta"], 1.0)


class VerificationTests(unittest.TestCase):
    def test_normalizes_verification_result(self) -> None:
        result = normalize_verification_result(
            {
                "agreement": "minor_concern",
                "confidence_0_to_100": 82,
                "issues": ["mark may be a little generous", "feedback is generic"],
                "recommendation": "review",
            }
        )
        self.assertEqual(result["agreement"], "minor_concern")
        self.assertEqual(result["confidence_0_to_100"], 82.0)

    @patch("marking_pipeline.core.requests.post")
    def test_verify_marking_result_calls_model(self, mock_post: Mock) -> None:
        context = prepare_marking_context("Maximum mark: 20", "", "", "", "")
        response = Mock()
        response.json.return_value = {
            "message": {
                "content": '{"agreement": "agree", "confidence_0_to_100": 77, "issues": [], "recommendation": "accept"}'
            }
        }
        response.raise_for_status.return_value = None
        mock_post.return_value = response

        result = verify_marking_result(
            script_text="Student answer",
            context=context,
            filename="student1.pdf",
            grading_result={"total_mark": 17, "max_mark": 20, "overall_feedback": LONG_FEEDBACK, "strengths": [], "weaknesses": [], "covered_parts": ["Whole Submission"]},
            model_name="gemma3:4b",
        )
        self.assertEqual(result["agreement"], "agree")

    @patch("marking_pipeline.core.requests.post")
    def test_regrade_marking_result_revises_grade(self, mock_post: Mock) -> None:
        context = prepare_marking_context("Maximum mark: 20", "", "", "", "")
        response = Mock()
        response.json.return_value = {
            "message": {
                "content": '{"total_mark": 16, "max_mark": 20, "overall_feedback": "Whole Submission is revised after reviewing the verifier concerns. The answer contains some useful explanation and a reasonable grasp of the topic, but the prior mark was too generous because important parts of the reasoning remain underdeveloped. The revised judgement now reflects the thinner evidence base, the incomplete treatment of the question, and the need for more explicit justification of key claims. There are still strengths in structure and engagement, but the weaknesses matter enough to justify a lower mark overall across the submission. The revision also takes account of the fact that some claims are asserted rather than defended, that the evidential support is uneven across the answer, and that the analysis does not sustain the stronger conclusions implied by the original mark.", "covered_parts": ["Whole Submission"], "strengths": ["clear structure", "engages with the topic"], "weaknesses": ["underdeveloped reasoning", "insufficient justification"]}'
            }
        }
        response.raise_for_status.return_value = None
        mock_post.return_value = response

        result = regrade_marking_result(
            script_text="Student answer",
            context=context,
            filename="student1.pdf",
            prior_result={"total_mark": 18, "max_mark": 20, "overall_feedback": LONG_FEEDBACK, "strengths": ["x", "y"], "weaknesses": ["a", "b"], "covered_parts": ["Whole Submission"]},
            verifier_result={"agreement": "major_concern", "confidence_0_to_100": 88, "issues": ["mark too generous"], "recommendation": "regrade"},
            model_name="qwen2:7b",
        )
        self.assertEqual(result["total_mark"], 16)


class MarkingContextTests(unittest.TestCase):
    def test_requires_rubric_or_marking_scheme(self) -> None:
        with self.assertRaises(ValueError):
            prepare_marking_context("", "", "", "", "")

    def test_requires_explicit_marking_scale(self) -> None:
        with self.assertRaises(ValueError):
            prepare_marking_context("Rubric without a total mark", "", "", "", "")

    def test_suggests_analytical_scale_from_documents(self) -> None:
        suggestion = suggest_marking_scale(
            rubric_text="Part 1 (15 marks)\nPart 2 (40 marks)\nPart 3 (30 marks)\nout of 85",
            brief_text="",
            marking_scheme_text="",
            graded_sample_text="",
            other_context_text="",
        )
        self.assertEqual(suggestion["label"], "Analytical / evaluative (0-85, typical 20-80)")
        self.assertEqual(suggestion["max_mark"], 85.0)


class AssessmentBundleTests(unittest.TestCase):
    def make_temp_dir(self) -> Path:
        path = Path("tests") / f".tmp_{uuid4().hex}"
        path.mkdir(parents=True, exist_ok=False)
        self.addCleanup(lambda: rmtree(path, ignore_errors=True))
        return path

    def test_discovers_assessment_subfolders_and_classifies_files(self) -> None:
        root = self.make_temp_dir()
        assessment_one = root / "assessment_one"
        assessment_one.mkdir()
        (assessment_one / "rubric.txt").write_text("Maximum mark: 20", encoding="utf-8")
        (assessment_one / "assignment_brief.txt").write_text("Brief text", encoding="utf-8")
        (assessment_one / "student_a.txt").write_text("Student answer A", encoding="utf-8")

        assessment_two = root / "assessment_two"
        assessment_two.mkdir()
        submissions = assessment_two / "submissions"
        submissions.mkdir()
        (assessment_two / "marking_scheme.txt").write_text("Mark out of 30", encoding="utf-8")
        (submissions / "student_b.txt").write_text("Student answer B", encoding="utf-8")

        bundles = discover_assessment_bundles(root)

        self.assertEqual([bundle.name for bundle in bundles], ["assessment_one", "assessment_two"])
        self.assertEqual([path.name for path in bundles[0].rubric_files], ["rubric.txt"])
        self.assertEqual([path.name for path in bundles[0].brief_files], ["assignment_brief.txt"])
        self.assertEqual([path.name for path in bundles[0].submission_files], ["student_a.txt"])
        self.assertEqual([path.name for path in bundles[1].marking_scheme_files], ["marking_scheme.txt"])
        self.assertEqual([path.name for path in bundles[1].submission_files], ["student_b.txt"])

    def test_builds_marking_context_per_assessment_bundle(self) -> None:
        root = self.make_temp_dir()
        assessment = root / "assessment_one"
        assessment.mkdir()
        (assessment / "rubric.txt").write_text("Maximum mark: 25", encoding="utf-8")
        (assessment / "student_a.txt").write_text("Student answer A", encoding="utf-8")

        bundle = discover_assessment_bundles(root)[0]
        context = build_marking_context_from_bundle(bundle)

        self.assertEqual(context.max_mark, 25)
        self.assertIn("Maximum mark: 25", context.rubric_text)

    def test_builds_single_assessment_bundle_from_student_subfolders(self) -> None:
        root = self.make_temp_dir()
        (root / "student_one").mkdir()
        (root / "student_two").mkdir()
        (root / "student_one" / "answer1.txt").write_text("Student answer A", encoding="utf-8")
        (root / "student_two" / "answer2.txt").write_text("Student answer B", encoding="utf-8")
        (root / "rubric.txt").write_text("Maximum mark: 20", encoding="utf-8")

        bundle = build_single_assessment_bundle(root, assessment_name="EC3040")

        self.assertEqual(bundle.name, "EC3040")
        self.assertEqual([path.name for path in bundle.rubric_files], ["rubric.txt"])
        self.assertEqual(sorted(path.name for path in bundle.submission_files), ["answer1.txt", "answer2.txt"])


class CacheTests(unittest.TestCase):
    class FakeUpload:
        def __init__(self, name: str, content: bytes) -> None:
            self.name = name
            self._content = content

        def getvalue(self) -> bytes:
            return self._content

    def make_temp_dir(self) -> Path:
        path = Path("tests") / f".tmp_{uuid4().hex}"
        path.mkdir(parents=True, exist_ok=False)
        self.addCleanup(lambda: rmtree(path, ignore_errors=True))
        return path

    def test_saves_and_loads_last_ingest_snapshot(self) -> None:
        temp_root = self.make_temp_dir()
        cache_dir = temp_root / "cache"
        ingest_dir = cache_dir / "last_ingest"
        manifest_path = ingest_dir / "manifest.json"

        with (
            patch.object(cache_module, "CACHE_ROOT", cache_dir),
            patch.object(cache_module, "LAST_INGEST_DIR", ingest_dir),
            patch.object(cache_module, "LAST_INGEST_MANIFEST", manifest_path),
        ):
            saved_path = cache_module.save_ingest_snapshot(
                use_assessment_folders=False,
                single_assessment_folder_mode=False,
                assessment_root="",
                folder_keywords={"rubric": ("rubric",), "brief": ("brief",), "marking_scheme": ("scheme",), "graded_sample": ("sample",), "other": ("other",)},
                scale_profile="Analytical / evaluative (0-85, typical 20-80)",
                manual_max_mark=85.0,
                document_only_mode=True,
                script_files=[self.FakeUpload("student one.pdf", b"script")],
                csv_file=None,
                rubric_files=[self.FakeUpload("rubric.pdf", b"rubric")],
                brief_files=[],
                marking_scheme_files=[],
                graded_sample_files=[],
                other_files=[],
            )

            self.assertEqual(saved_path, manifest_path)
            loaded = cache_module.load_ingest_snapshot()
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded["manual_max_mark"], 85.0)
            self.assertEqual(len(loaded["script_files"]), 1)
            self.assertTrue(Path(loaded["script_files"][0]).exists())
            self.assertTrue(Path(loaded["documents"]["rubric_files"][0]).exists())


class CallOllamaTests(unittest.TestCase):
    @patch("marking_pipeline.core.requests.post")
    def test_call_ollama_normalizes_percentage_scale(self, mock_post: Mock) -> None:
        context = prepare_marking_context("Question 1 (10 marks)\nQuestion 2 (10 marks)\nMaximum mark: 20", "", "", "", "")
        structure = Mock()
        structure.json.return_value = {
            "message": {
                "content": '{"sections": [{"label": "Question 1", "focus_hint": "First answer", "anchor_text": "Question 1"}, {"label": "Question 2", "focus_hint": "Second answer", "anchor_text": "Question 2"}]}'
            }
        }
        structure.raise_for_status.return_value = None
        part_one = Mock()
        part_one.json.return_value = {
            "message": {
                "content": '{"section_label": "Question 1", "provisional_score_0_to_100": 90, "strengths": ["good explanation", "relevant evidence"], "weaknesses": ["limited depth", "minor omission"], "evidence": ["mentions VAT incidence"], "coverage_comment": "Covers Question 1 clearly."}'
            }
        }
        part_one.raise_for_status.return_value = None
        part_two = Mock()
        part_two.json.return_value = {
            "message": {
                "content": '{"section_label": "Question 2", "provisional_score_0_to_100": 100, "strengths": ["clear reasoning", "good structure"], "weaknesses": ["small omission", "limited precision"], "evidence": ["mentions elasticity"], "coverage_comment": "Covers Question 2 clearly."}'
            }
        }
        part_two.raise_for_status.return_value = None
        synthesis = Mock()
        synthesis.json.return_value = {
            "message": {
                "content": '{"overall_feedback": "Question 1 is handled well and Question 2 is also handled well. The response includes clear explanation, relevant evidence, strong structure, and engagement with the policy problem. It shows a solid grasp of the theoretical core, a sensible attempt to apply the framework, and a reasonably disciplined use of the supplied material across both sections. The analysis is coherent, the answer remains focused on the task, and the student usually explains why their claims follow from the evidence they provide. There are still some limitations in depth and precision, and one omission remains in the treatment of the evidence. Even so, the script demonstrates strong command of the material across both sections and reaches a high standard overall with room for refinement in a few places, especially when moving from explanation to evaluation and from description to judgement.", "covered_parts": ["Question 1", "Question 2"], "strengths": ["clear explanation", "relevant evidence"], "weaknesses": ["limited depth", "limited precision"]}'
            }
        }
        synthesis.raise_for_status.return_value = None
        mock_post.side_effect = [structure, part_one, part_two, synthesis]

        result = call_ollama("Question 1\nStudent answer text\nQuestion 2\nMore answer text", context, "student1.pdf", "llama3.1:8b")

        self.assertEqual(result["total_mark"], 19)
        self.assertEqual(result["max_mark"], 20)
        self.assertEqual(result["detected_part_count"], 2)

    @patch("marking_pipeline.core.requests.post")
    def test_call_ollama_rejects_missing_message_content(self, mock_post: Mock) -> None:
        context = prepare_marking_context("Maximum mark: 20", "", "", "", "")
        response = Mock()
        response.json.return_value = {"unexpected": "shape"}
        response.raise_for_status.return_value = None
        mock_post.return_value = response

        with self.assertRaises(ValueError):
            call_ollama("Student answer text", context, "student1.pdf", "llama3.1:8b")

    def test_call_ollama_rejects_empty_script_text(self) -> None:
        context = prepare_marking_context("Maximum mark: 20", "", "", "", "")
        with self.assertRaises(ValueError):
            call_ollama("", context, "student1.pdf", "llama3.1:8b")


if __name__ == "__main__":
    unittest.main()
