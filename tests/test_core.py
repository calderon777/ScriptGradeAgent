from pathlib import Path
from shutil import rmtree
from uuid import uuid4
import unittest
from unittest.mock import Mock, patch

from marking_pipeline.core import (
    build_marking_context_from_bundle,
    build_single_assessment_bundle,
    build_submission_diagnostics,
    extract_expected_parts_from_context,
    extract_structure_guidance,
    calibrate_marks_across_students,
    call_ollama,
    compute_total_mark_from_part_scores,
    discover_assessment_bundles,
    infer_max_mark_from_texts,
    normalize_detected_parts,
    normalize_marking_result,
    normalize_verification_result,
    parse_json_object,
    prepare_marking_context,
    reconcile_detected_parts,
    regrade_marking_result,
    segment_submission_parts,
    SubmissionPart,
    verify_marking_result,
)
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

    def test_computes_single_part_total_from_expected_max_mark(self) -> None:
        total = compute_total_mark_from_part_scores(
            expected_max_mark=85.0,
            parts=[SubmissionPart(label="Whole Submission")],
            part_analyses=[
                {"provisional_score": None, "provisional_score_0_to_100": 80.0},
            ],
        )
        self.assertEqual(total, 68.0)

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
        self.assertEqual([part.label for part in reconciled], ["Part 1", "Part 3", "Part 2", "Part 4"])
        self.assertEqual([part.label for part in extract_expected_parts_from_context(context)], ["Part 1", "Part 2", "Part 3", "Part 4"])


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


class CallOllamaTests(unittest.TestCase):
    @patch("marking_pipeline.core.requests.post")
    def test_call_ollama_normalizes_percentage_scale(self, mock_post: Mock) -> None:
        context = prepare_marking_context("Maximum mark: 20", "", "", "", "")
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
                "content": '{"total_mark": 95, "max_mark": 100, "overall_feedback": "Question 1 is handled well and Question 2 is also handled well. The response includes clear explanation, relevant evidence, strong structure, and engagement with the policy problem. It shows a solid grasp of the theoretical core, a sensible attempt to apply the framework, and a reasonably disciplined use of the supplied material across both sections. The analysis is coherent, the answer remains focused on the task, and the student usually explains why their claims follow from the evidence they provide. There are still some limitations in depth and precision, and one omission remains in the treatment of the evidence. Even so, the script demonstrates strong command of the material across both sections and reaches a high standard overall with room for refinement in a few places, especially when moving from explanation to evaluation and from description to judgement.", "covered_parts": ["Question 1", "Question 2"], "strengths": ["clear explanation", "relevant evidence"], "weaknesses": ["limited depth", "limited precision"]}'
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
