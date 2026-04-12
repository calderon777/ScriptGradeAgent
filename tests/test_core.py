import unittest
from unittest.mock import Mock, patch

from marking_pipeline.core import call_ollama, infer_max_mark_from_texts, normalize_marking_result, parse_json_object, prepare_marking_context


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
                "overall_feedback": "Detailed feedback with strengths and weaknesses.",
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
                    "overall_feedback": "Detailed feedback with strengths and weaknesses.",
                },
                expected_max_mark=20,
            )


class MarkingContextTests(unittest.TestCase):
    def test_requires_rubric_or_marking_scheme(self) -> None:
        with self.assertRaises(ValueError):
            prepare_marking_context("", "", "", "", "")

    def test_requires_explicit_marking_scale(self) -> None:
        with self.assertRaises(ValueError):
            prepare_marking_context("Rubric without a total mark", "", "", "", "")


class CallOllamaTests(unittest.TestCase):
    @patch("marking_pipeline.core.requests.post")
    def test_call_ollama_normalizes_percentage_scale(self, mock_post: Mock) -> None:
        context = prepare_marking_context("Maximum mark: 20", "", "", "", "")
        response = Mock()
        response.json.return_value = {
            "message": {
                "content": '{"total_mark": 95, "max_mark": 100, "overall_feedback": "Detailed feedback with two strengths and two weaknesses."}'
            }
        }
        response.raise_for_status.return_value = None
        mock_post.return_value = response

        result = call_ollama("Student answer text", context, "student1.pdf", "llama3.1:8b")

        self.assertEqual(result["total_mark"], 19)
        self.assertEqual(result["max_mark"], 20)

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
