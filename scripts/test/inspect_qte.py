"""One-off script to inspect question_text_exact values in the prepared assessment map."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.run_human_benchmark import load_context
from marking_pipeline.core import prepare_assessment_map, _question_text_for_model, _build_part_task_text, SubmissionPart

ctx = load_context()
pam = prepare_assessment_map(context=ctx)

print("=== Assessment Map question_text_exact inventory ===\n")
for u in pam.original_map.units:
    qte = u.question_text_exact
    ends_ellipsis = qte.strip().endswith("...")
    print(f"Label:        {u.label}")
    print(f"task_type:    {u.task_type}")
    print(f"qte_len:      {len(qte)}")
    print(f"ends_ellipsis:{ends_ellipsis}")
    print(f"qte_full:     {repr(qte)}")
    # Simulate the payload cleaning
    limit_650 = _question_text_for_model(qte, max_chars=650)
    limit_1800 = _question_text_for_model(qte, max_chars=1800)
    limit_none = _question_text_for_model(qte, max_chars=None)
    print(f"after_clean_650:  ends_ellipsis={limit_650.endswith('...')}, len={len(limit_650)}")
    print(f"after_clean_1800: ends_ellipsis={limit_1800.endswith('...')}, len={len(limit_1800)}")
    print(f"after_clean_none: ends_ellipsis={limit_none.endswith('...')}, len={len(limit_none)}")
    print()
