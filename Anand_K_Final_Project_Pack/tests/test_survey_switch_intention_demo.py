import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from analyse_survey_and_demo_model import survey_switch_intention_demo


def make_survey() -> pd.DataFrame:
    columns = ["Timestamp"] + [
        f"{number}. Question {number}" for number in range(1, 18)
    ]
    rows = []
    for index in range(120):
        row = [f"2026-01-{index % 28 + 1:02d}"]
        row.extend(f"option_{(index + number) % 3}" for number in range(1, 9))
        row.extend((index + offset) % 5 + 1 for offset in range(9, 12))
        row.extend(f"option_{(index + number) % 3}" for number in range(12, 15))
        row.append(5 if index % 2 else 2)
        row.extend(("reason", "improvement"))
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def save_survey(directory: Path, survey: pd.DataFrame) -> Path:
    path = directory / "survey.csv"
    survey.to_csv(path, index=False)
    return path


class SurveySwitchIntentionDemoTests(unittest.TestCase):
    def test_writes_valid_metrics_and_confusion_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            survey_path = save_survey(output_dir, make_survey())
            with redirect_stdout(StringIO()):
                survey_switch_intention_demo(survey_path, output_dir)

            result = pd.read_csv(
                output_dir / "Survey_Switch_Intention_Model_DEMO_ONLY.csv"
            )

            self.assertEqual(len(result), 3)
            self.assertTrue((result["test_samples"] == 30).all())
            self.assertTrue(
                (
                    result["true_negatives"]
                    + result["false_positives"]
                    + result["false_negatives"]
                    + result["true_positives"]
                    == result["test_samples"]
                ).all()
            )
            self.assertTrue(
                result[["accuracy", "precision", "recall", "f1", "roc_auc"]]
                .apply(lambda values: values.between(0, 1))
                .all()
                .all()
            )
            self.assertEqual(
                set(result["target"]),
                {"High stated switching intention (rating 4-5)"},
            )

    def test_reports_missing_switch_question(self) -> None:
        survey = make_survey().drop(columns=["15. Question 15"])
        with tempfile.TemporaryDirectory() as temp_dir:
            survey_path = save_survey(Path(temp_dir), survey)
            with self.assertRaisesRegex(ValueError, "question 15"):
                survey_switch_intention_demo(survey_path, Path(temp_dir))

    def test_reports_when_only_one_target_class_is_present(self) -> None:
        survey = make_survey()
        survey["15. Question 15"] = 2
        with tempfile.TemporaryDirectory() as temp_dir:
            survey_path = save_survey(Path(temp_dir), survey)
            with self.assertRaisesRegex(ValueError, "each switching-intention class"):
                survey_switch_intention_demo(survey_path, Path(temp_dir))

    def test_saves_timestamped_file_if_standard_result_is_locked(self) -> None:
        survey = make_survey()
        original_to_csv = pd.DataFrame.to_csv

        def fail_standard_result_once(frame, path_or_buf=None, *args, **kwargs):
            if Path(path_or_buf).name == "Survey_Switch_Intention_Model_DEMO_ONLY.csv":
                raise PermissionError("simulated locked file")
            return original_to_csv(frame, path_or_buf, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            survey_path = save_survey(output_dir, survey)
            with patch.object(pd.DataFrame, "to_csv", fail_standard_result_once):
                with redirect_stdout(StringIO()):
                    survey_switch_intention_demo(survey_path, output_dir)

            fallback_files = list(
                output_dir.glob("Survey_Switch_Intention_Model_DEMO_ONLY_*.csv")
            )
            self.assertEqual(len(fallback_files), 1)
            self.assertEqual(len(pd.read_csv(fallback_files[0])), 3)


if __name__ == "__main__":
    unittest.main()
