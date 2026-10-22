"""Regression checks for repeated observations, estimability, ICC units, and trial inclusion.

Run with: python -m unittest discover -s tests -v
"""

import itertools
import unittest

import numpy as np
import pandas as pd
import pingouin as pg
import statsmodels.formula.api as smf

import paper_analysis_utils as u


class ReviewerCorrectionsTests(unittest.TestCase):
    def test_feature_distances_exclude_training_and_tutorials_in_every_session(self):
        for session in range(3):
            with self.subTest(session=session):
                if session == 0:
                    names = ["prototype_A", "example_d1"]
                else:
                    prototype = u.PROTOTYPE_CODES.loc[
                        (u.PROTOTYPE_CODES["prototype"] == "A") & (u.PROTOTYPE_CODES["retest"] == session),
                        "name",
                    ].iloc[0]
                    opposite = u.PROTOTYPE_CODES.loc[
                        (u.PROTOTYPE_CODES["prototype"] == "B") & (u.PROTOTYPE_CODES["retest"] == session),
                        "name",
                    ].iloc[0]
                    tokens = prototype.split("_")
                    tokens[0] = opposite.split("_")[0]
                    names = [prototype, "_".join(tokens)]
                data = pd.DataFrame({
                    "use": ["task", "task", "task", "tutorial"],
                    "block": ["test", "test", "train", "test"],
                    "name": names + [names[1], names[1]],
                    "correct": [1, 0, 1, 1],
                })
                before = data.copy(deep=True)
                selected = u.add_feature_distance(data, session)
                self.assertEqual(selected.index.tolist(), [0, 1])
                self.assertEqual(selected["distance"].tolist(), [0, 1])
                self.assertEqual(selected["correct"].mean(), 0.5)
                pd.testing.assert_frame_equal(data, before)

    @staticmethod
    def repeated_data():
        rng = np.random.default_rng(243)
        rows = []
        for subject in range(60):
            participant_effect = rng.normal(0, 0.08)
            for session in range(3):
                rows.append({
                    "subject_id": subject, "test": session,
                    "classification_strategy": ["All components", "No strategy", "Subset of components"][subject % 3],
                    "decision_strategy": ["Carefully evaluated", "Chose randomly", "Trusted gut"][(subject // 3) % 3],
                    "accuracy": 0.5 + participant_effect + 0.03 * session + rng.normal(0, 0.01),
                })
        return pd.DataFrame(rows)

    def test_repeated_sessions_are_not_independent_observations(self):
        data = self.repeated_data()
        tests, coefficients, diagnostics = u.strategy_metric_models(data, metrics=["accuracy"])
        naive = smf.ols("accuracy ~ " + u.STRATEGY_FORMULAS["additive"], data).fit()
        term = "C(classification_strategy)[T.No strategy]"
        corrected_se = coefficients.set_index("coefficient").loc[term, "se"]
        self.assertGreater(corrected_se, 1.2 * naive.bse[term])
        self.assertEqual(diagnostics.loc[0, "n_subjects"], 60)
        self.assertEqual(diagnostics.loc[0, "n_rows"], 180)
        self.assertGreater(diagnostics.loc[0, "random_to_residual_variance"], 5)
        self.assertEqual(set(tests["term"]), {"Classification strategy", "Decision strategy", "Session"})

    def test_empty_cells_make_full_factorial_unidentifiable(self):
        empty = {("A", "X", 0), ("A", "X", 1), ("C", "X", 2)}
        cells = [c for c in itertools.product("ABC", "XYZ", range(3)) if c not in empty]
        data = pd.DataFrame(cells, columns=["classification_strategy", "decision_strategy", "test"])
        data["subject_id"] = np.arange(len(data))
        audit, counts = u.strategy_design_audit(data)
        ranks = audit.set_index("model")
        self.assertEqual(ranks.loc["full_factorial", "rank"], 24)
        self.assertEqual(ranks.loc["full_factorial", "n_columns"], 27)
        self.assertTrue(ranks.loc["two_way", "full_rank"])
        self.assertTrue(ranks.loc["additive", "full_rank"])
        self.assertEqual(int(counts["n"].eq(0).sum()), 3)
        with self.assertRaisesRegex(ValueError, "auditing only"):
            u.strategy_metric_models(data, model_kind="full_factorial")

    def test_rank_deficiency_stops_before_inference(self):
        data = self.repeated_data()
        data["decision_strategy"] = data["classification_strategy"]
        with self.assertRaisesRegex(ValueError, "rank deficient"):
            u.strategy_metric_models(data, metrics=["accuracy"])

    def test_duplicate_sessions_and_missing_outcomes_are_rejected(self):
        data = self.repeated_data()
        duplicate = pd.concat([data, data.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "one row"):
            u.strategy_metric_models(duplicate, metrics=["accuracy"])
        with self.assertRaisesRegex(ValueError, "one observation"):
            u.icc_table(duplicate, metrics=["accuracy"])
        data.loc[0, "accuracy"] = np.nan
        with self.assertRaisesRegex(ValueError, "must be finite"):
            u.strategy_metric_models(data, metrics=["accuracy"])

    def test_icc_is_for_mean_of_three_and_retains_intervals(self):
        data = self.repeated_data()
        reported = u.icc_table(data, metrics=["accuracy"]).iloc[0]
        raw = pg.intraclass_corr(data=data, raters="test", targets="subject_id", ratings="accuracy").set_index("Type")
        single = raw.loc["ICC2", "ICC"]
        self.assertAlmostEqual(reported["icc"], 3 * single / (1 + 2 * single))
        self.assertEqual(reported["unit"], "Mean of 3 administrations")
        self.assertEqual(reported["k"], 3)
        np.testing.assert_array_equal([reported["ci95_low"], reported["ci95_high"]], raw.loc["ICC2k", "CI95%"])
        with self.assertRaisesRegex(ValueError, "same administrations"):
            u.icc_table(data.iloc[1:], metrics=["accuracy"])


if __name__ == "__main__":
    unittest.main()
