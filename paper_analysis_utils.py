"""Utilities for reproducing the metacognition game paper analyses.

This module contains reusable analysis functions for the manuscript
reproducibility notebook. The notebook can be read top-to-bottom and rerun
from a fresh kernel without hidden state.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import linregress, nct, norm, pearsonr, t, ttest_rel
from statsmodels.stats.multitest import multipletests

from analysisfunctions import (
    calcBrierScore,
    calcCalibration,
    calcOutcomeVariance,
    calcResolution,
    convertCalibration,
    renameDfColumns,
)


# The task confidence scale has 4 keys mapped onto the .5-1 probability range.
# Passed explicitly to every scoring call so the key-to-probability mapping is
# fixed for all subjects rather than inferred from each subject's max key used.
N_CONFIDENCE_KEYS = 4

TASK_METRICS = [
    "accuracy",
    "brier_score",
    "calibration",
    "resolution",
    "outcome_variance",
    "rewarded_score",
]

TEST_LABELS = {
    0: "Initial Test",
    1: "1 Week Retest",
    2: "Final Retest",
}

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "brier_score": "Brier Score",
    "calibration": "Calibration",
    "resolution": "Resolution",
    "outcome_variance": "Outcome Variance",
    "rewarded_score": "Rewarded Score",
    "calibration_converted_score": "Converted Calibration",
}

PROTOTYPE_CODES = pd.DataFrame(
    {
        "prototype": ["A", "B", "A", "B"],
        "retest": [1, 1, 2, 2],
        "name": [
            "0-A-1_1-B-1_2-C-1_3-D-2_4-E-1_5-A-3_6-B-3_7-C-3_8-D-4_9-E-3",
            "0-A-2_1-B-2_2-C-2_3-D-1_4-E-2_5-A-4_6-B-4_7-C-4_8-D-3_9-E-4",
            "0-A-1_1-B-2_2-C-1_3-D-2_4-E-1_5-A-4_6-B-4_7-C-3_8-D-4_9-E-4",
            "0-A-2_1-B-1_2-C-2_3-D-1_4-E-2_5-A-3_6-B-3_7-C-4_8-D-3_9-E-3",
        ],
    }
)


@dataclass(frozen=True)
class PaperData:
    questionnaire_items: pd.DataFrame
    questionnaire_overall: pd.DataFrame
    questionnaire_subscales: pd.DataFrame
    initial: pd.DataFrame
    retest_1: pd.DataFrame
    retest_2: pd.DataFrame
    all_task_rows: pd.DataFrame
    completer_task_rows: pd.DataFrame


def ensure_output_dirs(base_dir: str | Path = "paper_repro_outputs") -> dict[str, Path]:
    base = Path(base_dir)
    paths = {
        "base": base,
        "figures": base / "figures",
        "tables": base / "tables",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_paper_data(data_dir: str | Path = "data") -> PaperData:
    data_dir = Path(data_dir)
    questionnaire_items = pd.read_csv(data_dir / "questionnaire_data.csv")
    questionnaire_overall = pd.read_csv(data_dir / "questionnaire_overall_scores.csv")
    questionnaire_subscales = pd.read_csv(data_dir / "questionnaire_subscale_scores.csv")

    initial = pd.read_csv(data_dir / "task_data_initial_test.csv")
    retest_1 = pd.read_csv(data_dir / "task_data_retest_1.csv")
    retest_2 = pd.read_csv(data_dir / "task_data_retest_2.csv")
    initial["test"] = 0
    retest_1["test"] = 1
    retest_2["test"] = 2

    all_task_rows = pd.concat([initial, retest_1, retest_2], ignore_index=True)
    completer_ids = set(retest_2["subject_id"].unique())
    initial = initial.loc[initial["subject_id"].isin(completer_ids)].copy()
    retest_1 = retest_1.loc[retest_1["subject_id"].isin(completer_ids)].copy()
    completer_task_rows = pd.concat([initial, retest_1, retest_2], ignore_index=True)

    return PaperData(
        questionnaire_items=questionnaire_items,
        questionnaire_overall=questionnaire_overall,
        questionnaire_subscales=questionnaire_subscales,
        initial=initial,
        retest_1=retest_1,
        retest_2=retest_2,
        all_task_rows=all_task_rows,
        completer_task_rows=completer_task_rows,
    )


def session_completeness_table(data: PaperData) -> pd.DataFrame:
    rows = []
    for test, df in [(0, data.initial), (1, data.retest_1), (2, data.retest_2)]:
        rows.append(
            {
                "test": test,
                "test_label": TEST_LABELS[test],
                "n_subjects": df["subject_id"].nunique(),
                "n_rows": df.shape[0],
                "n_test_trials_per_subject": int(
                    df.loc[df["block"] == "test"].groupby("subject_id").size().median()
                ),
                "n_unique_test_trials_per_subject": int(
                    df.loc[df["block"] == "test"]
                    .drop_duplicates(["subject_id", "trial_number"])
                    .groupby("subject_id")
                    .size()
                    .median()
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_task_session(df: pd.DataFrame, correct_rewarded_score: bool = True) -> pd.DataFrame:
    rows = []
    for subject_id in df["subject_id"].unique():
        data_subject = df.loc[df["subject_id"] == subject_id]
        test_trials = data_subject.loc[data_subject["block"] == "test"]
        estimates = test_trials["confidence"].to_numpy().astype(int)
        outcomes = test_trials["correct"].to_numpy().astype(int)
        accuracy = test_trials["correct"].mean()
        calibration_converted = convertCalibration(estimates, outcomes, n_keys=N_CONFIDENCE_KEYS)
        rewarded_score = (accuracy + calibration_converted) / 2
        if correct_rewarded_score:
            rewarded_score = max(rewarded_score, 0)

        rows.append(
            {
                "subject_id": float(subject_id),
                "accuracy": accuracy,
                "brier_score": calcBrierScore(estimates, outcomes, n_keys=N_CONFIDENCE_KEYS),
                "calibration": calcCalibration(estimates, outcomes, n_keys=N_CONFIDENCE_KEYS),
                "resolution": calcResolution(estimates, outcomes, n_keys=N_CONFIDENCE_KEYS),
                "outcome_variance": calcOutcomeVariance(outcomes),
                "calibration_converted_score": calibration_converted,
                "rewarded_score": rewarded_score,
            }
        )
    return pd.DataFrame(rows)


def build_task_summaries(data: PaperData) -> pd.DataFrame:
    frames = []
    for test, df in [(0, data.initial), (1, data.retest_1), (2, data.retest_2)]:
        summary = summarize_task_session(df)
        summary["test"] = test
        summary["test_label"] = TEST_LABELS[test]
        frames.append(summary)
    return pd.concat(frames, ignore_index=True)


def above_chance_subjects(task_summaries: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
    accuracy = task_summaries.pivot(index="subject_id", columns="test", values="accuracy")
    return accuracy.index[(accuracy > threshold).all(axis=1)].to_numpy()


def less_stringent_performance_subjects(task_summaries: pd.DataFrame) -> np.ndarray:
    accuracy = task_summaries.pivot(index="subject_id", columns="test", values="accuracy")
    return accuracy.index[(accuracy > 0.55).sum(axis=1) >= 2].to_numpy()


def sample_definition_table(task_summaries: pd.DataFrame) -> pd.DataFrame:
    accuracy = task_summaries.pivot(index="subject_id", columns="test", values="accuracy")
    rows = [
        ("All three sessions completed", np.repeat(True, len(accuracy))),
        ("Accuracy > .5 in all three test sessions", (accuracy > 0.5).all(axis=1)),
        ("Accuracy >= .5 in all three test sessions", (accuracy >= 0.5).all(axis=1)),
        ("Accuracy > .5 in at least two test sessions", (accuracy > 0.5).sum(axis=1) >= 2),
        ("Mean test accuracy > .5 across sessions", accuracy.mean(axis=1) > 0.5),
        ("Less stringent sensitivity rule: accuracy > .55 in at least two sessions", (accuracy > 0.55).sum(axis=1) >= 2),
    ]
    return pd.DataFrame({"definition": [r[0] for r in rows], "n": [int(np.sum(r[1])) for r in rows]})


def accuracy_calibration_correlations(task_summaries: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for test, test_df in task_summaries.groupby("test"):
        r, p = pearsonr(test_df["accuracy"], test_df["calibration"])
        rows.append({"test": test, "test_label": TEST_LABELS[test], "r": r, "p": p, "n": test_df.shape[0]})
    return pd.DataFrame(rows)


def calibration_change_tests(task_summaries: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    means = (
        task_summaries.groupby(["test", "test_label"])["calibration"]
        .mean()
        .reset_index(name="mean_calibration")
    )
    anova = pg.rm_anova(
        data=task_summaries,
        dv="calibration",
        within="test",
        subject="subject_id",
        correction=True,
        detailed=True,
    )
    wide = task_summaries.pivot(index="subject_id", columns="test", values="calibration")
    t_stat, p_one_sided = ttest_rel(wide[0], wide[2], alternative="greater")
    diff = wide[0] - wide[2]
    paired = pd.DataFrame(
        [
            {
                "comparison": "Initial > Final",
                "t": t_stat,
                "p_one_sided": p_one_sided,
                "cohens_dz": diff.mean() / diff.std(ddof=1),
                "mean_initial": wide[0].mean(),
                "mean_final": wide[2].mean(),
                "n": wide.shape[0],
            }
        ]
    )
    return means, anova, paired


def icc_table(task_summaries: pd.DataFrame, metrics: Iterable[str] = TASK_METRICS) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        icc = pg.intraclass_corr(
            data=task_summaries,
            raters="test",
            targets="subject_id",
            ratings=metric,
        )
        selected = icc.loc[(icc["Type"] == "ICC2k") & (icc["Description"] == "Average random raters")].iloc[0]
        rows.append(
            {
                "metric": metric,
                "metric_label": METRIC_LABELS[metric],
                "icc_type": selected["Type"],
                "icc": selected["ICC"],
                "f": selected["F"],
                "p": selected["pval"],
                "ci95": selected["CI95%"],
            }
        )
    return pd.DataFrame(rows)


def strategy_count_tables(completer_task_rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    strategy = completer_task_rows[
        ["subject_id", "strategy_prompt", "strategy_response", "test"]
    ].drop_duplicates()

    tables: dict[str, pd.DataFrame] = {}
    for prompt in ["strategy_classification", "decision_process"]:
        table = (
            strategy.loc[strategy["strategy_prompt"] == prompt]
            .groupby(["test", "strategy_response"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        table.insert(1, "test_label", table["test"].map(TEST_LABELS))
        tables[prompt] = table

    joint_tables = []
    for test in sorted(strategy["test"].unique()):
        pivot = strategy.loc[strategy["test"] == test].pivot(
            index="subject_id", columns="strategy_prompt", values="strategy_response"
        )
        crosstab = pd.crosstab(pivot["decision_process"], pivot["strategy_classification"])
        crosstab.insert(0, "decision_process", crosstab.index)
        crosstab.insert(0, "test_label", TEST_LABELS[test])
        crosstab.insert(0, "test", test)
        joint_tables.append(crosstab.reset_index(drop=True))
    tables["joint"] = pd.concat(joint_tables, ignore_index=True)
    return tables


def get_retest_distances(df: pd.DataFrame, retest: int) -> np.ndarray:
    prototype_a = PROTOTYPE_CODES.loc[
        (PROTOTYPE_CODES["prototype"] == "A") & (PROTOTYPE_CODES["retest"] == retest), "name"
    ].iloc[0].split("_")
    prototype_b = PROTOTYPE_CODES.loc[
        (PROTOTYPE_CODES["prototype"] == "B") & (PROTOTYPE_CODES["retest"] == retest), "name"
    ].iloc[0].split("_")

    distances = np.zeros((df.shape[0], 2))
    for i, name in enumerate(df["name"].to_list()):
        stim_code = name.split("_")
        distances[i, 0] = sum(proto != stim for proto, stim in zip(prototype_a, stim_code))
        distances[i, 1] = sum(proto != stim for proto, stim in zip(prototype_b, stim_code))
    return np.min(distances, axis=1)


def add_feature_distance(df: pd.DataFrame, test: int) -> pd.DataFrame:
    data_slice = df.loc[df["use"] == "task"].copy()
    if test == 0:
        data_slice.loc[data_slice["name"].str.contains("prototype"), "distance"] = 0
        for distance in [1, 2, 3, 4]:
            data_slice.loc[data_slice["name"].str.contains(f"d{distance}"), "distance"] = distance
    else:
        data_slice["distance"] = get_retest_distances(data_slice, retest=test)
    data_slice["distance"] = data_slice["distance"].astype(int)
    return data_slice


def fit_distance_models(data: PaperData, subject_ids: Iterable[float] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    subject_ids = None if subject_ids is None else set(subject_ids)
    model_rows = []
    grouped_frames = []
    for test, df in [(0, data.initial), (1, data.retest_1), (2, data.retest_2)]:
        data_slice = add_feature_distance(df, test)
        if subject_ids is not None:
            data_slice = data_slice.loc[data_slice["subject_id"].isin(subject_ids)]

        grouped = data_slice.groupby(["subject_id", "distance"])["correct"].mean().reset_index()
        grouped["test"] = test
        grouped["test_label"] = TEST_LABELS[test]
        grouped_frames.append(grouped)

        result = smf.mixedlm("correct ~ distance", grouped, groups=grouped["subject_id"]).fit()
        model_rows.append(
            {
                "test": test,
                "test_label": TEST_LABELS[test],
                "n_subjects": grouped["subject_id"].nunique(),
                "distance_slope": result.params["distance"],
                "distance_se": result.bse["distance"],
                "distance_z": result.tvalues["distance"],
                "distance_p": result.pvalues["distance"],
                "scale": result.scale,
            }
        )
    return pd.DataFrame(model_rows), pd.concat(grouped_frames, ignore_index=True)


def subject_distance_slopes(grouped_distance: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (test, subject_id), group in grouped_distance.groupby(["test", "subject_id"]):
        slope, intercept, r_value, p_value, stderr = linregress(group["distance"], group["correct"])
        rows.append(
            {
                "test": test,
                "test_label": TEST_LABELS[test],
                "subject_id": subject_id,
                "slope": slope,
                "intercept": intercept,
                "r": r_value,
                "p": p_value,
                "stderr": stderr,
            }
        )
    return pd.DataFrame(rows)


def mean_task_summaries(task_summaries: pd.DataFrame) -> pd.DataFrame:
    return (
        task_summaries.groupby("subject_id")[
            TASK_METRICS + ["calibration_converted_score"]
        ]
        .mean()
        .reset_index()
    )


def questionnaire_task_correlations(
    questionnaire_subscales: pd.DataFrame,
    task_summaries: pd.DataFrame,
    subject_ids: Iterable[float] | None = None,
) -> pd.DataFrame:
    task_mean = mean_task_summaries(task_summaries)
    joint = pd.merge(questionnaire_subscales, task_mean, on="subject_id", how="inner")
    if subject_ids is not None:
        joint = joint.loc[joint["subject_id"].isin(set(subject_ids))]

    rows = []
    for subscale in sorted(joint["subscale"].dropna().unique()):
        subscale_df = joint.loc[joint["subscale"] == subscale]
        for metric in TASK_METRICS:
            r, p = pearsonr(subscale_df["score"], subscale_df[metric])
            rows.append(
                {
                    "subscale": subscale,
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "n": subscale_df.shape[0],
                    "r": r,
                    "p_uncorrected": p,
                }
            )
    results = pd.DataFrame(rows)
    _, bonf_p, _, _ = multipletests(results["p_uncorrected"], method="bonferroni")
    _, fdr_p, _, _ = multipletests(results["p_uncorrected"], method="fdr_bh")
    results["p_bonferroni"] = bonf_p
    results["p_fdr_bh"] = fdr_p
    results["significant_uncorrected"] = results["p_uncorrected"] < 0.05
    results["significant_bonferroni"] = results["p_bonferroni"] < 0.05
    results["significant_fdr_bh"] = results["p_fdr_bh"] < 0.05
    return results.sort_values(["p_uncorrected", "subscale", "metric"]).reset_index(drop=True)


def selected_questionnaire_correlations(correlations: pd.DataFrame) -> pd.DataFrame:
    selected = [
        ("bfi10_extraversion", "brier_score"),
        ("bfi10_extraversion", "resolution"),
        ("bfi10_agreeableness", "resolution"),
        ("bapq_aloof", "resolution"),
        ("bfi10_extraversion", "outcome_variance"),
    ]
    mask = pd.Series(False, index=correlations.index)
    for subscale, metric in selected:
        mask |= (correlations["subscale"] == subscale) & (correlations["metric"] == metric)
    return correlations.loc[mask].copy()


def pearson_required_n_fisher_z(
    r: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
) -> int:
    z_effect = abs(np.arctanh(r))
    z_alpha = norm.ppf(1 - alpha / 2) if two_sided else norm.ppf(1 - alpha)
    z_power = norm.ppf(power)
    return int(np.ceil(((z_alpha + z_power) / z_effect) ** 2 + 3))


def pearson_power_nct_approx(n: int, r: float, alpha: float = 0.05, two_sided: bool = True) -> float:
    df = n - 2
    noncentrality = r * np.sqrt(df / (1 - r**2))
    if two_sided:
        critical_t = t.ppf(1 - alpha / 2, df)
        return nct.sf(critical_t, df, noncentrality) + nct.cdf(-critical_t, df, noncentrality)
    critical_t = t.ppf(1 - alpha, df)
    return nct.sf(critical_t, df, noncentrality)


def pearson_required_n_nct_approx(
    r: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
    max_n: int = 10000,
) -> int:
    for n in range(4, max_n + 1):
        if pearson_power_nct_approx(n, r, alpha=alpha, two_sided=two_sided) >= power:
            return n
    raise ValueError(f"Required N exceeds max_n={max_n}")


def power_analysis_tables(
    target_rs: Iterable[float] = (0.20, 0.25, 0.30),
    powers: Iterable[float] = (0.80, 0.90),
    n_tests: int = 66,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    conservative_alpha = 0.05 / n_tests
    summary_rows = []
    for target_r in target_rs:
        for target_power in powers:
            summary_rows.append(
                {
                    "target_r": target_r,
                    "power": target_power,
                    "alpha": 0.05,
                    "correction_family": "single planned test",
                    "required_n_fisher_z": pearson_required_n_fisher_z(target_r, alpha=0.05, power=target_power),
                    "required_n_nct": pearson_required_n_nct_approx(target_r, alpha=0.05, power=target_power),
                }
            )
            summary_rows.append(
                {
                    "target_r": target_r,
                    "power": target_power,
                    "alpha": conservative_alpha,
                    "correction_family": f"conservative corrected threshold, m={n_tests}",
                    "required_n_fisher_z": pearson_required_n_fisher_z(
                        target_r, alpha=conservative_alpha, power=target_power
                    ),
                    "required_n_nct": pearson_required_n_nct_approx(
                        target_r, alpha=conservative_alpha, power=target_power
                    ),
                }
            )

    current_power_rows = []
    for target_r in target_rs:
        current_power_rows.append(
            {
                "target_r": target_r,
                "n": 80,
                "alpha": 0.05,
                "correction_family": "single planned test",
                "power_nct": pearson_power_nct_approx(80, target_r, alpha=0.05),
            }
        )
        current_power_rows.append(
            {
                "target_r": target_r,
                "n": 80,
                "alpha": conservative_alpha,
                "correction_family": f"conservative corrected threshold, m={n_tests}",
                "power_nct": pearson_power_nct_approx(80, target_r, alpha=conservative_alpha),
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(current_power_rows)


def questionnaire_subscale_covariance(questionnaire_subscales: pd.DataFrame) -> pd.DataFrame:
    vals = questionnaire_subscales.pivot_table(
        values="score", index="subject_id", columns="subscale", aggfunc="first"
    )
    vals = vals.apply(lambda col: (col - col.mean()) / col.std(ddof=0), axis=0)
    vals = renameDfColumns(vals.copy())
    return vals.corr()


def save_table(df: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def save_indexed_table(df: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df


def plot_accuracy_calibration(correlations: pd.DataFrame, task_summaries: pd.DataFrame, path: str | Path):
    fig, ax = plt.subplots(figsize=(5, 4))
    for test, test_df in task_summaries.groupby("test"):
        corr_row = correlations.loc[correlations["test"] == test].iloc[0]
        sns.regplot(
            data=test_df,
            x="accuracy",
            y="calibration",
            ax=ax,
            label=f"{TEST_LABELS[test]}, r={corr_row['r']:.2f}, p={corr_row['p']:.3g}",
        )
    ax.set_xlabel("Accuracy (higher better)")
    ax.set_ylabel("Calibration (lower better)")
    ax.set_title("Accuracy vs Calibration")
    ax.legend(fontsize=8)
    sns.despine()
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig


def plot_icc_reliability(task_summaries: pd.DataFrame, icc: pd.DataFrame, path: str | Path):
    plot_df = task_summaries.rename(columns=METRIC_LABELS).copy()
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    axes = axes.flatten()
    for i, metric in enumerate(TASK_METRICS):
        label = METRIC_LABELS[metric]
        sns.violinplot(data=plot_df, x="test_label", y=label, color="lightgray", ax=axes[i])
        sns.stripplot(data=plot_df, x="test_label", y=label, color="black", alpha=0.35, size=2, ax=axes[i])
        value = icc.loc[icc["metric"] == metric, "icc"].iloc[0]
        axes[i].set_title(f"{label}, ICC2k={value:.2f}")
        axes[i].set_xlabel("")
        axes[i].tick_params(axis="x", rotation=30)
    sns.despine()
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig


def plot_strategy_counts(strategy_tables: dict[str, pd.DataFrame], path: str | Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, title in [
        (axes[0], "strategy_classification", "Attention Strategy"),
        (axes[1], "decision_process", "Decision Strategy"),
    ]:
        table = strategy_tables[key].melt(
            id_vars=["test", "test_label"], var_name="strategy_response", value_name="count"
        )
        sns.barplot(data=table, x="test_label", y="count", hue="strategy_response", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    sns.despine()
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig


def plot_distance_regression(grouped_distance: pd.DataFrame, path: str | Path):
    slopes = subject_distance_slopes(grouped_distance)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for test in [0, 1, 2]:
        line_ax = axes[0, test]
        hist_ax = axes[1, test]
        test_grouped = grouped_distance.loc[grouped_distance["test"] == test]
        for _, subject_group in test_grouped.groupby("subject_id"):
            sns.regplot(
                data=subject_group,
                x="distance",
                y="correct",
                ci=None,
                scatter=False,
                line_kws={"alpha": 0.25, "linewidth": 1},
                ax=line_ax,
            )
        sns.scatterplot(
            data=test_grouped.groupby("distance")["correct"].mean().reset_index(),
            x="distance",
            y="correct",
            color="black",
            ax=line_ax,
        )
        line_ax.set_title(TEST_LABELS[test])
        line_ax.set_xlabel("Feature Distance")
        line_ax.set_ylabel("P(Correct)")

        sns.histplot(data=slopes.loc[slopes["test"] == test], x="slope", bins=18, ax=hist_ax)
        hist_ax.axvline(0, color="black", linestyle="--", linewidth=1)
        hist_ax.set_xlabel("Subject Slope")
        hist_ax.set_ylabel("Count")
    sns.despine()
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig


def plot_questionnaire_covariance(corr: pd.DataFrame, path: str | Path):
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Questionnaire Subscale Correlations")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig


def plot_questionnaire_task_heatmap(correlations: pd.DataFrame, path: str | Path):
    plot_df = correlations.pivot(index="subscale", columns="metric_label", values="r")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(plot_df, cmap="coolwarm", center=0, vmin=-0.5, vmax=0.5, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Questionnaire Subscale vs Mean Task Metric Correlations")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig
