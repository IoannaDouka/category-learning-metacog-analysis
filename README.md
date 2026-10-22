# Metacognition Drone-Recon Game: Analysis Code

Analysis code for the manuscript, using the publicly available behavioral data. Reviewer-related corrections and replacement manuscript wording are documented in [REVIEWER_REVISIONS.md](REVIEWER_REVISIONS.md). The task itself (stimulus presentation and data collection) lives in a separate repository.

## Contents

| File | Purpose |
|---|---|
| `paper_reproducibility_analysis.ipynb` | Main notebook. Runs every analysis in the paper top-to-bottom and writes all tables and figures to `paper_repro_outputs/`. |
| `paper_analysis_utils.py` | Analysis functions used by the notebook: data loading, sample definitions, task-metric summaries, statistical tests, and plotting. |
| `analysisfunctions.py` | Core metacognitive scoring functions (Brier score, calibration, resolution, outcome variance, converted calibration) based on the Murphy decomposition. |
| `REVIEWER_REVISIONS.md` | Strategy model correction, ICC interpretation, and testing-only feature-distance results. |
| `tests/test_reviewer_corrections.py` | Checks for dependence between sessions, model identifiability, ICC reporting, and trial inclusion. |
| `data/` | Place the downloaded behavioral data here (see below). |

## Data

The behavioral data are available on Zenodo: [https://zenodo.org/records/10155670](https://zenodo.org/records/10155670) (DOI: `10.5281/zenodo.10155670`).

Download all six CSV files into `data/`:

- `task_data_initial_test.csv`
- `task_data_retest_1.csv`
- `task_data_retest_2.csv`
- `questionnaire_data.csv`
- `questionnaire_overall_scores.csv`
- `questionnaire_subscale_scores.csv`

## Setup

Requires Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running The Analysis

Open and run the notebook in Jupyter, or execute it headlessly:

```bash
jupyter execute paper_reproducibility_analysis.ipynb
```

All outputs are written to `paper_repro_outputs/`:

- `paper_repro_outputs/tables/` — CSV tables for every reported statistic
- `paper_repro_outputs/figures/` — SVG figures
- `paper_repro_outputs/supplement/` — supplementary materials as US-letter PDF pages (Supplemental Table 1, Supplemental Figures 1-6)

## Analysis Overview

The notebook proceeds in order through:

1. Data loading and completer identification (`N=80` participants completed all three sessions)
2. Task-metric summaries per participant and session (accuracy, Brier score, calibration, resolution, outcome variance, rewarded score)
3. Sample definitions, including the strict above-chance sensitivity sample (`accuracy > .5` in all three sessions, `N=51`)
4. Accuracy-calibration correlations by session
5. Calibration change across sessions (repeated-measures ANOVA and paired t-test)
6. Reliability of the mean across three administrations (`ICC2k`, two-way random effects, absolute agreement, with 95% confidence intervals; all completers)
7. Strategy self-report counts
8. Feature-distance mixed-effects models using main-task testing trials only (primary `N=80`; sensitivity `N=51`; 42 distinct trials per participant per session)
9. Questionnaire subscale correlation structure
10. Exploratory questionnaire-task correlations with Bonferroni and Benjamini-Hochberg FDR correction (66-test family)
11. Strict-sample sensitivity for the selected questionnaire correlations
12. Power and sample-size estimates for Pearson correlations
13. A summary table of key result checkpoints
14. Supplementary materials, saved as US-letter PDF pages: Supplemental Table 1 (participant random-intercept strategy models), Supplemental Figure 1 (performance by reported strategy), and Supplemental Figures 2-6 (all questionnaire subscale by task metric correlations, split by instrument). Strategy cell counts, matrix ranks, coefficients, fitting diagnostics, and two-way interaction sensitivity tests are also exported.

### Corrected Strategy Analysis

The primary models include additive categorical terms for classification strategy, decision strategy, and session, with a participant random intercept. A sensitivity model includes all two-way interactions. The original full factorial model is rank deficient (24 independent columns out of 27), so it is not used for inference. The singleton `Wrote down` report is excluded from these models; that exclusion alone does not resolve the empty cells in the full factorial design.

`supplemental_table_1_strategy_mixed_models.csv` replaces the old OLS ANOVA. It reports asymptotic Wald chi-square tests and both unadjusted and Benjamini-Hochberg-adjusted P values across the 18 primary tests. Interaction sensitivity tests use a separate 18-test family. These are exploratory associations. The superseded ANOVA is retained only under `paper_repro_outputs/archive/`; its P values must not be used for the revised manuscript.

Run the focused regression checks with:

```bash
python -m unittest discover -s tests -v
```

### Feature-Distance Trial Inclusion

Feature-distance summaries and models include only rows with `use == "task"` and `block == "test"`. Training and tutorial trials are excluded before aggregation. Each session's participant random-intercept model uses mean accuracy per participant and feature distance. Model tables include the selected block, participant and trial counts, 95% slope intervals, and convergence status. The previous combined training/testing tables are retained under `paper_repro_outputs/archive/` solely to document this correction.

### Confidence Scoring

Confidence is reported on a 4-key scale. Keys 1-4 map to the midpoints of four equal bins on the .5-1 probability range (.5625, .6875, .8125, .9375), fixed across participants and sessions. The Brier score is the trial-weighted mean squared error between the mapped confidence probability and the outcome, and satisfies the Murphy decomposition `PS = C - R + O` with the calibration, resolution, and outcome-variance functions in `analysisfunctions.py`.
