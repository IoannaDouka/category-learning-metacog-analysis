# Metacognition Drone-Recon Game: Analysis Code

Analysis code for the manuscript. This repository reproduces every statistic, table, and figure reported in the paper from the publicly available behavioral data. The task itself (stimulus presentation and data collection) lives in a separate repository.

## Contents

| File | Purpose |
|---|---|
| `paper_reproducibility_analysis.ipynb` | Main notebook. Runs every analysis in the paper top-to-bottom and writes all tables and figures to `paper_repro_outputs/`. |
| `paper_analysis_utils.py` | Analysis functions used by the notebook: data loading, sample definitions, task-metric summaries, statistical tests, and plotting. |
| `analysisfunctions.py` | Core metacognitive scoring functions (Brier score, calibration, resolution, outcome variance, converted calibration) based on the Murphy decomposition. |
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

## Analysis Overview

The notebook proceeds in order through:

1. Data loading and completer identification (`N=80` participants completed all three sessions)
2. Task-metric summaries per participant and session (accuracy, Brier score, calibration, resolution, outcome variance, rewarded score)
3. Sample definitions, including the strict above-chance sensitivity sample (`accuracy > .5` in all three sessions, `N=51`)
4. Accuracy-calibration correlations by session
5. Calibration change across sessions (repeated-measures ANOVA and paired t-test)
6. Test-retest reliability (`ICC2k`, all completers)
7. Strategy self-report counts
8. Feature-distance mixed-effects models (primary `N=80`; sensitivity `N=51`)
9. Questionnaire subscale correlation structure
10. Exploratory questionnaire-task correlations with Bonferroni and Benjamini-Hochberg FDR correction (66-test family)
11. Strict-sample sensitivity for the selected questionnaire correlations
12. Power and sample-size estimates for Pearson correlations
13. A summary table of key result checkpoints

### Confidence Scoring

Confidence is reported on a 4-key scale. Keys 1-4 map to the midpoints of four equal bins on the .5-1 probability range (.5625, .6875, .8125, .9375), fixed across participants and sessions. The Brier score is the trial-weighted mean squared error between the mapped confidence probability and the outcome, and satisfies the Murphy decomposition `PS = C - R + O` with the calibration, resolution, and outcome-variance functions in `analysisfunctions.py`.
