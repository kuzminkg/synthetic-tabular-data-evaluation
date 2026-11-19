# Synthetic Replacement Experiments

This repository evaluates whether simple synthetic tabular data generators can replace or augment real training observations when performance is measured on real unseen data.

The core question is:

> If a generator is fitted only on a real training fold, can a classifier trained on synthetic or real-plus-synthetic rows match or improve a classifier trained on the corresponding real rows?

The package supports four public binary-classification datasets, three generator families, two classifier families, multiple real-data fractions, class-conditional generation, and minority-only sensitivity experiments.

## Current status

This is a clean reproducibility baseline. It intentionally does not include the legacy result folders or manuscript tables because the audited implementation required several corrections that can change the numerical findings. Publication results should be regenerated with this version before the paper is revised.

Important safeguards in this version include:

- the development/holdout split is created once per dataset;
- every validation fold remains real and unseen by preprocessing and generators;
- training fractions are sampled only from the training side of each cross-validation split;
- target-order encoding is cross-fitted for outer-training rows;
- features are standardized inside each outer training fold before distance-based generation;
- SMOTE-style interpolation uses one scalar interpolation coefficient per synthetic row;
- minority-only regimes are defined relative to the real minority count, not the total fold size;
- repeated-cross-validation comparisons use a corrected resampled t-test;
- Holm adjustment spans the complete searched family within each dataset, model, synthetic mode, and encoding;
- final holdout comparisons use a matched Real-only model at the same real-data fraction and paired class-stratified bootstrap intervals.

See `docs/FINISHING_PLAN.md` for the recommended sequence of remaining work and `docs/MANUSCRIPT_AUDIT.md` for the manuscript issues that should be addressed after the experiments are rerun.

## Repository structure

```text
.
├── README.md
├── pyproject.toml
├── requirements.txt
├── run_experiments.py
├── data/
│   └── README.md
├── docs/
│   ├── FINISHING_PLAN.md
│   └── MANUSCRIPT_AUDIT.md
├── results/
│   └── README.md
├── scripts/
│   ├── evaluate_holdout.py
│   ├── make_latex_tables.py
│   └── plot_results.py
├── src/
│   └── synthrep/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── data.py
│       ├── experiment.py
│       ├── generators.py
│       ├── holdout.py
│       ├── models.py
│       ├── plotting.py
│       ├── preprocessing.py
│       ├── registry.py
│       └── stats.py
└── tests/
```

## Installation

### Using uv

```bash
uv venv
uv sync --extra dev
```

Activate the environment if needed:

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
source .venv/bin/activate
```

### Using pip

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The editable install creates the command-line program `synthrep`.

## Prepare the data

Place the four raw CSV files in one local folder:

```text
data/
  cs-training.csv
  UCI_Credit_Card.csv
  adult.csv
  HTRU_2.csv
```

`HTRU_2.csv` should be the original headerless file. Raw data are intentionally excluded from Git. See `data/README.md` for details.

## Run the tests

```bash
python -m pytest
```

## Run a smoke test

```bash
python run_experiments.py \
  --data-dir ./data \
  --out-dir results/smoke \
  --quick
```

The smoke test verifies the complete pipeline but is not suitable for the paper.

## Main logistic-regression experiment

```bash
python run_experiments.py \
  --data-dir ./data \
  --out-dir results/main_logreg \
  --datasets give_me_credit default_credit adult htru2 \
  --models logreg \
  --generators bootstrap smote copula \
  --training-fractions 0.10 0.25 0.50 1.00 \
  --mix-amounts 1 3 \
  --n-splits 5 \
  --n-repeats 10
```

## Random-forest robustness experiment

```bash
python run_experiments.py \
  --data-dir ./data \
  --out-dir results/random_forest \
  --datasets give_me_credit default_credit adult htru2 \
  --models random_forest \
  --generators bootstrap smote copula \
  --training-fractions 0.10 0.25 0.50 1.00 \
  --mix-amounts 1 3 \
  --n-splits 5 \
  --n-repeats 10
```

## Minority-only sensitivity experiment

```bash
python run_experiments.py \
  --data-dir ./data \
  --out-dir results/minority_only \
  --synthetic-mode minority_only \
  --datasets give_me_credit default_credit adult htru2 \
  --models logreg \
  --generators bootstrap smote copula \
  --training-fractions 0.10 0.25 0.50 1.00 \
  --mix-amounts 1 3 \
  --n-splits 5 \
  --n-repeats 10
```

In minority-only mode the regimes are:

- `Real-only`: all real training rows;
- `Minority-replaced`: real majority rows plus a synthetic minority sample equal in size to the original minority sample;
- `Minority +1x`: all real rows plus one original-minority-count of synthetic minority rows;
- `Minority +3x`: all real rows plus three original-minority-counts of synthetic minority rows.

## Ordinal-encoding sensitivity experiment

Only Adult and Default Credit contain categorical processing in the current design:

```bash
python run_experiments.py \
  --data-dir ./data \
  --out-dir results/ordinal_encoding \
  --categorical-encoding ordinal \
  --datasets default_credit adult \
  --models logreg \
  --generators bootstrap smote copula \
  --training-fractions 0.10 0.25 0.50 1.00 \
  --mix-amounts 1 3 \
  --n-splits 5 \
  --n-repeats 10
```

## Output files

Each development run writes:

- `fold_scores.csv`: split-level scores and sample-size metadata;
- `summary_scores.csv`: descriptive means and standard deviations;
- `paired_tests_vs_real.csv`: corrected paired differences, confidence intervals, nominal p-values, and Holm-adjusted p-values;
- `run_manifest.json`: exact configuration, environment versions, dataset metadata, and SHA-256 hashes.

## Final holdout evaluation

Run the holdout script only after the development grid and selection rule are frozen:

```bash
python scripts/evaluate_holdout.py \
  --data-dir ./data \
  --summary results/main_logreg/summary_scores.csv \
  --manifest results/main_logreg/run_manifest.json \
  --out-dir results/main_logreg/holdout \
  --bootstrap-resamples 5000
```

It writes:

- `selected_development_configs.csv`;
- `holdout_comparisons.csv` with matched Real-only scores and paired bootstrap intervals;
- `holdout_predictions.csv.gz` with row-level paired predictions.
