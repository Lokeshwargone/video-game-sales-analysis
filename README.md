# Video Game Sales Analysis

A reproducible analysis of historical video game sales, regional performance, and the extent to which pre-release metadata can predict broad sales tiers.

## Dataset

Use the Kaggle **Video Game Sales** dataset (commonly distributed as `vgsales.csv`, with roughly 16,500 titles). Place the CSV at `data/raw/vgsales.csv`. The project expects these fields: `Name`, `Platform`, `Year`, `Genre`, `Publisher`, `NA_Sales`, `EU_Sales`, `JP_Sales`, `Other_Sales`, and `Global_Sales`. Sales are represented in millions of units in the commonly used source file.

The raw data is not included. Keep the dataset's source and licensing terms with your copy when distributing results.

## Workflow

Run the notebooks in order:

1. `notebooks/01_data_cleaning.ipynb` validates the input, standardizes labels and types, fills missing publishers with `Unknown`, median-imputes missing or invalid years, removes duplicate rows, and writes `data/processed/vgsales_clean.csv`.
2. `notebooks/02_eda.ipynb` analyzes yearly, genre, platform, regional, publisher, and feature-correlation patterns. Its figures are saved under `outputs/figures/`.
3. `notebooks/03_modeling.ipynb` derives console-generation labels and sales tiers, then evaluates Logistic Regression and Random Forest baselines with stratified cross-validation and a held-out test set.

Reusable functions live in `src/`. Model preprocessing is fit within each training fold, and predictors are restricted to `Platform`, `Genre`, `Publisher`, and `Year`; regional and global sales are used only to construct the target, avoiding target leakage. The classifiers use class weighting and macro-averaged precision, recall, and F1 to account for class imbalance.

## Key Findings

The notebooks compute the findings from the supplied CSV rather than hard-coding results. Review the year, genre, platform, region, publisher, and correlation figures in `outputs/figures/`, along with the printed model metrics and confusion matrices in the modeling notebook. Historical sales reflect the dataset's coverage and should not be interpreted as current market performance or causal evidence.

## Setup and Run

From the repository root:

```bash
python -m pip install -r requirements.txt
jupyter lab
```

Place `vgsales.csv` in `data/raw/`, then open and run the three notebooks in order. Cleaned data and generated figures are reproducible outputs and can be regenerated at any time.

## Project Layout

```text
data/
	raw/vgsales.csv
	processed/vgsales_clean.csv
notebooks/
	01_data_cleaning.ipynb
	02_eda.ipynb
	03_modeling.ipynb
outputs/figures/
src/
	data_cleaning.py
	features.py
	model.py
requirements.txt
```
