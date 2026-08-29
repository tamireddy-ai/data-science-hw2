# Data Science Homework 2 - Regression and Classification Error Analysis

This project studies where predictive models fail on an industrial LiFePO4 battery dataset. It contains a reproducible Jupyter notebook with out-of-fold predictions, model comparisons, detailed regression residual analysis, classification error analysis, threshold sensitivity, and a final reflection.

## Main questions

- **Regression:** How accurately can battery capacity be estimated from cycle and operational measurements?
- **Classification:** Can the same measurements identify a degraded battery state, defined as `SOH < 80%`?
- **Error analysis:** Which observations and operating regions produce systematic or high-confidence failures?

## Models

- Regression: Linear Regression, Decision Tree Regressor, Random Forest Regressor
- Classification: Logistic Regression, Decision Tree Classifier, Random Forest Classifier

All reported predictions are out-of-fold predictions from 5-fold cross-validation. The notebook explains the choice of `k`, uses identical features for fair comparisons, and documents every hyperparameter.

## Leakage controls

The raw dataset contains deterministic relationships: `SOH = Capacity / 40 * 100`, `RUL = 3000 - CycleNumber`, and `DoD = CycleNumber / 30`. Therefore `SOH`, `RUL`, and `DoD` are excluded from model features. `Capacity` is also excluded from the classification features because it directly defines `SOH` and thus the class label. `CycleNumber` is retained as an operational time feature, and its strong proxy relationship with degradation is discussed as a limitation.

## Project structure

```text
data-science-hw2/
├── data/raw/industrial_lifepo4_bms.csv
├── notebooks/homework2_error_analysis.ipynb
├── build_notebook.py
├── README.md
└── requirements.txt
```

The CSV is intentionally ignored by Git because it is source data. Dataset source: [Mendeley Data - Industrial LiFePO4 Battery Management System Dataset](https://data.mendeley.com/datasets/k675pxrd83/1).

## Run locally

1. Create and activate a Python virtual environment.
2. Install the dependencies with `pip install -r requirements.txt`.
3. Download the dataset and save the selected 3,000-cycle CSV as `data/raw/industrial_lifepo4_bms.csv`.
4. Open `notebooks/homework2_error_analysis.ipynb` and run all cells from top to bottom.

The notebook is designed to run without hidden state and uses a fixed random seed for reproducibility.

`build_notebook.py` regenerates the clean notebook structure if needed; the submitted notebook already contains the verified execution outputs.
