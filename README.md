# Chicago Crime Explorer

An end-to-end machine learning pipeline for predicting arrest likelihood and crime type from Chicago Police Department incident data. Built for CS 451 Introduction to Data Science, University of Alabama, Spring 2026.

## Project Overview

This project uses 520,417 incidents from the Chicago PD CLEAR system (2024-2026) to train classification models that predict:
- Whether an arrest will be made (binary classification)
- What type of crime occurred (multiclass classification across 16 categories)

The models are deployed in an interactive Streamlit application with two tabs:
- **Predict** — input incident details and get an arrest probability estimate
- **Neighborhood Explorer** — explore crime patterns across all 77 Chicago community areas

## Results

| Model | ROC-AUC | F1 Macro |
|---|---|---|
| Logistic Regression | 0.713 | 0.601 |
| Random Forest | 0.821 | 0.663 |
| XGBoost (default) | 0.830 | 0.669 |
| XGBoost (tuned) | 0.827 | 0.661 |

Best model (XGBoost) achieved ROC-AUC of 0.824 on the held-out test set.

## Dataset

Chicago Crime Dataset 2024-2026 from Kaggle:
https://www.kaggle.com/datasets/aliafzal9323/chicago-crime-dataset-2024-2026

Originally sourced from the Chicago Police Department CLEAR (Citizen Law Enforcement Analysis and Reporting) system.

## How to Run

### Option 1 - Google Colab (recommended)

1. Open the notebook in Google Colab
2. Run the kagglehub download cell to get the dataset
3. Run all cells top to bottom
4. The cleaned dataset will be saved to `/content/chicago_crimes_clean.parquet`
5. Trained models will be saved to `/content/` as `.pkl` files

### Option 2 - Run the Streamlit App

After running the full notebook and saving models to Google Drive:

1. Mount Google Drive and copy model files to `/content/`
2. Run the app.py creation cell
3. Run the ngrok launch cell
4. Open the printed URL in your browser

## Project Structure
chicago-crime-explorer/
├── FinalProj451.ipynb    # Main notebook with full pipeline
├── README.md             # This file
└── requirements.txt      # Python dependencies
## Requirements

See `requirements.txt` for full list. Key dependencies:
- pandas, numpy, matplotlib, seaborn
- scikit-learn, xgboost, shap
- streamlit, pyngrok, kagglehub

## Key Design Decisions

- **Time-based split** — 2024 train, Jan-Sep 2025 validation, Oct 2025-2026 test to prevent data leakage
- **Four-tier severity feature** — VIOLENT_SEVERE, VIOLENT_MODERATE, PROPERTY, OTHER
- **Target-encoded features** — location risk score and district arrest rate based on historical arrest patterns
- **Class imbalance handling** — scale_pos_weight in XGBoost to account for 15% arrest rate

## Limitations

- The arrest column measures on-scene apprehension only, not case clearance
- Target-encoded features should be recomputed inside CV folds in production
- Location type and district are correlated features
- Crime type model struggles with rare categories

## Author

Lily Oswald  
University of Alabama, CS 451 Spring 2026  
Loswald@crimson[.]ua.edu
