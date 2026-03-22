# Inventory Analysis Project

## Overview

This project combines two machine learning workflows around vendor invoice operations:

1. **Freight Cost Prediction**
   Predicts expected freight cost from invoice spending patterns.
2. **Invoice Manual Approval Flagging**
   Predicts whether an invoice should be escalated for manual review based on invoice and purchase-level features.

The repository also includes a Streamlit application that exposes both workflows through a single internal dashboard.

---

## Main Capabilities

- Train and evaluate a regression model for freight cost estimation.
- Train and evaluate a classification model for invoice approval risk.
- Load data from a local SQLite database.
- Save trained models to the top-level `models/` directory.
- Run an interactive Streamlit UI for internal analysis and decision support.

---

## Project Structure

```text
Inventory Analysis Project/
|-- app.py
|-- README.md
|-- .gitignore
|-- data/
|   `-- inventory.db
|-- models/
|   |-- predict_freight_model.pkl
|   |-- predict_flag_invoice.pkl
|   `-- scaler.pkl
|-- inference/
|   |-- predict_freight.py
|   `-- predict_invoice_flag.py
|-- freight_cost_prediction/
|   |-- data_preprocessing.py
|   |-- modeling_evaluation.py
|   `-- train.py
|-- invoice_flagging/
|   |-- data_preprocessing.py
|   |-- modeling_evaluation.py
|   `-- train.py
`-- notebooks/
    |-- Predict Freight Cost.ipynb
    `-- Invoice Flagging.ipynb
```

---

## Workflow Summary

### 1. Freight Cost Prediction

This pipeline:

- Loads vendor invoice data from the `vendor_invoice` table in `data/inventory.db`
- Selects model features in `freight_cost_prediction/data_preprocessing.py`
- Trains multiple regression models:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
- Evaluates models using:
  - MAE
  - RMSE
  - R-squared
- Saves the best performing model as `models/predict_freight_model.pkl`

### 2. Invoice Manual Approval Flagging

This pipeline:

- Loads invoice and purchase-derived features from the SQLite database
- Builds a label called `flag_invoice` using rule-based heuristics
- Splits data into training and testing sets
- Scales numeric features with `StandardScaler`
- Tunes a `RandomForestClassifier` using `GridSearchCV`
- Evaluates classification output using:
  - Accuracy
  - Classification report
  - F1-driven model search
- Saves the classifier as `models/predict_flag_invoice.pkl`
- Saves the scaler as `models/scaler.pkl`

---

## Data Source

The project reads from:

- `data/inventory.db`

Based on the training code, the database contains at least these relevant tables:

- `vendor_invoice`
- `purchases`

The invoice flagging workflow uses SQL aggregation logic to combine invoice data with purchase-level summaries such as:

- total item quantity
- total item dollars
- total brands
- average receiving delay

---

## Streamlit Application

The Streamlit app is located at:

- `app.py`

It exposes two UI modules:

1. **Freight Cost Prediction**
2. **Invoice Manual Approval Flag**

### Run the app

```bash
streamlit run app.py
```

---

## Local Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
```

### 2. Activate it

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

There is currently no `requirements.txt` in the repository, so install the core packages manually:

```bash
pip install streamlit pandas numpy scikit-learn joblib matplotlib jupyter
```

---

## Training the Models

### Freight model

Run from the `freight_cost_prediction` directory:

```bash
python train.py
```

This script:

- loads invoice data
- prepares features
- trains three regression models
- compares them by MAE
- saves the best model

### Invoice flagging model

Run from the `invoice_flagging` directory:

```bash
python train.py
```

This script:

- loads invoice and purchase data
- creates the target label
- scales features
- performs grid search for a random forest classifier
- saves both the classifier and scaler

---

## Inference Files

### Freight inference

File:

- `inference/predict_freight.py`

Purpose:

- loads the saved freight model
- accepts input data as a dictionary
- returns a DataFrame with `Predicted_Freight`

### Invoice flag inference

File:

- `inference/predict_invoice_flag.py`

Purpose:

- intended to provide invoice flag prediction support

Note:

- the current file does not yet expose the expected `predict_invoice_flag` helper used by the original UI code
- the Streamlit app has been written to remain usable without modifying other project files

---

## Expected Model Inputs

### Freight model input

The currently saved freight model is trained on:

- `Dollars`

### Invoice flagging model inputs

The classifier is trained on:

- `invoice_quantity`
- `invoice_dollars`
- `Freight`
- `total_item_quantity`
- `total_item_dollars`

---

## Outputs

### Freight prediction output

- estimated freight cost

### Invoice flagging output

- predicted flag indicating whether manual approval is recommended

---

## Known Limitations

- Some training scripts use absolute Windows paths such as `H:/ML Projects/Inventory Analysis Project/...`
- The repository does not currently include a dependency lock file such as `requirements.txt`
- `inference/predict_invoice_flag.py` is incomplete relative to the original UI import expectation
- The freight model training code currently uses `Dollars` only, even though quantity may be operationally useful context
- Model files are stored locally, so the app depends on those artifacts being present in `models/`

---

## Suggested Next Improvements

- add a `requirements.txt` or `pyproject.toml`
- convert absolute database paths to configurable relative paths
- complete and standardize the invoice inference helper
- add unit tests for inference and preprocessing functions
- add model versioning and metrics logging
- add input validation and error tracking for the Streamlit app

---

## Quick Start

If you already have the database and trained models in place:

```bash
pip install streamlit pandas numpy scikit-learn joblib matplotlib jupyter
streamlit run app.py
```

---

## Authoring Notes

This repository appears to be designed as an internal ML analytics tool that combines:

- data extraction from SQLite
- feature engineering
- model training and evaluation
- model serialization
- dashboard-based inference

It is a strong foundation for a lightweight finance and supply-chain decision support application.
