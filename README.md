# 🧠 Bank Marketing Predictor API

A modular **FastAPI** service for predicting whether a bank customer will subscribe to a term deposit — powered by multiple trained ML models (Logistic Regression, LightGBM, Random Forest, and Voting Classifier).

Built with production-style structure, class-based feature engineering, and versioned feature sets.

## How to run
1. Install dependencies:
```bash
    pip install -r requirements.txt
```

2. Start the api
```bash
    uvicorn src.api:app --reload
```
The server runs at: http://127.0.0.1:8000

3. Send a prediction request

```python
import requests, json

payload = {
    "age": 45,
    "job": "admin",
    "marital_status": "married",
    "education": "tertiary",
    "has_credit": "yes",
    "housing_loan": "no",
    "personal_loan": "no",
    "contact_mode": "cellular",
    "month": "may",
    "week_day": "friday",
    "previous_outcome": "unknown",
    "N_last_days": 999,
    "contacts_per_campaign": 1,
    "nb_previous_contact": 0,
    "emp_var_rate": 1.1,
    "cons_price_index": 93.2,
    "cons_conf_index": -40.0,
    "euri_3_month": 4.85,
    "nb_employees": 5228
}

response = requests.post(
    "http://127.0.0.1:8000/predict?model=logreg",
    data=json.dumps(payload),
    headers={"Content-Type": "application/json"}
)
print(response.json())
```

Or run the client/test_api.py script

## File overview
```bash
src/
├── models.py             # Loads trained models and runs predictions
└── api.py                # FastAPI endpoints
trained_models/           # Stored .pkl models
data/bank_dataset.csv     # Sample data
client/test_api.py        # Script for sending data requests
```

## Notes
The /predict endpoint accepts a query parameter model with values:
- logreg
- lgbm
- rf
- voting

Each model is trained on a different feature version defined in ModelManager.
