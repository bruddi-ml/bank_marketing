import pandas as pd
import requests
import json

def main():

    # --- Config ---
    api_url = "http://127.0.0.1:8000/predict"
    models = ["logreg", "lgbm", "rf", "voting"]  # must match /predict?model= param names
    csv_file = "data/bank_dataset.csv"

    # --- Load sample data ---
    df = pd.read_csv(csv_file)
    sample_data = df.drop(columns=["target"]).head(10)

    # --- Iterate through samples ---
    for i, row in sample_data.iterrows():
        payload = row.to_dict()
        payload_json = json.dumps(payload)

        print(payload)
        for model_name in models:
            try:
                response = requests.post(
                    f"{api_url}?model={model_name}",
                    data=payload_json,
                    headers={"Content-Type": "application/json"}
                )

                if response.status_code == 200:
                    result = response.json()
                    print(f"{model_name} → Prediction: {result['prediction']}, "
                          f"Probability: {result['probability']}")
                else:
                    print(f"{model_name} → Error {response.status_code}: {response.text}")
            except Exception as e:
                print(f"{model_name} → Request failed: {e}")

main()
