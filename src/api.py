from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from typing import Literal

from src.models import FeatureEngineer

# ------------------------------
# MODEL MANAGER
# ------------------------------
# Encapsulates model loading, feature engineering, and prediction logic
class ModelManager:
    def __init__(self, df, feature_version:str):
        # Version control for feature sets (different preprocessing logic)
        self.feature_version = feature_version

        # Mapping between model names and their saved .pkl files
        self.model_paths = {
            'logreg': "trained_models\\model_logreg.pkl",
            'lgbm': "trained_models\\model_lgbm.pkl",
            'rf': "trained_models\\model_rand_forest.pkl",
            'voting': "trained_models\\model_voting_classifier.pkl"
        }
        self.feature_sets = {}
        self.df = df.copy()

    def load_model(self, model_name:str):
        # Load serialized model object from disk based on model name
        if model_name not in self.model_paths:
            raise ValueError(f"Unknown model name: {model_name}")
        return joblib.load(self.model_paths[model_name])

    def feature_engineering(self):
        # Initialize FeatureEngineer to ensure consistent transformations
        fe = FeatureEngineer(self.df)
        df = fe.add_features(flag=1)    # Add engineered column
        self.df = df.copy()
        self.feature_sets = fe.generate_feature_sets() # Track available feature versions
        return self.df

    def predict(self, model_name):
        # Perform full inference pipeline: load transform predict
        model = self.load_model(model_name)
        transformed_df = self.feature_engineering()
        prob = model.predict_proba(transformed_df)[0, 1]
        pred = int(prob >= 0.5)
        return {"model": model_name, "prediction": pred, "probability": round(prob, 4)}


# ------------------------------
# Pydantic Model Generator
# ------------------------------
# Dynamically infers schema for FastAPI validation based on dataset columns
def generate_pydantic_class(csv_file, class_name: str = "CustomerData"):
    """
    Dynamically build a Pydantic model based on dataset schema.
    """
    df = pd.read_csv(csv_file)
    df = df.drop(columns=["target"])

    type_mapping = {
        "int64": int,
        "float64": float,
        "bool": bool,
        "object": str,
        "category": str
    }

    annotations = {col: type_mapping.get(str(df[col].dtype), str) for col in df.columns}
    class_dict = {"__annotations__": annotations}
    return type(class_name, (BaseModel,), class_dict)



app = FastAPI(title="Bank Marketing Predictor", version="2.0")
csv_file = "data\\bank_dataset.csv"
df = pd.read_csv(csv_file)
CustomerData = generate_pydantic_class(csv_file)

# --- ENDPOINTS ---

@app.post("/predict")
def predict_model(
    data: CustomerData,
    model: Literal["logreg", "lgbm", "rf", "voting"] = "logreg"
):
    """
    Universal prediction endpoint — choose model via query param or default to Logistic Regression.
    Example: POST /predict?model=lgbm
    """
    input_df = pd.DataFrame([data.dict()])
    if model == "logreg":
        manager = ModelManager(input_df, feature_version="v3")
    elif model == "lgbm":
        manager = ModelManager(input_df, feature_version="11.0")
    elif model == "rf":
        manager = ModelManager(input_df, feature_version="v9") 
    elif model == "voting":
        manager = ModelManager(input_df, feature_version="v9")

    result = manager.predict(model_name=model)
    return result

@app.get("/")
def home():
    return {"message": "🚀 Bank Marketing Predictor API is alive!"}


    
