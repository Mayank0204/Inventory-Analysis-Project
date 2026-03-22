import joblib
import pandas as pd
from pathlib import Path

# Setup paths relative to the file's location
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "predict_flag_invoice.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars",
]

def predict_invoice_flag(input_data: dict) -> pd.DataFrame:
    """
    Predicts if a vendor invoice requires manual approval.
    
    Args:
        input_data: Dictionary containing list values for features.
    
    Returns:
        DataFrame with features and 'Predicted_Flag'.
    """
    # Load assets
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    # Prepare data
    input_df = pd.DataFrame(input_data)
    
    # Scale features
    X = input_df[FEATURES]
    X_scaled = scaler.transform(X)
    
    # Predict
    input_df["Predicted_Flag"] = model.predict(X_scaled)
    
    return input_df

if __name__ == "__main__":
    # Test data
    sample_data = {
        "invoice_quantity": [50],
        "invoice_dollars": [352.95],
        "Freight": [1.73],
        "total_item_quantity": [162],
        "total_item_dollars": [2476.0],
    }
    
    if MODEL_PATH.exists() and SCALER_PATH.exists():
        prediction = predict_invoice_flag(sample_data)
        print("Test Prediction Results:")
        print(prediction)
    else:
        print("Model assets not found. Run training script first.")