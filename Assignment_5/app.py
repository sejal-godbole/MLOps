import uvicorn
import mlflow
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

# --- Configuration ---
# NOTE: Replace with your actual Registered Model Name from Assignment 4
MODEL_NAME = "IrisElasticNetPredictor" 

# Set the MLflow tracking URI used for the Model Registry (e.g., Databricks)
# If using a local MLflow server, use "http://127.0.0.1:5000"
# For this example, we assume local testing or a configured remote.
mlflow.set_tracking_uri("databricks") 

# --- FastAPI Setup ---
app = FastAPI(
    title="MLflow Model Service",
    description="ElasticNet model prediction API, loading the Production version from MLflow.",
    version="1.0.0"
)

# --- Pydantic Schema for Request Body ---
# Based on the features of the Iris dataset used in Assignment 2:
class IrisFeatures(BaseModel):
    """Defines the input features for prediction."""
    sepal_length: float = 5.1
    sepal_width: float = 3.5
    petal_length: float = 1.4
    petal_width: float = 0.2

# Global variable to hold the loaded model
model = None

@app.on_event("startup")
def load_production_model():
    """
    On application startup, fetch and load the model currently tagged as 'Production'.
    """
    global model
    try:
        # Load the model directly from the Model Registry using the stage tag
        model_uri = f"models:/{MODEL_NAME}/Production"
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Successfully loaded Production model: {MODEL_NAME}")
    except Exception as e:
        print(f"ERROR: Could not load model from MLflow Registry. Check configuration and model stage.")
        print(f"Details: {e}")
        # Use a dummy model if loading fails for local testing/graceful degradation
        class DummyModel:
             def predict(self, X):
                # Return a default/safe output shape
                return np.array([0.0]) 
        model = DummyModel()

@app.get("/", summary="Health Check")
def read_root():
    """Simple health check endpoint."""
    return {"status": "ok", "model_name": MODEL_NAME, "stage": "Production"}

@app.post("/predict", summary="Get model prediction")
def predict_endpoint(features: IrisFeatures):
    """
    Accepts Iris features and returns the model's regression prediction.
    """
    if model is None:
        return {"error": "Model not loaded on server startup."}, 500

    # Convert Pydantic model to a 2D NumPy array for prediction
    data_point = np.array([
        [
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]
    ])

    # Make prediction using the loaded ElasticNet model
    prediction = model.predict(data_point)[0]
    
    # The ElasticNet model on Iris predicts a float (target species label index)
    return {
        "model_name": MODEL_NAME,
        "model_stage": "Production",
        "input_features": features.dict(),
        "predicted_target_index": float(prediction)
    }

if __name__ == "__main__":
    # Running via uvicorn for development purposes
    uvicorn.run(app, host="0.0.0.0", port=8000)
