"""
FastAPI Model Serving Application
Loads a trained scikit-learn model and exposes prediction endpoint.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import joblib
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === ML Lifecycle Stage: DEPLOYMENT ===
# This API sits between the trained model artifact and downstream consumers
# Input: Raw features from client → Model: Loaded artifact → Output: Predictions

# Load model at startup (deterministic loading - happens once, not per request)
logger.info("Loading model artifact at startup...")
try:
    model = joblib.load('model.pkl')
    logger.info("✓ Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

# Initialize FastAPI app
app = FastAPI(
    title="ML Model Serving API",
    description="Iris classification model serving via FastAPI",
    version="1.0.0"
)

# === Pydantic Request Model ===
class PredictRequest(BaseModel):
    """Input schema for prediction requests."""
    features: List[float] = Field(
        ..., 
        min_length=4, 
        max_length=4,
        description="4 features: sepal_length, sepal_width, petal_length, petal_width",
        example=[5.1, 3.5, 1.4, 0.2]
    )

    class Config:
        json_schema_extra = {
            "example": {
                "features": [5.1, 3.5, 1.4, 0.2]
            }
        }

# === Pydantic Response Model ===
class PredictResponse(BaseModel):
    """Output schema for prediction responses."""
    prediction: int = Field(..., description="Predicted class: 0=setosa, 1=versicolor, 2=virginica")
    prediction_label: str = Field(..., description="Human-readable class name")
    confidence: float = Field(..., description="Prediction confidence (max probability)")
    model_version: str = Field(default="1.0.0", description="Model version identifier")

    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 0,
                "prediction_label": "setosa",
                "confidence": 0.98,
                "model_version": "1.0.0"
            }
        }

# === Health Check Endpoint ===
@app.get("/")
def health_check():
    """Health check endpoint to verify service is running."""
    return {
        "status": "healthy",
        "service": "ML Model Serving API",
        "model_loaded": model is not None
    }

# === Prediction Endpoint ===
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Make prediction using loaded model.
    
    Lifecycle Position:
    - Input: Client sends features
    - Processing: Model inference using pre-loaded artifact
    - Output: Structured prediction response
    
    Monitoring Touchpoints:
    - Log request count
    - Track prediction latency
    - Monitor prediction distribution
    - Alert on inference errors
    """
    try:
        # Convert input to numpy array
        features = np.array(request.features).reshape(1, -1)
        
        # Validate input shape
        if features.shape[1] != 4:
            raise HTTPException(
                status_code=422,
                detail=f"Expected 4 features, got {features.shape[1]}"
            )
        
        # Make prediction (model already loaded at startup)
        prediction = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]
        confidence = float(max(probabilities))
        
        # Map prediction to label
        class_names = {0: "setosa", 1: "versicolor", 2: "virginica"}
        prediction_label = class_names.get(prediction, "unknown")
        
        logger.info(f"Prediction: {prediction} ({prediction_label}), Confidence: {confidence:.2f}")
        
        return PredictResponse(
            prediction=prediction,
            prediction_label=prediction_label,
            confidence=confidence,
            model_version="1.0.0"
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction")

# === Additional Endpoint: Model Info ===
@app.get("/model-info")
def model_info():
    """Get information about the loaded model."""
    return {
        "model_type": str(type(model).__name__),
        "model_version": "1.0.0",
        "input_features": 4,
        "output_classes": 3,
        "class_names": ["setosa", "versicolor", "virginica"]
    }