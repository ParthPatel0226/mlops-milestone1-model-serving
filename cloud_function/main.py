"""
Google Cloud Function for ML Model Serving
Serverless inference endpoint with model loading optimization
"""
import functions_framework
from google.cloud import storage
import joblib
import numpy as np
import json
import os

# Global variable for model caching (loaded once per instance)
model = None

def load_model():
    """
    Load model from Cloud Storage or local file.
    Called once per cold start to minimize latency.
    """
    global model
    
    if model is not None:
        return model
    
    # For local testing or if model is deployed with function
    local_model_path = 'model.pkl'
    
    if os.path.exists(local_model_path):
        print("Loading model from local file...")
        model = joblib.load(local_model_path)
        print("✓ Model loaded successfully")
    else:
        # Alternative: Load from Cloud Storage (for production)
        # bucket_name = os.environ.get('MODEL_BUCKET', 'your-model-bucket')
        # blob_name = 'models/model.pkl'
        # storage_client = storage.Client()
        # bucket = storage_client.bucket(bucket_name)
        # blob = bucket.blob(blob_name)
        # blob.download_to_filename('/tmp/model.pkl')
        # model = joblib.load('/tmp/model.pkl')
        raise FileNotFoundError("Model file not found")
    
    return model

@functions_framework.http
def predict(request):
    """
    HTTP Cloud Function for model predictions.
    
    Args:
        request (flask.Request): HTTP request object
        
    Returns:
        JSON response with prediction results
    """
    # Set CORS headers for cross-origin requests
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST',
        'Access-Control-Allow-Headers': 'Content-Type',
    }
    
    # Handle OPTIONS request for CORS preflight
    if request.method == 'OPTIONS':
        return ('', 204, headers)
    
    # Only accept POST requests
    if request.method != 'POST':
        return (
            json.dumps({'error': 'Method not allowed. Use POST.'}),
            405,
            headers
        )
    
    try:
        # Load model (cached after first invocation in this instance)
        model = load_model()
        
        # Parse request body
        request_json = request.get_json(silent=True)
        
        if not request_json:
            return (
                json.dumps({'error': 'Request body must be JSON'}),
                400,
                headers
            )
        
        # Validate input
        if 'features' not in request_json:
            return (
                json.dumps({'error': 'Missing "features" field in request'}),
                422,
                headers
            )
        
        features = request_json['features']
        
        # Validate features length
        if not isinstance(features, list) or len(features) != 4:
            return (
                json.dumps({
                    'error': 'Features must be a list of 4 numbers',
                    'example': {'features': [5.1, 3.5, 1.4, 0.2]}
                }),
                422,
                headers
            )
        
        # Convert to numpy array
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = int(model.predict(features_array)[0])
        probabilities = model.predict_proba(features_array)[0]
        confidence = float(max(probabilities))
        
        # Map prediction to label
        class_names = {0: "setosa", 1: "versicolor", 2: "virginica"}
        prediction_label = class_names.get(prediction, "unknown")
        
        # Build response
        response = {
            'prediction': prediction,
            'prediction_label': prediction_label,
            'confidence': round(confidence, 4),
            'model_version': '1.0.0',
            'deployment_type': 'cloud_function'
        }
        
        print(f"Prediction: {prediction} ({prediction_label}), Confidence: {confidence:.4f}")
        
        return (json.dumps(response), 200, headers)
        
    except ValueError as e:
        return (
            json.dumps({'error': f'Validation error: {str(e)}'}),
            422,
            headers
        )
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return (
            json.dumps({'error': 'Internal server error', 'details': str(e)}),
            500,
            headers
        )