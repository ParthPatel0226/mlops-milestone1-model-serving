# MLOps Milestone 1: Web & Serverless Model Serving

FastAPI and Cloud Function deployment for ML model serving with lifecycle awareness and reproducibility.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange)

---

## Project Overview

This project demonstrates model serving patterns for MLOps:
- **FastAPI Web Service**: Containerized model serving with schema validation
- **Cloud Function**: Serverless inference endpoint
- **Comparative Analysis**: Architecture trade-offs and lifecycle implications

**ML Lifecycle Position:**
```
Data → Training → [Model Artifact] → API Serving → Consumer Applications
                        ↓
                   model.pkl
                        ↓
              [This Deployment Layer]
```

---

## Table of Contents

- [Local Setup](#local-setup)
- [API Usage](#api-usage)
- [Model Details](#model-details)
- [Live Deployments](#live-deployment-urls)
- [Deployment Instructions](#deployment-instructions)
- [Lifecycle & Architecture](#lifecycle--architecture)
- [Comparative Analysis](#comparative-analysis)

---

## Local Setup

### Prerequisites

- Python 3.10 or higher
- pip
- Git

### Installation Steps

1. **Clone the repository:**
```bash
   git clone https://github.com/ParthPatel0226/mlops-milestone1-model-serving.git
   cd mlops-milestone1-model-serving
```

2. **Create virtual environment:**
```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
   pip install -r requirements.txt
```

4. **Train model (if needed):**
```bash
   python train_model.py
```
   This creates `model.pkl` using scikit-learn's Iris dataset.

5. **Run FastAPI server:**
```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

6. **Test the API:**
```bash
   # Health check
   curl http://localhost:8000/

   # Prediction
   curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

---

## API Usage

### Endpoints

#### `GET /` - Health Check
Returns service status and model loading confirmation.

**Response:**
```json
{
  "status": "healthy",
  "service": "ML Model Serving API",
  "model_loaded": true
}
```

#### `POST /predict` - Make Prediction
Accepts 4 features and returns predicted Iris species.

**Request Schema (Pydantic):**
```json
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

**Response Schema (Pydantic):**
```json
{
  "prediction": 0,
  "prediction_label": "setosa",
  "confidence": 0.98,
  "model_version": "1.0.0"
}
```

**Example using Python requests:**
```python
import requests

response = requests.post(
    "https://ml-api-202083721674.us-central1.run.app/predict",
    json={"features": [5.1, 3.5, 1.4, 0.2]}
)
print(response.json())
```

#### `GET /model-info` - Model Information
Returns metadata about the loaded model.

**Response:**
```json
{
  "model_type": "LogisticRegression",
  "model_version": "1.0.0",
  "input_features": 4,
  "output_classes": 3,
  "class_names": ["setosa", "versicolor", "virginica"]
}
```

---

## Model Details

### Training

- **Dataset**: Iris classification (scikit-learn built-in)
- **Algorithm**: Logistic Regression
- **Features**: 4 (sepal length, sepal width, petal length, petal width)
- **Classes**: 3 (setosa, versicolor, virginica)
- **Accuracy**: ~100% on test set

### Artifact Management

**Deterministic Loading:**
The model is loaded once at application startup (not per-request) to ensure:
- ✅ Fast inference (no repeated disk I/O)
- ✅ Consistent predictions
- ✅ Reproducible behavior
```python
# Loaded at startup (main.py)
model = joblib.load('model.pkl')

@app.post("/predict")
def predict(request: PredictRequest):
    # Model already in memory - fast prediction
    prediction = model.predict(features)
```

**Reproducibility Test:**
```bash
python train_model.py
# Verifies that model.predict() returns identical results across loads
```

---

## Live Deployment URLs

### Cloud Run Service (Container)
**Service URL:** https://ml-api-202083721674.us-central1.run.app

**Endpoints:**
- Health: `GET /`
- Prediction: `POST /predict`
- Model Info: `GET /model-info`

**Artifact Registry Image:**
```
us-central1-docker.pkg.dev/project-c6c5efaf-f038-49a2-9fa/ml-models/ml-api:v1
```

**Test Command:**
```bash
curl -X POST "https://ml-api-202083721674.us-central1.run.app/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

**Response:**
```json
{
  "prediction": 0,
  "prediction_label": "setosa",
  "confidence": 0.98,
  "model_version": "1.0.0"
}
```

### Cloud Function (Serverless)
**Function URL:** https://us-central1-project-c6c5efaf-f038-49a2-9fa.cloudfunctions.net/predict

**Test Command:**
```bash
curl -X POST "https://us-central1-project-c6c5efaf-f038-49a2-9fa.cloudfunctions.net/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

**Response:**
```json
{
  "prediction": 0,
  "prediction_label": "setosa",
  "confidence": 0.9766,
  "model_version": "1.0.0",
  "deployment_type": "cloud_function"
}
```

**Both deployments tested and working successfully!**

---

## Deployment Instructions

### Cloud Run Deployment (HTTPS Container)

#### Prerequisites
- Google Cloud SDK installed
- GCP project with billing enabled
- Docker installed

#### Steps

1. **Enable required APIs:**
```bash
   gcloud services enable run.googleapis.com
   gcloud services enable artifactregistry.googleapis.com
   gcloud services enable cloudbuild.googleapis.com
```

2. **Create Artifact Registry repository:**
```bash
   gcloud artifacts repositories create ml-models \
     --repository-format=docker \
     --location=us-central1 \
     --description="ML model serving images"
```

3. **Configure Docker authentication:**
```bash
   gcloud auth configure-docker us-central1-docker.pkg.dev
```

4. **Build Docker image:**
```bash
   docker build -t us-central1-docker.pkg.dev/PROJECT_ID/ml-models/ml-api:v1 .
```

5. **Push to Artifact Registry:**
```bash
   docker push us-central1-docker.pkg.dev/PROJECT_ID/ml-models/ml-api:v1
```

6. **Deploy to Cloud Run:**
```bash
   gcloud run deploy ml-api \
     --image us-central1-docker.pkg.dev/PROJECT_ID/ml-models/ml-api:v1 \
     --region us-central1 \
     --allow-unauthenticated \
     --platform managed
```

### Cloud Function Deployment (Serverless)

**Location**: `cloud_function/` directory

#### Files
- `main.py` - Function entry point with prediction logic
- `requirements.txt` - Function dependencies
- `model.pkl` - Trained model artifact

#### Deploy Command
```bash
cd cloud_function

# Grant required permissions (one-time)
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member=serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com \
  --role=roles/cloudbuild.builds.builder

# Deploy function
gcloud functions deploy predict \
  --gen2 \
  --runtime python311 \
  --region us-central1 \
  --source . \
  --entry-point predict \
  --trigger-http \
  --allow-unauthenticated
```

---

## Lifecycle & Architecture

### ML Lifecycle Position
```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│   Data      │────▶│   Training   │────▶│   Model     │────▶│  Deployment  │
│ Collection  │     │  & Tuning    │     │  Artifact   │     │  & Serving   │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
                                                ↓                      ↓
                                          model.pkl           ┌────────────────┐
                                                              │ [THIS PROJECT] │
                                                              │  FastAPI/CF    │
                                                              └────────────────┘
                                                                      ↓
                                                              ┌────────────────┐
                                                              │   Consumers    │
                                                              │ (Apps/Users)   │
                                                              └────────────────┘
```

**This deployment sits at the "Serving" stage**, bridging model artifacts and consumer applications.

### Model-API Interaction

**Startup Phase:**
```python
# Application initialization (happens once)
model = joblib.load('model.pkl')  # Deterministic artifact loading
app = FastAPI()                    # Initialize API framework
```

**Request Phase:**
```python
# Per-request processing (happens many times)
1. Client sends POST /predict with features
2. Pydantic validates input schema
3. Model.predict(features) - uses pre-loaded model
4. Pydantic formats output schema
5. Return JSON response to client
```

**Key Design Decisions:**
- ✅ Load model at startup (not per-request) → Low latency
- ✅ Pydantic schemas → Input validation & output consistency
- ✅ Stateful container → Model stays in memory between requests

### Monitoring Touchpoints

**Where to add observability:**
```python
@app.post("/predict")
def predict(request: PredictRequest):
    # 1. Log request count (Prometheus counter)
    # 2. Start latency timer (Prometheus histogram)
    
    prediction = model.predict(features)
    
    # 3. Log prediction distribution (for drift detection)
    # 4. Track inference errors (alert on spike)
    # 5. Monitor memory usage (model size)
    
    return response
```

---

## Comparative Analysis

### FastAPI Container vs Cloud Function

| Aspect | FastAPI (Cloud Run) | Cloud Function |
|--------|---------------------|----------------|
| **Deployment Pattern** | Containerized web service | Serverless function |
| **State Management** | Stateful - model stays in memory | Stateless - reloads each cold start |
| **Artifact Loading** | Once at startup | Per cold start (15+ min idle) |
| **Cold Start Latency** | ~2-4 seconds (container start) | ~1-2 seconds (lighter weight) |
| **Warm Latency** | <100ms (model in memory) | <100ms (if warm) |
| **Concurrency** | Multiple requests share instance | One request per instance |
| **Cost Model** | Pay for CPU/memory-time | Pay per invocation + compute |
| **Best For** | High throughput, low latency | Sporadic/event-driven workloads |

### Lifecycle Differences

**FastAPI Container (Stateful):**
- ✅ Model loaded once → Fast subsequent requests
- ✅ Can cache intermediate results
- ✅ Better for consistent traffic patterns
- ❌ Heavier cold starts (full container)
- ❌ More complex deployment (Docker)

**Cloud Function (Stateless):**
- ✅ Lighter cold starts (no container overhead)
- ✅ Simpler deployment (just code)
- ✅ Auto-scales to zero (cost-efficient for low traffic)
- ❌ Model reloads after idle periods
- ❌ No shared state between invocations

### Artifact Loading Strategies

**FastAPI Approach:**
```python
# Global scope - loads once at container start
model = joblib.load('model.pkl')

@app.post("/predict")
def predict(...):
    return model.predict(...)  # Fast - model already in memory
```

**Cloud Function Approach:**
```python
# Global scope - loads at cold start
model = None

def predict(request):
    global model
    if model is None:  # Cold start
        model = joblib.load('model.pkl')  # Reload penalty
    return model.predict(...)  # Fast once loaded
```

### Reproducibility Considerations

**Container (Higher Reproducibility):**
- ✅ Pinned Python version in Dockerfile
- ✅ Exact dependency versions
- ✅ Same OS environment (Alpine/Debian)
- ✅ Immutable image in registry

**Function (Lower Reproducibility):**
- ⚠️ Runtime managed by GCP (updates automatically)
- ⚠️ Less control over system dependencies
- ⚠️ Potential version drift over time

### When to Choose Each

**Choose FastAPI Container when:**
- Steady traffic with predictable patterns
- Latency-sensitive applications (every ms counts)
- Need persistent connections or state
- Complex dependencies or custom runtime

**Choose Cloud Function when:**
- Sporadic/event-driven workloads
- Cost optimization for low traffic
- Simple inference logic
- Rapid prototyping/experimentation

---

## Project Structure
```
mlops-milestone1-model-serving/
├── main.py                 # FastAPI application
├── train_model.py          # Model training script
├── model.pkl              # Trained model artifact
├── requirements.txt       # Python dependencies (pinned versions)
├── Dockerfile            # Container definition for Cloud Run
├── cloud_function/       # Cloud Function code
│   ├── main.py          # Function entry point
│   ├── requirements.txt # Function dependencies
│   └── model.pkl        # Model artifact (copy)
├── screenshots/          # Deployment evidence
└── README.md            # This file
```

---

## Technologies Used

- **Python 3.11** - Programming language
- **FastAPI 0.104.1** - Web framework
- **Pydantic 2.12.5** - Schema validation
- **scikit-learn 1.8.0** - ML model training
- **Uvicorn 0.24.0** - ASGI server
- **Docker** - Containerization
- **GCP Cloud Run** - Container deployment
- **GCP Cloud Functions** - Serverless deployment
- **GCP Artifact Registry** - Container image storage

---

## Author

**Parth Patel**  
MS in Management Information Systems  
University of Illinois Chicago

**GitHub:** https://github.com/ParthPatel0226

---

## License

MIT License - see LICENSE file for details