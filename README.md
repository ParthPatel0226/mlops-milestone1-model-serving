\# MLOps Milestone 1: Web \& Serverless Model Serving



FastAPI and Cloud Function deployment for ML model serving with lifecycle awareness and reproducibility.



!\[Python](https://img.shields.io/badge/Python-3.11-blue)

!\[FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green)

!\[scikit--learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange)



---



\## Project Overview



This project demonstrates model serving patterns for MLOps:

\- \*\*FastAPI Web Service\*\*: Containerized model serving with schema validation

\- \*\*Cloud Function\*\*: Serverless inference endpoint

\- \*\*Comparative Analysis\*\*: Architecture trade-offs and lifecycle implications



\*\*ML Lifecycle Position:\*\*

```

Data → Training → \[Model Artifact] → API Serving → Consumer Applications

&nbsp;                       ↓

&nbsp;                  model.pkl

&nbsp;                       ↓

&nbsp;             \[This Deployment Layer]

```



---



\## Table of Contents



\- \[Local Setup](#local-setup)

\- \[API Usage](#api-usage)

\- \[Model Details](#model-details)

\- \[Deployment Instructions](#deployment-instructions)

\- \[Lifecycle \& Architecture](#lifecycle--architecture)

\- \[Comparative Analysis](#comparative-analysis)

\- \[GCP Access Note](#gcp-access-note)



---



\## Local Setup



\### Prerequisites



\- Python 3.10 or higher

\- pip

\- Git



\### Installation Steps



1\. \*\*Clone the repository:\*\*

```bash

&nbsp;  git clone https://github.com/ParthPatel0226/mlops-milestone1-model-serving.git

&nbsp;  cd mlops-milestone1-model-serving

```



2\. \*\*Create virtual environment:\*\*

```bash

&nbsp;  python -m venv venv

&nbsp;  source venv/bin/activate  # Windows: venv\\Scripts\\activate

```



3\. \*\*Install dependencies:\*\*

```bash

&nbsp;  pip install -r requirements.txt

```



4\. \*\*Train model (if needed):\*\*

```bash

&nbsp;  python train\_model.py

```

&nbsp;  This creates `model.pkl` using scikit-learn's Iris dataset.



5\. \*\*Run FastAPI server:\*\*

```bash

&nbsp;  uvicorn main:app --host 0.0.0.0 --port 8000 --reload

```



6\. \*\*Test the API:\*\*

```bash

&nbsp;  # Health check

&nbsp;  curl http://localhost:8000/



&nbsp;  # Prediction

&nbsp;  curl -X POST "http://localhost:8000/predict" \\

&nbsp;    -H "Content-Type: application/json" \\

&nbsp;    -d '{"features": \[5.1, 3.5, 1.4, 0.2]}'

```



---



\## API Usage



\### Endpoints



\#### `GET /` - Health Check

Returns service status and model loading confirmation.



\*\*Response:\*\*

```json

{

&nbsp; "status": "healthy",

&nbsp; "service": "ML Model Serving API",

&nbsp; "model\_loaded": true

}

```



\#### `POST /predict` - Make Prediction

Accepts 4 features and returns predicted Iris species.



\*\*Request Schema (Pydantic):\*\*

```json

{

&nbsp; "features": \[5.1, 3.5, 1.4, 0.2]

}

```



\*\*Response Schema (Pydantic):\*\*

```json

{

&nbsp; "prediction": 0,

&nbsp; "prediction\_label": "setosa",

&nbsp; "confidence": 0.98,

&nbsp; "model\_version": "1.0.0"

}

```



\*\*Example using Python requests:\*\*

```python

import requests



response = requests.post(

&nbsp;   "http://localhost:8000/predict",

&nbsp;   json={"features": \[5.1, 3.5, 1.4, 0.2]}

)

print(response.json())

```



\#### `GET /model-info` - Model Information

Returns metadata about the loaded model.



\*\*Response:\*\*

```json

{

&nbsp; "model\_type": "LogisticRegression",

&nbsp; "model\_version": "1.0.0",

&nbsp; "input\_features": 4,

&nbsp; "output\_classes": 3,

&nbsp; "class\_names": \["setosa", "versicolor", "virginica"]

}

```



---



\## Model Details



\### Training



\- \*\*Dataset\*\*: Iris classification (scikit-learn built-in)

\- \*\*Algorithm\*\*: Logistic Regression

\- \*\*Features\*\*: 4 (sepal length, sepal width, petal length, petal width)

\- \*\*Classes\*\*: 3 (setosa, versicolor, virginica)

\- \*\*Accuracy\*\*: ~100% on test set



\### Artifact Management



\*\*Deterministic Loading:\*\*

The model is loaded once at application startup (not per-request) to ensure:

\- ✅ Fast inference (no repeated disk I/O)

\- ✅ Consistent predictions

\- ✅ Reproducible behavior

```python

\# Loaded at startup (main.py)

model = joblib.load('model.pkl')



@app.post("/predict")

def predict(request: PredictRequest):

&nbsp;   # Model already in memory - fast prediction

&nbsp;   prediction = model.predict(features)

```



\*\*Reproducibility Test:\*\*

```bash

python train\_model.py

\# Verifies that model.predict() returns identical results across loads

```



---



\## Deployment Instructions



\### Cloud Run Deployment (HTTPS Container)



\*\*Note\*\*: Requires GCP billing account access (see \[GCP Access Note](#gcp-access-note) below).



\#### Prerequisites

\- Google Cloud SDK installed

\- GCP project with billing enabled

\- Docker installed



\#### Steps



1\. \*\*Build Docker image:\*\*

```bash

&nbsp;  docker build -t gcr.io/PROJECT\_ID/ml-api:v1 .

```



2\. \*\*Push to Artifact Registry:\*\*

```bash

&nbsp;  docker push gcr.io/PROJECT\_ID/ml-api:v1

```



3\. \*\*Deploy to Cloud Run:\*\*

```bash

&nbsp;  gcloud run deploy ml-api \\

&nbsp;    --image gcr.io/PROJECT\_ID/ml-api:v1 \\

&nbsp;    --platform managed \\

&nbsp;    --region us-central1 \\

&nbsp;    --allow-unauthenticated

```



4\. \*\*Get service URL:\*\*

```bash

&nbsp;  gcloud run services describe ml-api --region us-central1 --format 'value(status.url)'

```



\*\*Expected URL format:\*\*

```

https://ml-api-xxxxx-uc.a.run.app

```



\### Cloud Function Deployment (Serverless)



\*\*Location\*\*: `cloud\_function/` directory



\#### Files

\- `main.py` - Function entry point with prediction logic

\- `requirements.txt` - Function dependencies



\#### Deploy Command

```bash

gcloud functions deploy predict \\

&nbsp; --runtime python311 \\

&nbsp; --trigger-http \\

&nbsp; --allow-unauthenticated \\

&nbsp; --region us-central1 \\

&nbsp; --entry-point predict

```



---



\## Lifecycle \& Architecture



\### ML Lifecycle Position

```

┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐

│   Data      │────▶│   Training   │────▶│   Model     │────▶│  Deployment  │

│ Collection  │     │  \& Tuning    │     │  Artifact   │     │  \& Serving   │

└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘

&nbsp;                                               ↓                      ↓

&nbsp;                                         model.pkl           ┌────────────────┐

&nbsp;                                                             │ \[THIS PROJECT] │

&nbsp;                                                             │  FastAPI/CF    │

&nbsp;                                                             └────────────────┘

&nbsp;                                                                     ↓

&nbsp;                                                             ┌────────────────┐

&nbsp;                                                             │   Consumers    │

&nbsp;                                                             │ (Apps/Users)   │

&nbsp;                                                             └────────────────┘

```



\*\*This deployment sits at the "Serving" stage\*\*, bridging model artifacts and consumer applications.



\### Model-API Interaction



\*\*Startup Phase:\*\*

```python

\# Application initialization (happens once)

model = joblib.load('model.pkl')  # Deterministic artifact loading

app = FastAPI()                    # Initialize API framework

```



\*\*Request Phase:\*\*

```python

\# Per-request processing (happens many times)

1\. Client sends POST /predict with features

2\. Pydantic validates input schema

3\. Model.predict(features) - uses pre-loaded model

4\. Pydantic formats output schema

5\. Return JSON response to client

```



\*\*Key Design Decisions:\*\*

\- ✅ Load model at startup (not per-request) → Low latency

\- ✅ Pydantic schemas → Input validation \& output consistency

\- ✅ Stateful container → Model stays in memory between requests



\### Monitoring Touchpoints



\*\*Where to add observability:\*\*

```python

@app.post("/predict")

def predict(request: PredictRequest):

&nbsp;   # 1. Log request count (Prometheus counter)

&nbsp;   # 2. Start latency timer (Prometheus histogram)

&nbsp;   

&nbsp;   prediction = model.predict(features)

&nbsp;   

&nbsp;   # 3. Log prediction distribution (for drift detection)

&nbsp;   # 4. Track inference errors (alert on spike)

&nbsp;   # 5. Monitor memory usage (model size)

&nbsp;   

&nbsp;   return response

```



---



\## Comparative Analysis



\### FastAPI Container vs Cloud Function



| Aspect | FastAPI (Cloud Run) | Cloud Function |

|--------|---------------------|----------------|

| \*\*Deployment Pattern\*\* | Containerized web service | Serverless function |

| \*\*State Management\*\* | Stateful - model stays in memory | Stateless - reloads each cold start |

| \*\*Artifact Loading\*\* | Once at startup | Per cold start (15+ min idle) |

| \*\*Cold Start Latency\*\* | ~2-4 seconds (container start) | ~1-2 seconds (lighter weight) |

| \*\*Warm Latency\*\* | <100ms (model in memory) | <100ms (if warm) |

| \*\*Concurrency\*\* | Multiple requests share instance | One request per instance |

| \*\*Cost Model\*\* | Pay for CPU/memory-time | Pay per invocation + compute |

| \*\*Best For\*\* | High throughput, low latency | Sporadic/event-driven workloads |



\### Lifecycle Differences



\*\*FastAPI Container (Stateful):\*\*

\- ✅ Model loaded once → Fast subsequent requests

\- ✅ Can cache intermediate results

\- ✅ Better for consistent traffic patterns

\- ❌ Heavier cold starts (full container)

\- ❌ More complex deployment (Docker)



\*\*Cloud Function (Stateless):\*\*

\- ✅ Lighter cold starts (no container overhead)

\- ✅ Simpler deployment (just code)

\- ✅ Auto-scales to zero (cost-efficient for low traffic)

\- ❌ Model reloads after idle periods

\- ❌ No shared state between invocations



\### Artifact Loading Strategies



\*\*FastAPI Approach:\*\*

```python

\# Global scope - loads once at container start

model = joblib.load('model.pkl')



@app.post("/predict")

def predict(...):

&nbsp;   return model.predict(...)  # Fast - model already in memory

```



\*\*Cloud Function Approach:\*\*

```python

\# Global scope - loads at cold start

model = None



def predict(request):

&nbsp;   global model

&nbsp;   if model is None:  # Cold start

&nbsp;       model = joblib.load('model.pkl')  # Reload penalty

&nbsp;   return model.predict(...)  # Fast once loaded

```



\### Reproducibility Considerations



\*\*Container (Higher Reproducibility):\*\*

\- ✅ Pinned Python version in Dockerfile

\- ✅ Exact dependency versions

\- ✅ Same OS environment (Alpine/Debian)

\- ✅ Immutable image in registry



\*\*Function (Lower Reproducibility):\*\*

\- ⚠️ Runtime managed by GCP (updates automatically)

\- ⚠️ Less control over system dependencies

\- ⚠️ Potential version drift over time



\### When to Choose Each



\*\*Choose FastAPI Container when:\*\*

\- Steady traffic with predictable patterns

\- Latency-sensitive applications (every ms counts)

\- Need persistent connections or state

\- Complex dependencies or custom runtime



\*\*Choose Cloud Function when:\*\*

\- Sporadic/event-driven workloads

\- Cost optimization for low traffic

\- Simple inference logic

\- Rapid prototyping/experimentation



---



\## GCP Access Note



\*\*Current Status\*\*: Local development and FastAPI implementation complete. Cloud deployments (Cloud Run and Cloud Function) are pending due to GCP billing account access restrictions.



\*\*Issue\*\*: University organization policies prevent billing account creation on both institutional and personal Google accounts.



\*\*Resolution Path\*\*:

1\. Awaiting guidance from course instructor regarding GCP access for students

2\. Alternative: Create independent Google account not linked to any organization

3\. Upon access resolution, will complete Cloud Run and Cloud Function deployments



\*\*What's Ready\*\*:

\- ✅ Complete FastAPI application (tested locally)

\- ✅ Dockerfile for Cloud Run deployment

\- ✅ Cloud Function code prepared (see `cloud\_function/`)

\- ✅ Comparative analysis (research-based)



\*\*What's Pending\*\*:

\- ⏳ Cloud Run service URL

\- ⏳ Artifact Registry image reference

\- ⏳ Cloud Function deployment

\- ⏳ Live endpoint testing and latency benchmarks



---



\## Project Structure

```

mlops-milestone1-model-serving/

├── main.py                 # FastAPI application

├── train\_model.py          # Model training script

├── model.pkl              # Trained model artifact

├── requirements.txt       # Python dependencies (pinned versions)

├── Dockerfile            # Container definition for Cloud Run

├── cloud\_function/       # Cloud Function code

│   ├── main.py          # Function entry point

│   └── requirements.txt # Function dependencies

├── screenshots/          # Deployment evidence (when available)

└── README.md            # This file

```



---



\## Technologies Used



\- \*\*Python 3.11\*\* - Programming language

\- \*\*FastAPI 0.104.1\*\* - Web framework

\- \*\*Pydantic 2.12.5\*\* - Schema validation

\- \*\*scikit-learn 1.8.0\*\* - ML model training

\- \*\*Uvicorn 0.24.0\*\* - ASGI server

\- \*\*Docker\*\* - Containerization

\- \*\*GCP Cloud Run\*\* - Container deployment (pending)

\- \*\*GCP Cloud Functions\*\* - Serverless deployment (pending)



---



\## Author



\*\*Parth Patel\*\*  

MS in Management Information Systems  

University of Illinois Chicago



---



\## License



MIT License - see LICENSE file for details

