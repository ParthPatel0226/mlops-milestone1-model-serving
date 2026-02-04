\# GCP Access Issue - Milestone 1 Submission



\## Student Information

\- \*\*Name:\*\* Parth Patel

\- \*\*Course:\*\* MLOps

\- \*\*Assignment:\*\* Milestone 1 - Web \& Serverless Model Serving



---



\## Issue Summary



Unable to complete Cloud Run and Cloud Function deployments due to Google Cloud Platform billing account restrictions.



\### Error Encountered

```

Free trial unavailable

Sorry, you can't start a free trial because you don't have permission 

to create a billing account for your organization.

```



\### Accounts Tested

1\. University email (@uic.edu) - Organization restrictions

2\. Personal Gmail - Also blocked (linked to organization)



\### Resolution Attempts

\- Attempted free trial activation on both accounts

\- Verified no existing projects accessible

\- Reviewed course documentation on GCP free tier



---



\## What Has Been Completed



\### ✅ FastAPI Service (2 pts)

\- \*\*Status:\*\* COMPLETE and TESTED

\- Pydantic request/response schemas implemented

\- Deterministic model loading at startup

\- Reproducible environment with pinned dependencies

\- Tested locally - all endpoints working

\- \*\*Evidence:\*\* `main.py`, `requirements.txt`, local test screenshots possible



\### ✅ Lifecycle Understanding (2 pts)

\- \*\*Status:\*\* COMPLETE

\- Comprehensive documentation of deployment stages

\- Model-API interaction explained

\- Monitoring touchpoints identified

\- Lifecycle diagram included

\- \*\*Evidence:\*\* README.md sections on "Lifecycle \& Architecture"



\### ✅ Comparative Analysis (1 pt)

\- \*\*Status:\*\* COMPLETE (research-based)

\- FastAPI container vs Cloud Function comparison

\- Latency characteristics analyzed (based on GCP documentation)

\- Statelessness and reproducibility discussed

\- Artifact loading strategies compared

\- \*\*Evidence:\*\* README.md "Comparative Analysis" section



\### ✅ Documentation \& Reproducibility (1 pt)

\- \*\*Status:\*\* COMPLETE

\- Clear setup instructions

\- API usage examples with sample requests/responses

\- Well-organized code structure

\- Comprehensive README

\- \*\*Evidence:\*\* README.md, code organization



\### ✅ Cloud Deployment Code Prepared

\- \*\*Status:\*\* READY TO DEPLOY

\- Dockerfile complete for Cloud Run

\- Cloud Function code implemented in `cloud\_function/`

\- Both ready to deploy immediately upon GCP access

\- \*\*Evidence:\*\* `Dockerfile`, `cloud\_function/main.py`



---



\## What Is Pending



\### ⏳ Cloud Run Deployment (2 pts)

\- \*\*Status:\*\* CODE READY, ACCESS BLOCKED

\- Dockerfile complete and tested syntax

\- Cannot push to Artifact Registry (requires billing)

\- Cannot deploy to Cloud Run (requires billing)



\### ⏳ Serverless Function (2 pts)

\- \*\*Status:\*\* CODE READY, ACCESS BLOCKED

\- Cloud Function implementation complete

\- Cannot deploy to GCP (requires billing)

\- Cannot test invocation (requires deployment)



---



\## Repository Information



\*\*GitHub Repository:\*\* https://github.com/ParthPatel0226/mlops-milestone1-model-serving



\### Repository Contents

\- `main.py` - FastAPI application (working)

\- `train\_model.py` - Model training script (working)

\- `model.pkl` - Trained model artifact

\- `Dockerfile` - Cloud Run container definition (ready)

\- `cloud\_function/` - Complete Cloud Function code (ready)

\- `requirements.txt` - Pinned dependencies

\- `README.md` - Comprehensive documentation



---



\## Local Testing Evidence



\### FastAPI Service Endpoints

All endpoints tested and working locally on `http://localhost:8000`:



\*\*Health Check (`GET /`):\*\*

```json

{

&nbsp; "status": "healthy",

&nbsp; "service": "ML Model Serving API",

&nbsp; "model\_loaded": true

}

```



\*\*Prediction (`POST /predict`):\*\*

Request:

```json

{"features": \[5.1, 3.5, 1.4, 0.2]}

```

Response:

```json

{

&nbsp; "prediction": 0,

&nbsp; "prediction\_label": "setosa",

&nbsp; "confidence": 0.98,

&nbsp; "model\_version": "1.0.0"

}

```



\*\*Model Info (`GET /model-info`):\*\*

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



\## Requested Assistance



\### Options for Completion

1\. \*\*Instructor-provided GCP project:\*\* Access to course GCP organization

2\. \*\*Temporary billing account:\*\* Assistance with creating unblocked account

3\. \*\*Alternative grading:\*\* Evaluate based on completed components + code readiness

4\. \*\*Extension:\*\* Additional time to resolve GCP access independently



\### Commitment

Upon receiving GCP access, I can complete the cloud deployments within 1-2 hours:

\- Build and push Docker image to Artifact Registry

\- Deploy to Cloud Run

\- Deploy Cloud Function

\- Conduct latency testing

\- Update repository with live URLs



---



\## Skills Demonstrated



Despite GCP access limitations, this submission demonstrates:

\- ✅ FastAPI development with proper schema validation

\- ✅ ML artifact management and deterministic loading

\- ✅ Docker containerization skills

\- ✅ Serverless architecture understanding

\- ✅ Comparative analysis and critical thinking

\- ✅ Lifecycle awareness and monitoring considerations

\- ✅ Clear documentation and reproducibility practices

\- ✅ Professional code organization



---



\## Contact



For questions or to discuss GCP access resolution:

\- \*\*GitHub:\*\* ParthPatel0226

\- \*\*Email:\*\* \[your email if you want to include it]



---



\*\*Submission Date:\*\* February 3, 2026

