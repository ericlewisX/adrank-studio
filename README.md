# 📊 AdRank Studio — Ad Relevance & Ranking System

A machine learning project exploring how modern advertising platforms retrieve, score, rank, and serve advertisements using CTR prediction and ranking pipelines.

This project focuses on building a modular ranking system that resembles the architecture patterns used in ad tech and recommendation systems, while remaining small enough to understand end-to-end.

---

## Project Overview

AdRank Studio simulates a simplified advertising ranking workflow.

The current implementation:

* Generates synthetic advertising interaction data
* Builds model-ready ranking features
* Trains a LightGBM click-through-rate (CTR) prediction model
* Produces ranked advertisements based on predicted engagement
* Exposes ranking functionality through a FastAPI service
* Runs inside a Docker container

The project is being expanded incrementally to include experimentation tooling, monitoring, caching, and interactive visualization components.

---

## Problem Statement

When multiple advertisements are eligible to be shown to a user, which ads should appear first?

Advertising and recommendation systems commonly rank candidates using signals such as:

* Predicted click probability
* User behavior
* Contextual information
* Campaign characteristics
* Business objectives

This project explores one approach to that ranking process using machine learning and modular service design.

---

## Current Status

### ✅ Completed

* Synthetic ad interaction data generation
* Feature engineering pipeline
* LightGBM CTR prediction model
* Model serialization and loading
* Ad ranking workflow
* FastAPI inference service
* Dockerized deployment
* Modular project architecture

### 🚧 In Progress

* Streamlit dashboard
* Ranking quality metrics (NDCG, MAP, Precision@K)
* Experimentation and A/B testing framework
* Monitoring and drift detection
* Redis-backed caching
* Scheduled retraining workflows

---

## System Architecture

```text
Synthetic Data Generation
            │
            ▼
Feature Engineering
            │
            ▼
LightGBM CTR Model
            │
            ▼
Ranking & Scoring Logic
            │
            ▼
FastAPI Service
            │
            ▼
Ranked Ad Results
```

The repository separates reusable machine learning logic from orchestration and serving layers.

---

## Project Structure

```text
adrank-studio/
│
├── models/                  # Trained model artifacts
├── pipelines/               # Workflow entrypoints
├── requirements/            # Dependency management
├── services/
│   └── api/                 # FastAPI service
├── src/
│   └── arrs/
│       ├── data/            # Synthetic data generation
│       ├── features/        # Feature engineering
│       ├── models/          # Model training
│       └── ranking/         # Ranking logic
│
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md
```

---

## Tech Stack

### Machine Learning

* Python
* Pandas
* NumPy
* Scikit-learn
* LightGBM

### API & Serving

* FastAPI
* Uvicorn

### Infrastructure

* Docker
* Git
* GitHub

### Planned Additions

* Streamlit
* Redis
* MLflow
* EvidentlyAI
* Prometheus / Grafana

---

## End-to-End Workflow

### 1. Generate Synthetic Data

The project begins by creating synthetic advertising interaction data.

The generated dataset includes information such as:

* User engagement signals
* Ad identifiers
* Bid values
* Contextual attributes
* Click labels

### 2. Build Features

Raw data is transformed into model-ready features for CTR prediction.

The feature engineering layer provides a dedicated location for adding additional ranking signals as the project grows.

### 3. Train CTR Model

A LightGBM classifier is trained to predict click probability.

The current training workflow:

* Splits data into training and testing sets
* Trains a LightGBM model
* Evaluates ROC-AUC performance
* Saves the trained model to disk

Training entrypoint:

```bash
python pipelines/train_ctr_model.py
```

### 4. Rank Candidate Ads

The ranking layer uses model predictions to score and order advertisements.

This simulates a simplified version of the ranking stage found in advertising and recommendation systems.

### 5. Serve Results Through FastAPI

The FastAPI service exposes ranking functionality through an HTTP API.

This separation between model logic and serving logic mirrors common deployment patterns used in production ML systems.

---

## Running the Project

### Install Dependencies

```bash
pip install -r requirements/requirements-lock.txt
```

### Train the Model

```bash
python pipelines/train_ctr_model.py
```

Example output:

```text
Generating synthetic data...
Building features...
Training model...

ROC-AUC: 0.84

Ranking ads...

Top Ranked Ads:
...
```

### Run the API

```bash
uvicorn services.api.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

---

## Docker

### Build Image

```bash
docker build -t adrank-studio .
```

### Run Container

```bash
docker run -p 8000:8000 adrank-studio
```

---

## Why I Built This Project

Many machine learning portfolio projects stop at model training.

I wanted to build a project that also explores:

* Ranking system design
* API deployment
* Project organization
* Containerization
* ML system architecture

The goal is to better understand the engineering considerations involved in moving from a trained model to a usable service.

---

## Future Roadmap

### Experimentation

* Traffic splitting
* Cohort analysis
* Ranking policy comparison
* Uplift evaluation

### Monitoring

* Feature drift detection
* CTR degradation monitoring
* Latency tracking
* Ranking stability metrics

### Infrastructure

* Redis caching layer
* Automated retraining workflows
* Experiment tracking
* Dashboarding and observability

---

## Author

**Eric Lewis**

Areas of interest:

* Machine Learning Engineering
* Product Data Science
* Recommendation Systems
* Ranking Systems
* ML Infrastructure
* Applied Machine Learning
