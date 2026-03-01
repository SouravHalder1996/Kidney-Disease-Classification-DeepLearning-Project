<!-- ## Workflows

1. Update config.yaml
2. Update secrets.yaml/.env [optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline
8. Update the main.py
9. Update the dvc.yaml
10. app.py -->

# Kidney Disease Classification — Deep Learning Project

An **end-to-end deep learning pipeline** for classifying kidney disease from medical images. Built on a fine-tuned VGG16 architecture, the system automates data ingestion, model preparation, training, and evaluation — with experiment tracking via MLflow and DagsHub, pipeline reproducibility via DVC, and deployment through a Flask REST API.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Pipeline Stages](#pipeline-stages)
4. [Project Structure](#project-structure)
5. [Technology Stack](#technology-stack)
6. [Installation & Setup](#installation--setup)
7. [Usage](#usage)
8. [API Endpoints](#api-endpoints)
9. [Model Configuration](#model-configuration)

---

## Project Overview

The **Kidney Disease Classification** project is a production-grade deep learning solution that classifies medical kidney images to aid in disease detection. It leverages transfer learning on VGG16 pretrained on ImageNet, with a complete MLOps workflow covering experiment tracking, pipeline versioning, and cloud deployment.

**Key Objectives**
- Automate the full deep learning lifecycle from data ingestion to model deployment.
- Enable reproducible experiments using DVC pipeline versioning.
- Track and compare model runs with MLflow on DagsHub.
- Expose predictions via a lightweight Flask REST API.

**Business Impact**
- Support medical professionals with fast, automated kidney disease screening.
- Maintain a fully reproducible pipeline for continuous model improvement.
- Enable seamless retraining via a single API call.

---

## Architecture

![Architecture](README_assets/Project%20Structure.png)

### System Layers

| Layer | Components |
|-------|-----------|
| **Client** | Flask REST API — home, training trigger, and prediction endpoints |
| **Orchestration** | `main.py` sequences all four pipeline stages end-to-end |
| **Pipeline** | Data Ingestion → Prepare Base Model → Model Training → Model Evaluation |
| **Tracking** | MLflow experiment tracking synced to DagsHub |
| **Versioning** | DVC for pipeline and data versioning (`dvc repro`) |
| **Storage** | Local filesystem for artifacts, remote storage for DVC-tracked data |

### High-Level Flow

```
Client (Flask API)
        ↓
Pipeline Orchestration (main.py / dvc repro)
        ↓
Stage 1: Data Ingestion
        ↓
Stage 2: Prepare Base Model (VGG16 + Custom Head)
        ↓
Stage 3: Model Training (Fine-tuning with Augmentation)
        ↓
Stage 4: Model Evaluation (MLflow Logging)
        ↓
Prediction API (/predict)
```

---

## Pipeline Stages

### 1. Data Ingestion
Downloads and prepares the kidney disease image dataset, organising it into the expected directory structure for training and validation.

### 2. Prepare Base Model
Loads VGG16 pretrained on ImageNet (`WEIGHTS: imagenet`), freezes the convolutional base (`INCLUDE_TOP: False`), and attaches a custom classification head for 2-class output.

### 3. Model Training
Fine-tunes the model on the prepared dataset with data augmentation enabled. Training runs for the configured number of epochs using the specified batch size and learning rate.

### 4. Model Evaluation
Evaluates the trained model and logs all metrics and parameters to MLflow via DagsHub, enabling full experiment tracking and run comparison.

---

## Project Structure

```
Kidney-Disease-Classification-DeepLearning-Project/
│
├── main.py                          # Entry point — sequential pipeline execution
├── app.py                           # Flask application with API endpoints
├── dvc.yaml                         # DVC pipeline stage definitions
├── dvc.lock                         # DVC lock file for reproducibility
├── params.yaml                      # Model hyperparameters
├── scores.json                      # Model evaluation scores output
├── requirements.txt                 # Python dependencies
├── setup.py                         # Package setup
├── template.py                      # Project template scaffold
├── Dockerfile                       # Container definition
├── inputImage.jpg                   # Sample input image
├── mlflow.db                        # Local MLflow tracking database
├── .env                             # Environment variables (not committed)
├── .dvcignore                       # DVC ignore rules
│
├── config/
│   └── config.yaml                  # Pipeline paths and configuration
│
├── src/
│   └── Kidney_Disease_Classifier/
│       ├── components/              # Core stage implementations
│       │   ├── data_ingestion.py
│       │   ├── prepare_base_model.py
│       │   ├── model_training.py
│       │   └── model_evaluation_mlflow.py
│       │
│       ├── pipeline/                # Pipeline stage orchestration
│       │   ├── stage_01_data_ingestion.py
│       │   ├── stage_02_prepare_base_model.py
│       │   ├── stage_03_model_training.py
│       │   ├── stage_04_model_evaluation.py
│       │   └── prediction.py        # Inference pipeline
│       │
│       ├── config/
│       │   └── configuration.py     # Configuration manager
│       │
│       ├── entity/
│       │   └── config_entity.py     # Configuration data classes
│       │
│       ├── constants/               # Project-wide constants
│       │
│       └── utils/
│           └── common.py            # Shared utilities (decodeImage, etc.)
│
├── model/
│   └── model.h5                     # Final deployed model
│
├── artifacts/                       # Generated pipeline artifacts
│   ├── data_ingestion/
│   │   ├── kidney-dataset/          # Extracted image dataset
│   │   └── data.zip                 # Raw downloaded data
│   ├── prepare_base_model/
│   │   ├── base_model.h5            # Original VGG16 base
│   │   └── base_model_updated.h5    # VGG16 with custom classification head
│   └── training/
│       └── model.h5                 # Trained model artifact
│
├── notebooks/                       # Exploratory notebooks
├── templates/
│   └── index.html                   # Web UI template
├── logs/                            # Application logs (generated)
│
└── .github/
    └── workflows/
        └── main.yaml                # GitHub Actions CI/CD workflow
```

---

## Technology Stack

| Category | Technology | Purpose |
|----------|------------|---------|
| Deep Learning | TensorFlow / Keras | Model building and training |
| Base Model | VGG16 (ImageNet) | Transfer learning backbone |
| Web Framework | Flask | REST API and web interface |
| Experiment Tracking | MLflow + DagsHub | Metric logging and run comparison |
| Pipeline Versioning | DVC | Reproducible pipeline execution |
| Containerisation | Docker | Portable deployment |
| CI/CD | GitHub Actions | Automated testing and deployment |
| Language | Python 3.10 | Core implementation |

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- Conda (recommended) or pip
- DagsHub account (for MLflow tracking)
- Git

### Step 1 — Clone Repository
```bash
git clone https://github.com/SouravHalder1996/Kidney-Disease-Classification-DeepLearning-Project.git
cd Kidney-Disease-Classification-DeepLearning-Project
```

### Step 2 — Create Conda Environment
```bash
conda create -n kidney-disease python=3.10 -y
conda activate kidney-disease
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Configure MLflow Tracking
Export the following environment variables with credentials copied from your DagsHub repository:
```bash
export MLFLOW_TRACKING_URI=<your_dagshub_mlflow_tracking_uri>
export MLFLOW_TRACKING_USERNAME=<your_dagshub_username>
export MLFLOW_TRACKING_PASSWORD=<your_dagshub_token>
```

---

## Usage

### Run Full Training Pipeline
Executes all four stages sequentially:
```bash
python main.py
```

### Run Pipeline via DVC
Reproduces only the stages that have changed since the last run:
```bash
dvc repro
```

### Start the Flask API Server
```bash
python app.py
# Visit http://localhost:8080
```

---

## API Endpoints

**Base URL:** `http://localhost:8080`

| Method | Endpoint | Description | Response |
|--------|----------|-------------|----------|
| `GET` | `/` | Renders the web UI | `index.html` |
| `GET/POST` | `/train` | Triggers `dvc repro` to retrain the model | `200 OK` — "Training completed successfully!" |
| `POST` | `/predict` | Accepts a base64-encoded image and returns classification result | JSON with prediction |

**POST `/predict` — Request format:**
```json
{
  "image": "<base64_encoded_image_string>"
}
```

**POST `/predict` — Response format:**
```json
[{ "image": "Tumor" }]
```

---

## Model Configuration

All hyperparameters are managed centrally in `params.yaml`

To modify training behaviour, update `params.yaml` and re-run `dvc repro`. DVC will automatically detect the change and re-execute only the affected downstream stages.

---