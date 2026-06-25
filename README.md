# 🔐 Phishing Website Detection

An end-to-end **MLOps pipeline** for detecting phishing websites using classical ML models, with full experiment tracking via MLflow + DagsHub and data storage on MongoDB Atlas.

## 📌 What It Does

This system takes phishing website data, runs it through a complete ML pipeline (ingestion → validation → transformation → training), and produces a model that classifies URLs as **phishing** or **legitimate** based on 30 engineered URL features.


## 🧱 Project Structure

```
Network_Security_System/
├── networksecurity/
│   ├── components/            # Core pipeline stages
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/
│   │   ├── training_pipeline.py   # Full pipeline orchestration
│   │   └── batch_prediction.py
│   ├── utils/
│   │   ├── url_feature_extractor.py  # 30-feature URL parser
│   │   └── ml_utils/                 # Metrics, estimator
│   ├── entity/                # Config & artifact dataclasses
│   ├── constant/              # Pipeline constants
│   ├── exception/             # Custom exception handling
│   └── logging/               # Custom logger
├── Network_Data/
│   └── phisingData.csv        # Raw dataset
├── push_data.py               # MongoDB data upload script
├── main.py                    # Pipeline entry point
├── requirements.txt
└── setup.py
```

## ⚙️ ML Pipeline Stages

### 1. Data Ingestion
- Pulls phishing data from **MongoDB Atlas** (`JahanviAI.NetworkData` collection)
- Splits into train/test sets (80/20 ratio) and saves as CSV

### 2. Data Validation
- Validates schema and checks for data drift
- Generates a drift report (`report.yaml`)

### 3. Data Transformation
- Handles missing values using **KNN Imputer** (k=3)
- Saves preprocessor as `preprocessing.pkl`
- Outputs train/test data as `.npy` arrays

### 4. Model Training
- Trains and evaluates 5 classifiers via **GridSearchCV**:
  - Random Forest
  - Decision Tree
  - Gradient Boosting
  - Logistic Regression
  - AdaBoost
- Selects the best model by score
- Logs metrics (F1, Precision, Recall) to **MLflow via DagsHub**
- Registers the best model as `NetworkSecurityModel`

## 🔍 URL Feature Extractor

The `url_feature_extractor.py` module converts any raw URL into **30 binary/ternary features** used by the model. Features include:

| Category | Examples |
|---|---|
| URL-based | IP address in URL, URL length, shortening service, `@` symbol |
| Domain-based | Subdomain count, prefix/suffix (`-`), suspicious TLDs |
| SSL/Protocol | HTTPS status, favicon source, port number |
| Content-based | Redirects, anchor tags, iframes, popups |
| External | DNS record, web traffic, domain age, page rank |

Each feature returns `-1` (phishing), `0` (suspicious), or `1` (legitimate).

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/jahanvigarg238/Network_Security_System.git
cd Network_Security_System
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
Create a `.env` file:
```
MONGO_DB_URL=your_mongodb_atlas_connection_string
```

### 4. Push data to MongoDB (first-time setup)
```bash
python push_data.py
```

### 5. Run the training pipeline
```bash
python main.py
```

## 📊 Experiment Tracking

MLflow experiments are tracked on DagsHub:

🔗 [View on DagsHub](https://dagshub.com/jahanvigarg238/Network_Security_System.mlflow)

Tracked metrics per run:
- `f1_score`
- `precision`
- `recall_score`

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Scikit-learn | ML models + KNN Imputer |
| MongoDB Atlas + PyMongo | Data storage |
| MLflow | Experiment tracking |
| DagsHub | Remote MLflow backend |
| Pandas / NumPy | Data processing |
| FastAPI + Uvicorn | (API serving — planned) |
| python-dotenv | Environment config |

## 📂 Dataset

**File:** `Network_Data/phisingData.csv`  
**Target column:** `Result` (`-1` = phishing, `1` = legitimate)  
Features represent URL and domain characteristics extracted from known phishing and legitimate websites.
