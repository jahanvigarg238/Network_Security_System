import sys
import os
from typing import Annotated

import certifi
ca = certifi.where()

from dotenv import load_dotenv
load_dotenv()
mongo_db_url = os.getenv("MONGODB_URL_KEY")
print(mongo_db_url)
import pymongo
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Request, Form
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse
import pandas as pd

from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

# NEW: URL feature extractor
from networksecurity.utils.url_feature_extractor import extract_features, features_to_dataframe

client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)

from networksecurity.constant.training_pipeline import DATA_INGESTION_COLLECTION_NAME
from networksecurity.constant.training_pipeline import DATA_INGESTION_DATABASE_NAME

database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="./templates")


@app.get("/", tags=["authentication"])
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/predict-url")
async def predict_url(request: Request, url: Annotated[str, Form(...)]):
    try:
        features = extract_features(url)
        df = features_to_dataframe(features)
        preprocessor = load_object("final_model/preprocessor.pkl")
        final_model = load_object("final_model/model.pkl")
        network_model = NetworkModel(preprocessor=preprocessor, model=final_model)
        y_pred = network_model.predict(df)
        prediction = int(y_pred[0])
        is_phishing = prediction == -1
        feature_breakdown = []
        for name, value in features.items():
            if value == -1:
                status = "danger"
            elif value == 0:
                status = "warn"
            else:
                status = "safe"
            feature_breakdown.append({
                "name": name,
                "value": value,
                "status": status
            })
        red_flags = sum(1 for f in feature_breakdown if f["status"] == "danger")
        risk_score = min(100, int((red_flags / 30) * 100) + (20 if is_phishing else 0))
        return templates.TemplateResponse(
            request=request,
            name="result.html",
            context={
                "url": url,
                "is_phishing": is_phishing,
                "prediction": prediction,
                "risk_score": risk_score,
                "red_flags": red_flags,
                "features": feature_breakdown,
            }
        )
    except Exception as e:
        raise NetworkSecurityException(e, sys)



@app.post("/predict")
async def predict_route(request: Request, file: Annotated[UploadFile, File(...)]):
    try:
        df = pd.read_csv(file.file)
        if "Result" in df.columns:
            df = df.drop(columns=["Result"])
        preprocessor = load_object("final_model/preprocessor.pkl")
        final_model = load_object("final_model/model.pkl")
        network_model = NetworkModel(preprocessor=preprocessor, model=final_model)
        y_pred = network_model.predict(df)
        df['predicted_column'] = y_pred
        os.makedirs('prediction_output', exist_ok=True)
        df.to_csv('prediction_output/output.csv', index=False)
        table_html = df.to_html(classes='table table-striped', index=False)
        return templates.TemplateResponse(
            request=request,
            name="table.html",
            context={"table": table_html}
        )
    except Exception as e:
        raise NetworkSecurityException(e, sys)


@app.get("/train")
async def train_route():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training is successful")
    except Exception as e:
        raise NetworkSecurityException(e, sys)
if __name__ == "__main__":
    app_run(app, host="localhost", port=8000)
