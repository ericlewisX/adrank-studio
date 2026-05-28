from fastapi import FastAPI
import pandas as pd
import joblib

from services.api.schema import RankRequest


app = FastAPI()

model = joblib.load(
    "models/lgbm_ctr_model.pkl"
)


@app.get("/")
def health_check():
    return {"status": "ARRS API running"}


@app.post("/rank")
def rank_ad(request: RankRequest):

    data = pd.DataFrame([{
        "historical_ctr": request.historical_ctr,
        "bid_amount": request.bid_amount,
        "session_depth": request.session_depth,
        "time_of_day": request.time_of_day,
        "device_type": request.device_type
    }])

    data = pd.get_dummies(
        data,
        columns=["device_type"]
    )

    expected_cols = [
        "historical_ctr",
        "bid_amount",
        "session_depth",
        "time_of_day",
        "device_type_mobile",
        "device_type_tablet"
    ]

    for col in expected_cols:
        if col not in data.columns:
            data[col] = 0

    data = data[expected_cols]

    ctr_score = model.predict_proba(data)[0][1]

    final_score = (
        ctr_score *
        request.bid_amount
    )

    return {
        "ctr_score": round(float(ctr_score), 4),
        "final_score": round(float(final_score), 4)
    }