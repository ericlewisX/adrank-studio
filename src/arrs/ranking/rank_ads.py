import pandas as pd


def rank_ads(df, model, feature_cols):

    X = df[feature_cols]  # critical fix

    predictions = model.predict_proba(X)[:, 1]

    df["ctr_score"] = predictions

    df["final_score"] = (df["ctr_score"] * df["bid_amount"])

    ranked = df.sort_values(by="final_score",ascending=False)

    return ranked[
        [
            "ad_id",
            "ctr_score",
            "bid_amount",
            "final_score"
        ]
    ]