import pandas as pd


def build_features(df):

    df = pd.get_dummies(
        df,
        columns=["device_type"],
        drop_first=True
    )

    feature_cols = [
        "historical_ctr",
        "bid_amount",
        "session_depth",
        "time_of_day",
        "device_type_mobile",
        "device_type_tablet"
    ]

    X = df[feature_cols]
    y = df["clicked"]

    return X, y