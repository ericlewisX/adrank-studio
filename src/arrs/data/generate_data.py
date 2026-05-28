import pandas as pd
import numpy as np


def generate_ctr_data(n_samples=10000, seed=42):
    np.random.seed(seed)

    df = pd.DataFrame({
        "user_id": np.random.randint(1, 1000, n_samples),
        "ad_id": np.random.randint(1, 500, n_samples),

        "historical_ctr":
            np.random.beta(2, 8, n_samples),

        "bid_amount":
            np.random.uniform(0.5, 5.0, n_samples),

        "session_depth":
            np.random.randint(1, 20, n_samples),

        "time_of_day":
            np.random.randint(0, 24, n_samples),

        "device_type":
            np.random.choice(
                ["mobile", "desktop", "tablet"],
                n_samples
            )
    })

    click_prob = (
        0.35 * df["historical_ctr"] +
        0.25 * (df["bid_amount"] / 5.0) +
        0.15 * (1 / df["session_depth"]) +
        0.25 * np.random.rand(n_samples)
    )

    df["clicked"] = (
        click_prob > 0.5
    ).astype(int)

    return df


if __name__ == "__main__":
    data = generate_ctr_data()

    data.to_csv(
        "data/synthetic_ctr_data.csv",
        index=False
    )

    print(data.head())
    print("\nSaved synthetic dataset.")