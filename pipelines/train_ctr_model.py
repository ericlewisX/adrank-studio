import pandas as pd

from src.arrs.data.generate_data import (
    generate_ctr_data
)

from src.arrs.features.build_features import (
    build_features
)

from src.arrs.models.train_model import (
    train_ctr_model
)

from src.arrs.ranking.rank_ads import (
    rank_ads
)


print("\nGenerating synthetic data...")
df = generate_ctr_data()

print("\nBuilding features...")
X, y = build_features(df)

print("\nTraining model...")
model = train_ctr_model(X, y)

feature_cols = X.columns.tolist()

print("\nRanking ads...")
ranking_df = X.copy()

ranking_df["ad_id"] = df["ad_id"]
ranking_df["bid_amount"] = df["bid_amount"]

ranked_ads = rank_ads(
    ranking_df,
    model,
    feature_cols
)

print("\nTop Ranked Ads:")
print(ranked_ads.head(10))