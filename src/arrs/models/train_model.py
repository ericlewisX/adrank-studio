from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib


def train_ctr_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=6
    )

    model.fit(X_train, y_train)

    preds = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, preds)

    print(f"\nROC-AUC: {auc:.4f}")

    joblib.dump(
        model,
        "models/lgbm_ctr_model.pkl"
    )

    print("Model saved.")

    return model