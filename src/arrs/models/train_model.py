from lightgbm import LGBMClassifier

from sklearn.model_selection import (
    train_test_split
)

import joblib

from src.arrs.evaluation import (
    evaluate_model,
    save_metrics,
    save_calibration_plot
)


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

    model.fit(
        X_train,
        y_train
    )

    preds = model.predict_proba(
        X_test
    )[:, 1]

    metrics = evaluate_model(
        y_test,
        preds
    )

    print("\nEvaluation Metrics")

    for name, value in metrics.items():
        print(
            f"{name}: {value:.4f}"
        )

    save_metrics(metrics)

    save_calibration_plot(
        y_test,
        preds
    )

    joblib.dump(
        model,
        "models/lgbm_ctr_model.pkl"
    )

    print(
        "\nModel saved."
    )

    return model
