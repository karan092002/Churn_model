import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

from src.exception import CustomException
from src.logger import logger


def save_object(file_path: str, obj) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(obj, file_path)
        logger.info(f"Object saved to {file_path}")
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No file found at {file_path}")
        obj = joblib.load(file_path)
        logger.info(f"Object loaded from {file_path}")
        return obj
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    models: dict,
) -> pd.DataFrame:
    from sklearn.metrics import (
        accuracy_score, precision_score,
        recall_score, f1_score,
    )

    try:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        results = {}

        for name, pipeline in models.items():
            logger.info(f"Training model: {name}")

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            cv_scores = cross_val_score(
                pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
            )

            results[name] = {
                "Accuracy"  : accuracy_score(y_test, y_pred),
                "Precision" : precision_score(y_test, y_pred),
                "Recall"    : recall_score(y_test, y_pred),
                "F1"        : f1_score(y_test, y_pred),
                "ROC-AUC"   : roc_auc_score(y_test, y_prob),
                "CV Mean"   : cv_scores.mean(),
                "CV Std"    : cv_scores.std(),
            }
            logger.info(f"{name} — AUC: {results[name]['ROC-AUC']:.4f}")

        results_df = pd.DataFrame(results).T.round(4)
        return results_df

    except Exception as e:
        raise CustomException(e, sys)
