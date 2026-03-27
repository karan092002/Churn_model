import os
import sys
import numpy as np
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from src.exception import CustomException
from src.logger import logger
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    model_path : str = os.path.join("artifacts", "model.pkl")
    threshold : float = 0.5


class ModelTrainer:
    
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_training(self, train_arr: np.ndarray, test_arr: np.ndarray):
        logger.info("Model training started")
        try:
            # Split arrays back into features and target
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test,  y_test  = test_arr[:, :-1],  test_arr[:, -1]

            # Data is already preprocessed by DataTransformation.
            # Pipelines here contain only the model — no scaler needed again.
            models = {
                "Logistic Regression" : LogisticRegression(
                                            max_iter=1000, random_state=42
                                            ),
                "Decision Tree" : DecisionTreeClassifier(
                                            max_depth=6, random_state=42
                                            ),
                "Random Forest" : RandomForestClassifier(
                                            n_estimators=200, random_state=42
                                            ),
                "AdaBoost" : AdaBoostClassifier(
                                            n_estimators=200, random_state=42
                                            ),
                "SVM" : SVC(probability=True, random_state=42),
                "KNN" : KNeighborsClassifier(n_neighbors=7),
            }

            results_df = evaluate_models(X_train, y_train, X_test, y_test, models)

            logger.info("Model comparison results:\n" + results_df.to_string())
            print("\nModel comparison results:")
            print(results_df.to_string())

            # Pick the model with the highest ROC-AUC on the test set
            best_model_name = results_df["ROC-AUC"].idxmax()
            best_auc = results_df.loc[best_model_name, "ROC-AUC"]

            if best_auc < 0.6:
                raise ValueError(
                    f"Best model AUC is {best_auc:.4f} — below acceptable threshold of 0.6"
                )

            best_model = models[best_model_name]
            best_model.fit(X_train, y_train)  # refit on full train set

            save_object(self.config.model_path, best_model)

            logger.info(
                f"Best model: {best_model_name} — AUC: {best_auc:.4f} — "
                f"saved to {self.config.model_path}"
            )
            print(f"\nBest model: {best_model_name} (AUC = {best_auc:.4f})")

            return best_model_name, best_auc, results_df

        except Exception as e:
            raise CustomException(e, sys)
