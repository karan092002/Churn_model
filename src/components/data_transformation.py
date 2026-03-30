import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer

from src.exception import CustomException
from src.logger import logger
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_path : str = os.path.join("artifacts", "preprocessor.pkl")
    train_arr_path : str = os.path.join("artifacts", "train_arr.pkl")
    test_arr_path : str = os.path.join("artifacts", "test_arr.pkl")


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        # Fix TotalCharges — stored as string with blank values for new customers
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

        # Drop identifier — no predictive value
        df.drop(columns=["customerID"], inplace=True, errors="ignore")

        # Normalised spend — raw TotalCharges is misleading across different tenures
        df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)

        # Engagement score — more add-ons means more locked in
        df["NumAddOns"] = (
            (df["OnlineSecurity"] == "Yes").astype(int) +
            (df["OnlineBackup"] == "Yes").astype(int) +
            (df["DeviceProtection"] == "Yes").astype(int) +
            (df["TechSupport"] == "Yes").astype(int) +
            (df["StreamingTV"] == "Yes").astype(int) +
            (df["StreamingMovies"] == "Yes").astype(int)
        )

        # Binary flags derived from EDA patterns
        df["HasStreaming"] = ((df["StreamingTV"] == "Yes") |
                                    (df["StreamingMovies"] == "Yes")).astype(int)
        df["HasOnlineServices"] = ((df["OnlineSecurity"] == "Yes") |
                                    (df["OnlineBackup"] == "Yes") |
                                    (df["DeviceProtection"] == "Yes") |
                                    (df["TechSupport"] == "Yes")).astype(int)
        df["IsMonthToMonth"] = (df["Contract"] == "Month-to-month").astype(int)
        df["HasFiberOptic"] = (df["InternetService"] == "Fiber optic").astype(int)
        df["IsElectronicCheck"] = (df["PaymentMethod"] == "Electronic check").astype(int)

        # Tenure bucketed — non-linear relationship with churn
        df["TenureGroup"] = pd.cut(
            df["tenure"],
            bins=[0, 12, 24, 48, 72],
            labels=["0-12m", "13-24m", "25-48m", "49-72m"]
        ).astype(str)

        return df

    def _get_preprocessor(self, numeric_cols: list, categorical_cols: list):
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            ))
        ])

        preprocessor = ColumnTransformer(transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ])

        return preprocessor

    def initiate_data_transformation(self, train_path: str, test_path: str):
        logger.info("Data transformation started")
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Encode target
            train_df["Churn"] = (train_df["Churn"] == "Yes").astype(int)
            test_df["Churn"] = (test_df["Churn"]  == "Yes").astype(int)

            # Feature engineering
            train_df = self._engineer_features(train_df)
            test_df = self._engineer_features(test_df)
            logger.info("Feature engineering complete")

            # Separate features and target
            X_train = train_df.drop(columns=["Churn"])
            y_train = train_df["Churn"]
            X_test = test_df.drop(columns=["Churn"])
            y_test = test_df["Churn"]

            # Identify column types
            numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = X_train.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

            logger.info(f"Numeric columns : {numeric_cols}")
            logger.info(f"Categorical columns : {categorical_cols}")

            # Build and fit preprocessor on train only
            preprocessor = self._get_preprocessor(numeric_cols, categorical_cols)
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)

            # Combine features and target into single arrays for downstream use
            train_arr = np.c_[X_train_processed, np.array(y_train)]
            test_arr = np.c_[X_test_processed,  np.array(y_test)]

            # Save artifacts
            save_object(self.config.preprocessor_path, preprocessor)
            save_object(self.config.train_arr_path, train_arr)
            save_object(self.config.test_arr_path, test_arr)

            logger.info(
                f"Data transformation complete — "
                f"train shape: {train_arr.shape}, test shape: {test_arr.shape}"
            )

            return train_arr, test_arr, self.config.preprocessor_path

        except Exception as e:
            raise CustomException(e, sys)
