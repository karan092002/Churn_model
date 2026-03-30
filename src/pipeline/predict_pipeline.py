import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logger
from src.utils import load_object


@dataclass
class PredictPipelineConfig:
    model_path : str = os.path.join("artifacts", "model.pkl")
    preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl")


class PredictPipeline:
    """
    Loads the saved preprocessor and model from artifacts,
    applies the same feature engineering and transformation that was
    done during training, and returns a churn probability for each input row.
    """

    def __init__(self):
        self.config = PredictPipelineConfig()
        self.model = load_object(self.config.model_path)
        self.preprocessor = load_object(self.config.preprocessor_path)

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Mirrors the feature engineering in DataTransformation exactly.
        Must stay in sync with any changes made there.
        """
        df = df.copy()

        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df.drop(columns=["customerID"], inplace=True, errors="ignore")

        df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)
        df["NumAddOns"] = (
            (df["OnlineSecurity"] == "Yes").astype(int) +
            (df["OnlineBackup"] == "Yes").astype(int) +
            (df["DeviceProtection"] == "Yes").astype(int) +
            (df["TechSupport"] == "Yes").astype(int) +
            (df["StreamingTV"] == "Yes").astype(int) +
            (df["StreamingMovies"] == "Yes").astype(int)
        )
        df["HasStreaming"] = ((df["StreamingTV"] == "Yes") |
                                    (df["StreamingMovies"] == "Yes")).astype(int)
        df["HasOnlineServices"] = ((df["OnlineSecurity"] == "Yes") |
                                    (df["OnlineBackup"] == "Yes") |
                                    (df["DeviceProtection"] == "Yes") |
                                    (df["TechSupport"] == "Yes")).astype(int)
        df["IsMonthToMonth"] = (df["Contract"] == "Month-to-month").astype(int)
        df["HasFiberOptic"] = (df["InternetService"] == "Fiber optic").astype(int)
        df["IsElectronicCheck"] = (df["PaymentMethod"] == "Electronic check").astype(int)
        df["TenureGroup"] = pd.cut(
            df["tenure"],
            bins=[0, 12, 24, 48, 72],
            labels=["0-12m", "13-24m", "25-48m", "49-72m"]
        ).astype(str)

        return df

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Accepts a raw DataFrame (same structure as the original CSV, minus
        the Churn column), applies feature engineering and preprocessing,
        and returns an array of churn probabilities.
        """
        try:
            logger.info(f"Prediction requested for {len(df)} row(s)")
            df_engineered = self._engineer_features(df)
            processed = self.preprocessor.transform(df_engineered)
            probabilities = self.model.predict_proba(processed)[:, 1]
            logger.info("Prediction complete")
            return probabilities

        except Exception as e:
            raise CustomException(e, sys)


class CustomerData:
    """
    Helper class that collects individual field values (e.g. from a web form)
    and converts them into a single-row DataFrame that PredictPipeline.predict()
    can consume.
    """

    def __init__(
        self,
        gender : str,
        senior_citizen : int,
        partner : str,
        dependents : str,
        tenure : float,
        phone_service : str,
        multiple_lines : str,
        internet_service : str,
        online_security : str,
        online_backup : str,
        device_protection : str,
        tech_support : str,
        streaming_tv : str,
        streaming_movies : str,
        contract : str,
        paperless_billing : str,
        payment_method : str,
        monthly_charges : float,
        total_charges : float,
    ):
        self.gender = gender
        self.senior_citizen = senior_citizen
        self.partner = partner
        self.dependents = dependents
        self.tenure = tenure
        self.phone_service = phone_service
        self.multiple_lines = multiple_lines
        self.internet_service = internet_service
        self.online_security = online_security
        self.online_backup = online_backup
        self.device_protection = device_protection
        self.tech_support = tech_support
        self.streaming_tv = streaming_tv
        self.streaming_movies = streaming_movies
        self.contract = contract
        self.paperless_billing = paperless_billing
        self.payment_method = payment_method
        self.monthly_charges = monthly_charges
        self.total_charges = total_charges

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "gender" : self.gender,
            "SeniorCitizen" : self.senior_citizen,
            "Partner" : self.partner,
            "Dependents" : self.dependents,
            "tenure" : self.tenure,
            "PhoneService" : self.phone_service,
            "MultipleLines" : self.multiple_lines,
            "InternetService" : self.internet_service,
            "OnlineSecurity" : self.online_security,
            "OnlineBackup" : self.online_backup,
            "DeviceProtection" : self.device_protection,
            "TechSupport" : self.tech_support,
            "StreamingTV" : self.streaming_tv,
            "StreamingMovies" : self.streaming_movies,
            "Contract" : self.contract,
            "PaperlessBilling" : self.paperless_billing,
            "PaymentMethod" : self.payment_method,
            "MonthlyCharges" : self.monthly_charges,
            "TotalCharges" : self.total_charges,
        }])
