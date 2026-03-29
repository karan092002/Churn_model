import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logger


@dataclass
class DataIngestionConfig:
    """
    Holds file paths for all artifacts produced by data ingestion.
    Using a dataclass keeps configuration separate from logic.
    """
    raw_data_path : str = os.path.join("artifacts", "raw.csv")
    train_data_path : str = os.path.join("artifacts", "train.csv")
    test_data_path : str = os.path.join("artifacts", "test.csv")


class DataIngestion:
    """
    Responsible for:
    - Reading the raw CSV from the data directory
    - Saving a copy of the raw data to artifacts
    - Splitting into train and test sets (stratified on Churn)
    - Saving both splits to artifacts
    """

    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logger.info("Data ingestion started")
        try:
            df = pd.read_csv(os.path.join("data", "Telco-Customer-Churn.csv"))
            logger.info(f"Dataset loaded — shape: {df.shape}")

            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)
            df.to_csv(self.config.raw_data_path, index=False)
            logger.info(f"Raw data saved to {self.config.raw_data_path}")

            train_df, test_df = train_test_split(
                df, test_size=0.2, random_state=42, stratify=df["Churn"]
            )

            train_df.to_csv(self.config.train_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)
            logger.info(
                f"Train/test split complete — "
                f"train: {train_df.shape[0]} rows, test: {test_df.shape[0]} rows"
            )

            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)
