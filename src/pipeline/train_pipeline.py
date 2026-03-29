import sys

from src.exception import CustomException
from src.logger import logger
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation


class TrainPipeline:
    """
    Orchestrates the full training sequence in order:
        DataIngestion → DataTransformation → ModelTrainer → ModelEvaluation

    Each component saves its outputs to the artifacts directory so that
    any step can be re-run independently without rerunning the whole pipeline.
    """

    def run(self):
        try:
            logger.info("=" * 60)
            logger.info("TRAINING PIPELINE STARTED")
            logger.info("=" * 60)

            # Step 1 — Ingest raw data and split into train/test
            logger.info("Step 1: Data Ingestion")
            ingestion = DataIngestion()
            train_path, test_path = ingestion.initiate_data_ingestion()

            # Step 2 — Feature engineering and preprocessing
            logger.info("Step 2: Data Transformation")
            transformation = DataTransformation()
            train_arr, test_arr, preprocessor_path = (
                transformation.initiate_data_transformation(train_path, test_path)
            )

            # Step 3 — Train and compare models, save the best one
            logger.info("Step 3: Model Training")
            trainer = ModelTrainer()
            best_model_name, best_auc, results_df = (
                trainer.initiate_model_training(train_arr, test_arr)
            )

            # Step 4 — Full evaluation report and plots for the best model
            logger.info("Step 4: Model Evaluation")
            evaluator = ModelEvaluation()
            evaluator.initiate_model_evaluation(test_arr, best_model_name)

            logger.info("=" * 60)
            logger.info(f"TRAINING PIPELINE COMPLETE")
            logger.info(f"Best model : {best_model_name}")
            logger.info(f"ROC-AUC    : {best_auc:.4f}")
            logger.info("=" * 60)

            print(f"\nTraining pipeline complete.")
            print(f"Best model : {best_model_name}")
            print(f"ROC-AUC    : {best_auc:.4f}")
            print(f"Artifacts  : artifacts/")

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run()
