import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from src.exception import CustomException
from src.logger import logger
from src.utils import load_object


@dataclass
class ModelEvaluationConfig:
    report_dir : str = os.path.join("artifacts", "evaluation")
    cm_plot_path : str = os.path.join("artifacts", "evaluation", "confusion_matrix.png")
    roc_plot_path : str = os.path.join("artifacts", "evaluation", "roc_curve.png")
    report_path : str = os.path.join("artifacts", "evaluation", "classification_report.txt")


class ModelEvaluation:
    
    def __init__(self):
        self.config = ModelEvaluationConfig()
        os.makedirs(self.config.report_dir, exist_ok=True)
        sns.set_theme(style="whitegrid", palette="muted")

    def _plot_confusion_matrix(self, y_test, y_pred, model_name: str):
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"],
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {model_name}")
        plt.tight_layout()
        plt.savefig(self.config.cm_plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Confusion matrix saved to {self.config.cm_plot_path}")

    def _plot_roc_curve(self, y_test, y_prob, model_name: str, auc: float):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve — {model_name}")
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(self.config.roc_plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"ROC curve saved to {self.config.roc_plot_path}")

    def initiate_model_evaluation(
        self,
        test_arr: np.ndarray,
        model_name: str,
    ):
        logger.info("Model evaluation started")
        try:
            X_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

            model = load_object(os.path.join("artifacts", "model.pkl"))

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)

            report = classification_report(
                y_test, y_pred, target_names=["No Churn", "Churn"]
            )

            print(f"\nEvaluation Report — {model_name}")
            print(f"ROC-AUC : {auc:.4f}")
            print(report)

            # Save text report
            with open(self.config.report_path, "w") as f:
                f.write(f"Model: {model_name}\n")
                f.write(f"ROC-AUC: {auc:.4f}\n\n")
                f.write(report)
            logger.info(f"Classification report saved to {self.config.report_path}")

            # Save plots
            self._plot_confusion_matrix(y_test, y_pred, model_name)
            self._plot_roc_curve(y_test, y_prob, model_name, auc)

            logger.info(f"Model evaluation complete — AUC: {auc:.4f}")
            return auc

        except Exception as e:
            raise CustomException(e, sys)
