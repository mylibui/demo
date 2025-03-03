import tensorflow as tf
import numpy as np
import os
import datetime
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
)
from typing import Union, Optional, Dict

from ..models import UnsupervisedAE, UnsupervisedVAE, SupervisedAE, SupervisedVAE
from .detect_fraud import detect_fraud
from .threshhold import find_optimal_threshold


def evaluate_fraud_detection(
    model: Union["UnsupervisedAE", "UnsupervisedVAE", "SupervisedAE", "SupervisedVAE"],
    X: np.ndarray,
    y: np.ndarray,
    experiment_name: str,
    save_metrics: bool = False,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """
    Evaluert die Fraud-Detektion, berechnet Metriken und gibt einen Klassifikationsbericht aus.

    Args:
        model: Das Modell, für das der Rekonstruktionsfehler berechnet werden soll (UnsupervisedAE, UnsupervisedVAE, SupervisedAE, oder SupervisedVAE).
        X: Eingabedaten als NumPy-Array oder Tensor der Form (Batch-Größe, input_dim).
        y: Labels als NumPy-Array (0 für Nicht Fraud, 1 für Fraud).
        threshold: Optionaler Schwellenwert für Anomaliedetektion (Standardwert ist None, dann wird find_optimal_threshold verwendet).
        save_metrics: Boolean, ob die Metriken in einer Textdatei in Google Drive gespeichert werden sollen (Standardwert ist False).
        folder_name: Name des Ordners (relativ zu /content/drive/MyDrive/), in dem die Metriken gespeichert werden sollen (Standardwert ist "vae_results/metrics").

    Returns:
        Dict[str, float]: Dictionary mit Metriken (Precision, Recall, F1-Score, Accuracy, ROC-AUC).

    Raises:
        ValueError: Wenn das Modell einen unbekannten Typ hat, X oder y ungültig sind.
        TypeError: Wenn X oder y kein NumPy-Array oder Tensor ist.
    """
    # Bestimme optimalen Schwellenwert, falls keiner angegeben ist
    if threshold is None:
        threshold, _ = find_optimal_threshold(model, X, y)
        print(f"Optimaler Schwellenwert basierend auf F1-Score: {threshold}")

    # Detektiere Fraud
    fraud_predictions = detect_fraud(model, X, threshold)

    # Berechne Metriken
    precision = precision_score(y, fraud_predictions)
    recall = recall_score(y, fraud_predictions)
    f1 = f1_score(y, fraud_predictions)
    accuracy = accuracy_score(y, fraud_predictions)

    # Berechne ROC-AUC (benötigt Wahrscheinlichkeiten, daher verwenden wir Rekonstruktionsfehler als Score)
    roc_auc = roc_auc_score(
        y, fraud_predictions
    )  # Negieren, da niedrigere Rekonstruktionsfehler normal sind

    # Erstelle Klassifikationsbericht
    print("\nKlassifikationsbericht:")
    print(
        classification_report(
            y, fraud_predictions, target_names=["Nicht Fraud", "Fraud"]
        )
    )

    # Gib Metriken aus
    metrics_dict = {
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Accuracy": accuracy,
        "ROC-AUC": roc_auc,
    }
    print("\nMetriken:")
    for metric, value in metrics_dict.items():
        print(f"{metric}: {value:.4f}")

    # Speichere Metriken in Google Drive, falls gewünscht
    if save_metrics:
        # Basisverzeichnis für Google Drive
        base_dir = "Results"
        folder_path = os.path.join(base_dir, experiment_name)

        # Erstelle den Ordner, falls er nicht existiert
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Generiere einen eindeutigen Dateinamen
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_path = os.path.join(
            folder_path, f"metrics_{model.type}_{timestamp}.txt"
        )

        with open(metrics_path, "w") as f:
            f.write("Klassifikationsbericht:\n")
            f.write(
                classification_report(
                    y, fraud_predictions, target_names=["Nicht Fraud", "Fraud"]
                )
            )
            f.write("\nMetriken:\n")
            for metric, value in metrics_dict.items():
                f.write(f"{metric}: {value:.4f}\n")

        print(f"Metriken wurden gespeichert unter: {metrics_path}")

    return metrics_dict
