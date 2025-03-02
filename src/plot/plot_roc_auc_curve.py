import os
from typing import Union, Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve

from ..models import UnsupervisedAE, UnsupervisedVAE, SupervisedAE, SupervisedVAE, calculate_reconstruction_error
from ..fraud_detection import find_optimal_threshold

def plot_roc_auc_curve(
    model: Union['UnsupervisedAE'
                 , 'UnsupervisedVAE', 'SupervisedAE'
                 , 'SupervisedVAE'
                 ],
    X: np.ndarray,
    y: np.ndarray,
    experiment_name: str,
    threshold: Optional[float] = None,
) -> None:
    """
    Plottet die ROC-AUC-Kurve für ein Modell und speichert das Plot optional in einem Google Drive-Ordner.

    Args:
        model: Das Modell, für das die ROC-Kurve berechnet werden soll (UnsupervisedAE, UnsupervisedVAE, SupervisedAE, oder SupervisedVAE).
        X: Eingabedaten als NumPy-Array oder Tensor der Form (Batch-Größe, input_dim).
        y: Labels als NumPy-Array (0 für Nicht Fraud, 1 für Fraud).
        threshold: Optionaler Schwellenwert für Anomaliedetektion (Standardwert ist None, dann wird find_optimal_threshold verwendet).
        save_path: Optionaler Pfad, um das Plot zu speichern (Standardwert ist None). Wenn None, wird der Pfad aus folder_name generiert.
        folder_name: Name des Ordners (relativ zu /content/drive/MyDrive/), in dem das Plot gespeichert werden soll (Standardwert ist "vae_results/roc_curves").

    Raises:
        ValueError: Wenn das Modell einen unbekannten Typ hat, X oder y ungültig sind.
        TypeError: Wenn X oder y kein NumPy-Array oder Tensor ist.
    """

    # Berechne Rekonstruktionsfehler
    reconstruction_error = calculate_reconstruction_error(model, X)

    # Bestimme optimalen Schwellenwert, falls keiner angegeben ist
    threshold, _ = find_optimal_threshold(model, X, y)
    print(f"Optimaler Schwellenwert basierend auf F1-Score: {threshold}")
    # Extrahiere den Modellnamen
    model_name = model.__class__.__name__  # z. B. "UnsupervisedAE"
    # Berechne ROC-Kurve und AUC
    fpr, tpr, thresholds_roc = roc_curve(y,reconstruction_error)  # Negieren, da niedrigere Rekonstruktionsfehler normal sind
    roc_auc = auc(fpr, tpr)

    # Speichern des Plots in einem Google Drive-Ordner
    save_path = None
    if save_path is None:
        # Basisverzeichnis für Google Drive
        base_dir = "/content/drive/MyDrive"
        folder_path = os.path.join(base_dir, experiment_name)

        # Erstelle den Ordner, falls er nicht existiert
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Generiere einen eindeutigen Dateinamen
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(folder_path, f"{model_name}roc_curve_{timestamp}.png")

    # Erstellen des Plots
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{model_name} ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Zufallsprognose')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC-Kurve für Fraud-Detektion')
    plt.legend(loc="lower right")

    # Speichern des Plots
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #plt.close()  # Speicherfreigabe in Google Colab

    print(f"ROC-Kurve wurde gespeichert unter: {save_path}")
    print(f"ROC-AUC: {roc_auc:.4f}")