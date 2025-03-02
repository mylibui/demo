import os
import numpy as np
from typing import Union, Optional
import matplotlib.pyplot as plt
import seaborn as sns


def plot_reconstruction_analysis(
    model: Union['UnsupervisedAE'
                 , 'UnsupervisedVAE', 'SupervisedAE'
                 , 'SupervisedVAE'
                 ],
    X: np.ndarray,
    y: np.ndarray,
    threshold: Optional[float] = None,
    save_path: Optional[str] = None,
    folder_name: str = "vae_results/reconstruction_analysis"
) -> None:
    """
    Plottet einen Boxplot und die Verteilung (Histogramm mit Dichte) des Rekonstruktionsfehlers für Fraud und Nicht Fraud in einem Google Drive-Ordner.

    Args:
        model: Das Modell, für das der Rekonstruktionsfehler berechnet werden soll (UnsupervisedAE, UnsupervisedVAE, SupervisedAE, oder SupervisedVAE).
        X: Eingabedaten als NumPy-Array oder Tensor der Form (Batch-Größe, input_dim).
        y: Labels als NumPy-Array (0 für Nicht Fraud, 1 für Fraud).
        threshold: Optionaler Schwellenwert für Anomaliedetektion (Standardwert ist None, dann wird find_optimal_threshold verwendet).
        save_path: Optionaler Pfad, um das Plot zu speichern (Standardwert ist None). Wenn None, wird der Pfad aus folder_name generiert.
        folder_name: Name des Ordners (relativ zu /content/drive/MyDrive/), in dem das Plot gespeichert werden soll (Standardwert ist "vae_results/reconstruction_analysis").

    Raises:
        ValueError: Wenn das Modell einen unbekannten Typ hat, X oder y ungültig sind.
        TypeError: Wenn X oder y kein NumPy-Array oder Tensor ist.
    """
    # Berechne Rekonstruktionsfehler
    reconstruction_error = calculate_reconstruction_error(model, X)

    # Bestimme optimalen Schwellenwert, falls keiner angegeben ist
    threshold, _ = find_optimal_threshold(model, X, y)

    # Trennen von Fraud und Nicht Fraud Daten
    not_fraud_error = reconstruction_error[y == 0]
    fraud_error = reconstruction_error[y == 1]

    # Kombinierte Daten für Boxplot und Histogramm
    data = np.concatenate([not_fraud_error, fraud_error])
    labels = np.concatenate([np.repeat('Nicht Fraud', len(not_fraud_error)), np.repeat('Fraud', len(fraud_error))])

    # Speichern des Plots in einem Google Drive-Ordner
    if save_path is None:
        # Basisverzeichnis für Google Drive
        base_dir = "/content/drive/MyDrive"
        folder_path = os.path.join(base_dir, folder_name)

        # Erstelle den Ordner, falls er nicht existiert
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Generiere einen eindeutigen Dateinamen
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(folder_path, f"reconstruction_analysis_{timestamp}.png")

    # Erstellen des Plots
    plt.figure(figsize=(12, 8))

    # Boxplot (oben)
    plt.subplot(2, 1, 1)
    sns.boxplot(x=labels, y=data, palette={'Nicht Fraud': 'blue', 'Fraud': 'red'})
    plt.title("Boxplot des Rekonstruktionsfehlers: Fraud vs. Nicht Fraud")
    plt.xlabel("")
    plt.ylabel("Rekonstruktionsfehler (MSE)")
    plt.axhline(y=threshold, color='black', linestyle='--', label=f'Optimaler Schwellenwert = {threshold:.4f}')
    plt.legend()

    # Histogramm mit Dichte (unten)
    plt.subplot(2, 1, 2)
    sns.histplot(data=not_fraud_error, bins=50, color='blue', label='Nicht Fraud', alpha=0.5, stat='density')
    sns.histplot(data=fraud_error, bins=50, color='red', label='Fraud', alpha=0.5, stat='density')

    plt.axvline(x=threshold, color='black', linestyle='--', label=f'Optimaler Schwellenwert = {threshold:.4f}')
    plt.title("Verteilung des Rekonstruktionsfehlers (Dichte): Fraud vs. Nicht Fraud")
    plt.xlabel("Rekonstruktionsfehler (MSE)")
    plt.ylabel("Dichte")
    plt.legend()

    plt.tight_layout()

    # Speichern des Plots
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #plt.close()  # Speicherfreigabe in Google Colab

    print(f"Analyse wurde gespeichert unter: {save_path}")