from typing import Union, Optional
import numpy as np

from ..models import (
    calculate_reconstruction_error,
    UnsupervisedVAE,
    UnsupervisedAE,
    SupervisedAE,
    SupervisedVAE,
)
from .threshhold import find_optimal_threshold


def detect_fraud(
    model: Union["UnsupervisedAE", "UnsupervisedVAE", "SupervisedAE", "SupervisedVAE"],
    X: np.ndarray,
    threshold: Optional[float] = None,
) -> np.ndarray:
    """
    Detektiert Fraud-Transaktionen basierend auf dem Rekonstruktionsfehler und einem Schwellenwert.

    Args:
        model: Das Modell, für das der Rekonstruktionsfehler berechnet werden soll (UnsupervisedAE, UnsupervisedVAE, SupervisedAE, oder SupervisedVAE).
        X: Eingabedaten als NumPy-Array oder Tensor der Form (Batch-Größe, input_dim).
        threshold: Optionaler Schwellenwert für Anomaliedetektion (Standardwert ist None, dann wird find_optimal_threshold verwendet).

    Returns:
        np.ndarray: Array mit Vorhersagen (0 für Nicht Fraud, 1 für Fraud).

    Raises:
        ValueError: Wenn das Modell einen unbekannten Typ hat oder X ungültig ist.
        TypeError: Wenn X kein NumPy-Array oder Tensor ist.
    """

    # Berechne Rekonstruktionsfehler
    reconstruction_error = calculate_reconstruction_error(model, X)

    # Bestimme optimalen Schwellenwert, falls keiner angegeben ist
    if threshold is None:
        # Hier simulieren wir, dass wir Labels haben (z. B. aus Testdaten)
        # In der Praxis müssen Sie y_test bereitstellen
        # Für dieses Beispiel nehmen wir an, dass wir keine Labels haben und einen Standard-Schwellenwert verwenden
        threshold, _ = find_optimal_threshold(
            model, X, np.zeros(len(X))
        )  # Dummy-Labels für Demo
        print(f"Optimaler Schwellenwert (ohne echte Labels): {threshold}")

    # Detektiere Fraud basierend auf Schwellenwert
    fraud_predictions = (reconstruction_error > threshold).astype(int)

    return fraud_predictions
