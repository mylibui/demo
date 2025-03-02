import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score
from typing import Union, Optional, Tuple

from ..models import calculate_reconstruction_error

def find_optimal_threshold(
    model: Union['UnsupervisedAE'
                 , 'UnsupervisedVAE'
                 , 'SupervisedAE'
                 , 'SupervisedVAE'
                 ],
    X: np.ndarray,
    y: np.ndarray,
    thresholds: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    """
    Bestimmt den optimalen Schwellenwert für die Anomaliedetektion basierend auf dem F1-Score.

    Args:
        model: Das Modell, für das der Rekonstruktionsfehler berechnet werden soll (UnsupervisedAE, UnsupervisedVAE, SupervisedAE, oder SupervisedVAE).
        X: Eingabedaten als NumPy-Array oder Tensor der Form (Batch-Größe, input_dim).
        y: Labels als NumPy-Array (0 für Nicht Fraud, 1 für Fraud).
        thresholds: Optionaler Array mit Schwellenwerten zum Testen (Standardwert ist None, dann wird ein Bereich von 0.1 bis 5.0 mit 100 Schritten verwendet).

    Returns:
        Tuple[float, float]: (optimaler Schwellenwert, maximaler F1-Score).

    Raises:
        ValueError: Wenn das Modell einen unbekannten Typ hat, X oder y ungültig sind.
        TypeError: Wenn X oder y kein NumPy-Array oder Tensor ist.
    """

    # Konvertiere X zu Tensor, falls es ein NumPy-Array ist
    x = tf.convert_to_tensor(X, dtype=tf.float32)

    # Berechne Rekonstruktionsfehler
    reconstruction_error = calculate_reconstruction_error(model, X)

    # Standard-Schwellenwerte, falls keine angegeben sind
    if thresholds is None:
        # Wähle Schwellenwerte basierend auf den tatsächlichen Rekonstruktionsfehlern
        min_error = np.min(reconstruction_error)
        max_error = np.max(reconstruction_error)
        thresholds = np.linspace(min_error, max_error, 100)

    # Rest der Funktion bleibt gleich
    best_f1 = 0.0
    best_threshold = 0.0

    for th in thresholds:
        y_pred = (reconstruction_error > th).astype(int)
        f1 = f1_score(y, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = th

    return best_threshold, best_f1