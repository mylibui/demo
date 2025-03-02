from typing import Union
import numpy as np
import tensorflow as tf

def calculate_reconstruction_error(
    model: Union['UnsupervisedAE'
                 , 'UnsupervisedVAE'
                 , 'SupervisedAE'
                 , 'SupervisedVAE'
                 ],
    X: np.ndarray
) -> np.ndarray:
    """
    Berechnet den Rekonstruktionsfehler (MSE) für ein Modell.

    Args:
        model: Das Modell, für das der Rekonstruktionsfehler berechnet werden soll (UnsupervisedAE, UnsupervisedVAE, SupervisedAE, oder SupervisedVAE).
        X: Eingabedaten als NumPy-Array oder Tensor der Form (Batch-Größe, input_dim).

    Returns:
        np.ndarray: Array mit MSE-Werten pro Sample.

    Raises:
        ValueError: Wenn das Modell einen unbekannten Typ hat oder X ungültig ist.
        TypeError: Wenn X kein NumPy-Array oder Tensor ist.
    """
    # Rekonstruktion basierend auf Modelltyp
    if isinstance(model, UnsupervisedAE):
        reconstructed = model(X, training=False)  # Nur rekonstruierte Daten für UnsupervisedAE
    elif isinstance(model, UnsupervisedVAE):
        reconstructed, _, _ = model(X, training=False)  # Ignoriere z_mean und z_log_var, nehme nur rekonstruiert
    elif isinstance(model, SupervisedAE):
        reconstructed, _ = model(X, training=False)  # Ignoriere Klassifikationsausgabe
    elif isinstance(model, SupervisedVAE):
        reconstructed, _, _, _ = model(X, training=False)  # Ignoriere z_mean, z_log_var und Klassifikation

    # Berechnung des MSE pro Sample
    reconstruction_error = tf.reduce_mean(tf.square(X - reconstructed), axis=-1)

    # Konvertierung zu NumPy-Array für einfache Handhabung
    return reconstruction_error.numpy()