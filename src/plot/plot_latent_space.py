import os
from typing import Union
import numpy as np
from sklearn.decomposition import PCA
import tensorflow as tf
import matplotlib.pyplot as plt

from ..models import UnsupervisedAE, UnsupervisedVAE, SupervisedAE, SupervisedVAE


def plot_latent_space(
    model: Union["UnsupervisedAE", "UnsupervisedVAE", "SupervisedAE", "SupervisedVAE"],
    X: np.ndarray,
    y: np.ndarray,
    experiment_name: str,
) -> None:
    """
    Plottet den latenten Raum eines SupervisedAE-Modells als 2D-Scatterplot und speichert das Plot als PNG in einem Google Drive-Ordner.

    Args:
        model: Das SupervisedAE-Modell, dessen latenter Raum visualisiert werden soll.
        X: Eingabedaten als NumPy-Array oder Tensor der Form (Batch-Größe, input_dim).
        y: Labels als NumPy-Array (0 für Nicht Fraud, 1 für Fraud).
        save_path: Optionaler Pfad, um das Plot zu speichern (Standardwert ist None). Wenn None, wird der Pfad aus folder_name generiert.
        folder_name: Name des Ordners (relativ zu /content/drive/MyDrive/), in dem das Plot gespeichert werden soll (Standardwert ist "vae_results/latent_space").

    Raises:
        ValueError: Wenn das Modell einen unbekannten Typ hat, X oder y ungültig sind.
        TypeError: Wenn X oder y kein NumPy-Array oder Tensor ist.
    """

    # Konvertiere X zu Tensor, falls es ein NumPy-Array ist
    x = tf.convert_to_tensor(X, dtype=tf.float32)

    # Extrahiere den latenten Raum mit der encode-Methode
    # The original line was trying to call .numpy() on a tuple, which is not possible.
    # Assuming your encode method returns a tuple where the first element is the latent representation,
    # we need to access that element.
    # Extrahiere den latenten Raum basierend auf dem Modelltyp
    if isinstance(model, UnsupervisedAE):
        latent_representation = model.encode(x).numpy()  # Direkte latente Darstellung
    elif isinstance(model, UnsupervisedVAE):
        z_mean, _, _ = model.encode(x)  # Nimm z_mean als latente Darstellung
        latent_representation = z_mean.numpy()
    elif isinstance(model, SupervisedAE):
        latent_representation = model.encode(x).numpy()  # Direkte latente Darstellung
    elif isinstance(model, SupervisedVAE):
        z_mean, _, _ = model.encode(x)  # Nimm z_mean als latente Darstellung
        latent_representation = z_mean.numpy()
    else:
        raise ValueError(
            "Unbekannter Modelltyp – sollte nicht auftreten aufgrund vorheriger Validierung"
        )

    # Reduziere die Dimensionen auf 2, falls latent_dim > 2
    if latent_representation.shape[1] > 2:
        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(latent_representation)
    else:
        latent_2d = latent_representation  # Verwende direkt die ersten zwei Dimensionen, wenn latent_dim <= 2

    # Speichern des Plots in einem Google Drive-Ordner
    save_path = None
    if save_path is None:
        # Basisverzeichnis für Google Drive
        base_dir = "Results"
        folder_path = os.path.join(base_dir, experiment_name)

        # Erstelle den Ordner, falls er nicht existiert
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Generiere einen eindeutigen Dateinamen basierend auf dem Modelltyp
        model_name: str = model.type
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(
            folder_path, f"latent_space_{model_name}_{timestamp}.png"
        )

    # Erstellen des Plots
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=y, cmap="bwr", alpha=0.5)
    plt.colorbar(scatter, label="Klasse (0 = Nicht Fraud, 1 = Fraud)")
    plt.title(f"Latenter Raum des {model_name.upper()}-Modells")
    plt.xlabel("Latente Dimension 1")
    plt.ylabel("Latente Dimension 2")

    # Speichern des Plots
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    # plt.close()  # Speicherfreigabe in Google Colab

    print(f"Latenter Raum wurde gespeichert unter: {save_path}")
