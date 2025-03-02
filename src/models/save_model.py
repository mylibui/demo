import os

def save_model_to_drive(self, folder_path: str = "vae_results/models") -> None:
    """
    Speichert das Modell und seine Gewichte in einem Google Drive-Ordner.

    Args:
        folder_path: Pfad im Google Drive, in dem das Modell gespeichert werden soll (relativ zu /content/drive/MyDrive/). Standardwert ist "vae_results/models".
    """
    import os
    import datetime

    # Basisverzeichnis für Google Drive
    base_dir = "/content/drive/MyDrive"
    full_path = os.path.join(base_dir, folder_path)

    # Erstelle den Ordner, falls er nicht existiert
    if not os.path.exists(full_path):
        os.makedirs(full_path)

    # Speichere das Modell
    model_path = os.path.join(full_path, f"uae_model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.weights.h5") # Changed the file extension to .weights.h5
    self.save_weights(model_path)
    print(f"Modell wurde gespeichert unter: {model_path}")