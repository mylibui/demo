import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import os
from typing import Union, Tuple, List, Dict

def plot_loss_curves(
    history: Union[tf.keras.callbacks.History, Dict],
    model_type: str,  
    experiment_name: str,
    figsize: Tuple[int, int] = (16, 10),
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Plottet umfassende Loss-Kurven für verschiedene Autoencoder-Modelltypen (UnsupervisedAE, UnsupervisedVAE,
    SupervisedAE, SupervisedVAE) und speichert sie optional in einem Google Drive-Ordner.

    Args:
        history: Das History-Objekt aus dem `model.fit()`-Aufruf oder ein Dictionary mit Metriken.
        model_type: Typ des Modells ('uae', 'uvae', 'sae', 'svae'). Standardwert ist 'uae'.
        figsize: Größe der Abbildung als Tuple (Breite, Höhe). Standardwert ist (16, 10).
        save_path: Optionaler Pfad, um das Plot zu speichern (Standardwert ist None). Wenn None, wird der Pfad aus folder_name generiert.
        folder_name: Name des Ordners (relativ zu /content/drive/MyDrive/), in dem das Plot gespeichert werden soll (Standardwert ist "vae_results/loss_curves").

    Returns:
        Tuple[plt.Figure, List[plt.Axes]]: Figure- und Axes-Objekte für weitere Anpassungen.

    Raises:
        ValueError: Wenn `history` kein gültiges History-Objekt oder Dictionary ist, `model_type` unbekannt ist, oder keine gültigen Metriken vorhanden sind.
        TypeError: Wenn `figsize` kein Tuple oder `history` kein Dict/History-Objekt ist.
    """
    # Eingabevalidierung
    if not isinstance(history, (tf.keras.callbacks.History, dict)):
        raise TypeError("history muss ein tf.keras.callbacks.History-Objekt oder Dictionary sein")
    if not isinstance(figsize, tuple) or len(figsize) != 2 or not all(isinstance(d, int) and d > 0 for d in figsize):
        raise TypeError("figsize muss ein Tuple von zwei positiven Ganzzahlen sein")
    valid_model_types = {'uae', 'uvae', 'sae', 'svae'}
    if model_type not in valid_model_types:
        raise ValueError(f"model_type muss einer von {valid_model_types} sein, nicht {model_type}")

    # Extrahiere Metriken aus History oder Dictionary
    metrics = list(history.history.keys() if isinstance(history, tf.keras.callbacks.History) else history.keys())
    if not metrics:
        raise ValueError("Keine Metriken in history vorhanden")

    # Bestimme relevante Loss-Metriken basierend auf dem Modelltyp
    loss_metrics = [m for m in metrics if 'loss' in m.lower() and 'val' not in m]
    val_metrics_exist = any('val_' in m for m in metrics)

    # Konfiguriere die Anzahl der Subplots basierend auf dem Modelltyp
    if model_type in ['uae', 'uvae']:
        expected_losses = ['loss', 'reconstruction_loss']
        if model_type == 'uvae':
            expected_losses.append('kl_loss')
        n_plots = min(3, len([m for m in loss_metrics if m in expected_losses]))
    else:  # 'sae', 'svae'
        expected_losses = ['loss', 'reconstruction_loss', 'classification_loss']
        if model_type == 'svae':
            expected_losses.append('kl_loss')
        n_plots = min(4, len([m for m in loss_metrics if m in expected_losses]))

    # Erstelle Figure und Axes
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]  # Für Konsistenz bei einem einzigen Plot

    # Farben und Stil für die Plots
    colors = {
        'loss': 'blue',
        'reconstruction_loss': 'green',
        'kl_loss': 'cyan',
        'classification_loss': 'orange'
    }
    line_styles = {'training': '-', 'validation': '--'}

    # Plot-Index
    ax_idx = 0

    # Plot 1: Gesamtverlust (falls vorhanden, ansonsten überspringen)
    if 'loss' in metrics:
        ax = axes[ax_idx]
        ax.plot(history.history['loss'], color=colors['loss'], linestyle=line_styles['training'],
                linewidth=2, label='Training')
        if val_metrics_exist and 'val_loss' in metrics:
            ax.plot(history.history['val_loss'], color=colors['loss'], linestyle=line_styles['validation'],
                    linewidth=2, label='Validierung')

        ax.set_title('Gesamtverlust', fontsize=14)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(alpha=0.3)

        # Annotations für Start- und Endwerte
        start_val = history.history['loss'][0]
        end_val = history.history['loss'][-1]
        ax.annotate(f'Start: {start_val:.4f}', xy=(0, start_val), xytext=(5, 10),
                    textcoords='offset points', color=colors['loss'], fontsize=10)
        ax.annotate(f'End: {end_val:.4f}', xy=(len(history.history['loss'])-1, end_val),
                   xytext=(5, -15), textcoords='offset points', color=colors['loss'], fontsize=10)

        if val_metrics_exist and 'val_loss' in metrics:
            val_start = history.history['val_loss'][0]
            val_end = history.history['val_loss'][-1]
            ax.annotate(f'Start: {val_start:.4f}', xy=(0, val_start), xytext=(5, -30),
                        textcoords='offset points', color=colors['loss'], fontsize=10)
            ax.annotate(f'End: {val_end:.4f}', xy=(len(history.history['val_loss'])-1, val_end),
                       xytext=(5, -45), textcoords='offset points', color=colors['loss'], fontsize=10)

        ax_idx += 1
    else:
        print("Warnung: 'loss' nicht in history.history gefunden – Gesamtverlust wird nicht geplottet.")

    # Plot 2: Rekonstruktionsverlust
    if 'reconstruction_loss' in metrics and ax_idx < n_plots:
        ax = axes[ax_idx]
        ax.plot(history.history['reconstruction_loss'], color=colors['reconstruction_loss'], linestyle=line_styles['training'],
                linewidth=2, label='Training')
        if val_metrics_exist and 'val_reconstruction_loss' in metrics:
            ax.plot(history.history['val_reconstruction_loss'], color=colors['reconstruction_loss'], linestyle=line_styles['validation'],
                    linewidth=2, label='Validierung')

        ax.set_title('Rekonstruktionsverlust', fontsize=14)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(alpha=0.3)

        # Annotations für Endwerte
        end_val = history.history['reconstruction_loss'][-1]
        ax.annotate(f'End: {end_val:.4f}', xy=(len(history.history['reconstruction_loss'])-1, end_val),
                   xytext=(5, 10), textcoords='offset points', color=colors['reconstruction_loss'], fontsize=10)

        if val_metrics_exist and 'val_reconstruction_loss' in metrics:
            val_end = history.history['val_reconstruction_loss'][-1]
            ax.annotate(f'End: {val_end:.4f}', xy=(len(history.history['reconstruction_loss'])-1, val_end),
                       xytext=(5, -15), textcoords='offset points', color=colors['reconstruction_loss'], fontsize=10)

        ax_idx += 1

    # Plot 3: KL-Divergenz (nur für UnsupervisedVAE und SupervisedVAE)
    if 'kl_loss' in metrics and ax_idx < n_plots and model_type in ['uvae', 'svae']:
        ax = axes[ax_idx]
        ax.plot(history.history['kl_loss'], color=colors['kl_loss'], linestyle=line_styles['training'],
                linewidth=2, label='Training')
        if val_metrics_exist and 'val_kl_loss' in metrics:
            ax.plot(history.history['val_kl_loss'], color=colors['kl_loss'], linestyle=line_styles['validation'],
                    linewidth=2, label='Validierung')

        ax.set_title('KL-Divergenz', fontsize=14)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(alpha=0.3)

        # Annotations für Endwerte
        end_val = history.history['kl_loss'][-1]
        ax.annotate(f'End: {end_val:.4f}', xy=(len(history.history['kl_loss'])-1, end_val),
                   xytext=(5, 10), textcoords='offset points', color=colors['kl_loss'], fontsize=10)

        if val_metrics_exist and 'val_kl_loss' in metrics:
            val_end = history.history['val_kl_loss'][-1]
            ax.annotate(f'End: {val_end:.4f}', xy=(len(history.history['kl_loss'])-1, val_end),
                       xytext=(5, -15), textcoords='offset points', color=colors['kl_loss'], fontsize=10)

        ax_idx += 1

    # Plot 4: Klassifikationsverlust (nur für SupervisedAE und SupervisedVAE)
    if 'classification_loss' in metrics and ax_idx < n_plots and model_type in ['sae', 'svae']:
        ax = axes[ax_idx]
        ax.plot(history.history['classification_loss'], color=colors['classification_loss'], linestyle=line_styles['training'],
                linewidth=2, label='Training')
        if val_metrics_exist and 'val_classification_loss' in metrics:
            ax.plot(history.history['val_classification_loss'], color=colors['classification_loss'], linestyle=line_styles['validation'],
                    linewidth=2, label='Validierung')

        ax.set_title('Klassifikationsverlust', fontsize=14)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(alpha=0.3)

        # Annotations für Endwerte
        end_val = history.history['classification_loss'][-1]
        ax.annotate(f'End: {end_val:.4f}', xy=(len(history.history['classification_loss'])-1, end_val),
                   xytext=(5, 10), textcoords='offset points', color=colors['classification_loss'], fontsize=10)

        if val_metrics_exist and 'val_classification_loss' in metrics:
            val_end = history.history['val_classification_loss'][-1]
            ax.annotate(f'End: {val_end:.4f}', xy=(len(history.history['classification_loss'])-1, val_end),
                       xytext=(5, -15), textcoords='offset points', color=colors['classification_loss'], fontsize=10)

        ax_idx += 1

    # Gesamttitel
    plt.suptitle(f'Trainingsverlauf - {model_type.upper()} Modell', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Platz für den Suptitle
    
    save_path = None
    
    # Speichern des Plots in einem Google Drive-Ordner
    if save_path is None:
        folder_path = os.path.join('Results', f'{experiment_name}')

        # Erstelle den Ordner, falls er nicht existiert
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Generiere einen eindeutigen Dateinamen basierend auf dem Modelltyp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(folder_path, f"loss_curves_{model_type}_{timestamp}.png")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss-Kurven wurden gespeichert unter: {save_path}")
        #plt.close()  # Speicherfreigabe in Google Colab

    return fig, axes