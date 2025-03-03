from typing import Optional
import tensorflow as tf
import numpy as np

from fraud_detection import evaluate_fraud_detection, find_optimal_threshold
from plot import (
    plot_latent_space,
    plot_loss_curves,
    plot_reconstruction_analysis,
    plot_roc_auc_curve,
)
from data import Data
from models import UnsupervisedAE, UnsupervisedVAE, SupervisedAE, SupervisedVAE


def train_model_and_evaluate(
    model_class,
    data: Data,
    hidden_dims: list[int],
    latent_dim: int,
    activation: str,
    dropout_rate: float,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    kl_weight: Optional[float] = None,
    classifier_dims: Optional[list[int]] = None,
):
    experiment_name = "tuned_hyperparameters"
    if model_class == UnsupervisedAE:
        model = model_class(
            input_dim=data.input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            activation=activation,
            dropout_rate=dropout_rate,
        )
    elif model_class == UnsupervisedVAE:
        model = model_class(
            input_dim=data.input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            activation=activation,
            dropout_rate=dropout_rate,
            kl_weight=kl_weight,
        )
    elif model_class == SupervisedAE:
        model = model_class(
            input_dim=data.input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            activation=activation,
            dropout_rate=dropout_rate,
            classifier_dims=classifier_dims,
        )
    elif model_class == SupervisedVAE:
        model = model_class(
            input_dim=data.input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            activation=activation,
            dropout_rate=dropout_rate,
            classifier_dims=classifier_dims,
            kl_weight=kl_weight,
        )
    else:
        raise ValueError(f"unknown model class {model_class}")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

    history = model.fit(
        data.X_train_normal,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1,
    )

    model.print_summary()
    plot_loss_curves(
        history,
        model_type=model.type,
        experiment_name=experiment_name,
    )
    best_threshold = find_optimal_threshold(model, data.X_test, data.y_test)
    plot_reconstruction_analysis(
        model,
        data.X_test,
        data.y_test,  # Labels für Testdaten (0 für Nicht Fraud, 1 für Fraud)
        threshold=best_threshold,  # Basierend auf Ihren vorherigen Histogram-Daten für AE
        experiment_name=experiment_name,
    )
    evaluate_fraud_detection(
        model,
        data.X_test,
        data.y_test,
        save_metrics=True,  # Speichert Metriken in Google Drive
        experiment_name=experiment_name,
    )
    plot_roc_auc_curve(
        model,
        data.X_test,
        data.y_test,  # Labels für Testdaten (0 für Nicht Fraud, 1 für Fraud)
        experiment_name=experiment_name,
    )
    plot_latent_space(
        model, data.X_test, np.squeeze(data.y_test), experiment_name=experiment_name
    )
