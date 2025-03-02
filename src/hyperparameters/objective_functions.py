import numpy as np
from sklearn.metrics import roc_auc_score
import tensorflow as tf

from ..models import SupervisedAE, SupervisedVAE, UnsupervisedVAE, UnsupervisedAE, calculate_reconstruction_error
from .create_model import create_model

def build_objective_uae(trial, X_train, y_train, skf):
    return lambda trial: objective_uae(trial, X_train, y_train, skf)

def build_objective_uvae(trial, X_train, y_train, skf):
    return lambda trial: objective_uvae(trial, X_train, y_train, skf)

def build_objective_sae(trial, X_train, y_train, skf):
    return lambda trial: objective_sae(trial, X_train, y_train, skf)

def objective_uae(trial, X_train, y_train, skf):
    # Hyperparameter-Suchraum
    latent_dim = trial.suggest_categorical('latent_dim', [2, 8, 16])
    activation = trial.suggest_categorical('activation', ['relu', 'mish',"swish"])
    learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.01, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.2)

    # Initialisiere ROC-AUC über die Folds
    roc_auc_scores = []

    # StratifiedKFold für robuste Evaluierung
    for train_idx, val_idx in skf.split(X_train, y_train.flatten()):
        X_train_fold = X_train[train_idx]
        y_train_fold = y_train[train_idx]
        X_val_fold = X_train[val_idx]
        y_val_fold = y_train[val_idx]

        # Erstelle und kompiliere das Modell
        model = create_model(UnsupervisedAE,latent_dim, activation, dropout_rate)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

        # Baue das Modell
        X_train_normal_fold = X_train_fold[y_train_fold.flatten() == 0]
        if len(X_train_normal_fold) == 0:
            raise ValueError("Keine normalen Daten gefunden!")
        model.build(input_shape=(None, X_train_normal_fold.shape[1]))
        history = model.fit(X_train_normal_fold, epochs=5, batch_size=128, validation_data=(X_val_fold[y_val_fold.flatten() == 0], None), verbose=0)

        # Berechne Rekonstruktionsfehler für Validierungsdaten
        recon_error = calculate_reconstruction_error(model, X_val_fold)
        if np.any(np.isnan(recon_error)):
            recon_error = np.nan_to_num(recon_error, nan=np.mean(recon_error))
        roc_auc = roc_auc_score(y_val_fold.flatten(), recon_error)

        roc_auc_scores.append(roc_auc)

    # Rückgabe des durchschnittlichen ROC-AUC über alle Folds
    mean_roc_auc = np.mean(roc_auc_scores)
    return mean_roc_auc

def objective_uvae(trial, X_train, y_train, skf):
    # Hyperparameter-Suchraum
    latent_dim = trial.suggest_categorical('latent_dim', [2, 8, 16])
    activation = trial.suggest_categorical('activation', ['relu', 'mish',"swish"])
    learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.01, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.2)
    kl_weight = trial.suggest_float('kl_weight', 0.5, 2.0)
    # Initialisiere ROC-AUC über die Folds
    roc_auc_scores = []

    # StratifiedKFold für robuste Evaluierung
    for train_idx, val_idx in skf.split(X_train, y_train.flatten()):
        X_train_fold = X_train[train_idx]
        y_train_fold = y_train[train_idx]
        X_val_fold = X_train[val_idx]
        y_val_fold = y_train[val_idx]

        # Erstelle und kompiliere das Modell
        model = create_model(UnsupervisedVAE,latent_dim, activation, dropout_rate)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

        # Baue das Modell
        X_train_normal_fold = X_train_fold[y_train_fold.flatten() == 0]
        if len(X_train_normal_fold) == 0:
            raise ValueError("Keine normalen Daten gefunden!")
        model.build(input_shape=(None, X_train_normal_fold.shape[1]))
        history = model.fit(X_train_normal_fold, epochs=5, batch_size=64, validation_data=(X_val_fold[y_val_fold.flatten() == 0], None), verbose=0)

        # Berechne Rekonstruktionsfehler für Validierungsdaten
        recon_error = calculate_reconstruction_error(model, X_val_fold)
        if np.any(np.isnan(recon_error)):
            recon_error = np.nan_to_num(recon_error, nan=np.finfo(recon_error.dtype).max)
        roc_auc = roc_auc_score(y_val_fold.flatten(), recon_error)

        roc_auc_scores.append(roc_auc)

    # Rückgabe des durchschnittlichen ROC-AUC über alle Folds
    mean_roc_auc = np.mean(roc_auc_scores)
    return mean_roc_auc


def objective_sae(trial, X_train, y_train, skf):
    # Hyperparameter-Suchraum
    latent_dim = trial.suggest_categorical('latent_dim', [2, 8, 16])
    activation = trial.suggest_categorical('activation', ['relu', 'mish', 'swish'])
    learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.01, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.2)
    classifier_dims = [16,8]
    l2_lambda = trial.suggest_float('l2_lambda', 0.001, 0.1, log=True)  # L2-Regularisierungsfaktor optimieren
    # Initialisiere ROC-AUC über die Folds
    roc_auc_scores = []

    # StratifiedKFold für robuste Evaluierung
    for train_idx, val_idx in skf.split(X_train, y_train.flatten()):
        X_train_fold = X_train[train_idx]
        y_train_fold = y_train[train_idx]
        X_val_fold = X_train[val_idx]
        y_val_fold = y_train[val_idx]

        # Erstelle und kompiliere das Modell
        model = create_model(SupervisedAE,latent_dim,
            classifier_dims=classifier_dims,  # Pass classifier_dims here
            activation=activation,
            dropout_rate=dropout_rate)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
        # Baue das Modell
        model.build(input_shape=(None, X_train_fold.shape[1]))
        history = model.fit(X_train_fold, y_train_fold, epochs=5, batch_size=64, validation_data=(X_val_fold, y_val_fold), verbose=0)

        # Berechne Klassifikations-ROC-AUC
        preds = (model.classify(model.encode(tf.convert_to_tensor(X_val_fold, dtype=tf.float32)))>0.5).numpy()
        roc_auc = roc_auc_score(y_val_fold.flatten(), preds)

        roc_auc_scores.append(roc_auc)

    # Rückgabe des durchschnittlichen ROC-AUC über alle Folds
    mean_roc_auc = np.mean(roc_auc_scores)
    return mean_roc_auc