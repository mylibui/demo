from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split


@dataclass
class Data:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    X_train_normal: np.ndarray
    input_dim: int


def load(file_path: str) -> Data:
    df = pd.read_csv(file_path)

    # Normalize the 'Amount' column
    scaler = StandardScaler()
    df["Amount"] = scaler.fit_transform(df["Amount"].values.reshape(-1, 1))

    # Drop the 'Time' column as it is not useful for anomaly detection
    df = df.drop(["Time"], axis=1)

    # Separate features and labels
    X = df.drop(["Class"], axis=1).values
    y = df["Class"].values

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Only use normal transactions for training (Class = 0)
    X_train_normal = X_train[y_train == 0]
    return Data(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        X_train_normal=X_train_normal,
        input_dim=X_train.shape[0],
    )
