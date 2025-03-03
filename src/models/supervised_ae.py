import tensorflow as tf
from tensorflow.keras import layers, Model, metrics, regularizers
from typing import List, Optional, Union, Tuple, Dict


class SupervisedAE(Model):
    """
    Überwachter Autoencoder für Anomalieerkennung mit Klassifikation und L2-Regularisierung für den Klassifikationsverlust.

    Args:
        input_dim (int): Dimension der Eingabedaten.
        hidden_dims (List[int]): Liste der Dimensionen der versteckten Schichten im Encoder/Decoder (Standard: [64, 32]).
        latent_dim (int): Dimension des latenten Raums (Standard: 32).
        classifier_dims (List[int]): Liste der Dimensionen der versteckten Schichten im Klassifikator (Standard: [16, 8]).
        activation (str, optional): Aktivierungsfunktion ('relu', 'mish', 'swish'). Standardwert ist 'relu'.
        dropout_rate (float, optional): Dropout-Rate zwischen 0 und 1. Standardwert ist 0.0.
        l2_lambda (float, optional): L2-Regularisierungsfaktor für den Klassifikator (Standard: 0.01).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32],
        latent_dim: int = 32,
        classifier_dims: List[int] = [16, 8],
        activation: str = "relu",
        dropout_rate: float = 0.0,
        l2_lambda: float = 0.01,  # L2-Regularisierungsfaktor für den Klassifikator
    ):
        super(SupervisedAE, self).__init__()
        self.type = "sae"

        # Aktivierungsfunktion direkt definieren
        if activation == "mish":
            self.activation = lambda x: x * tf.math.tanh(tf.math.softplus(x))
        elif activation == "swish":
            self.activation = lambda x: x * tf.nn.sigmoid(x)
        elif activation == "relu":
            self.activation = "relu"
        else:
            raise ValueError(f"Unbekannte Aktivierungsfunktion: {activation}")

        self.dropout_rate = dropout_rate
        self.l2_lambda = l2_lambda  # L2-Regularisierungsfaktor speichern

        # Encoder-Schichten mit optionalem Dropout
        self.encoder_layers = [
            layers.Dense(
                dim,
                activation=(
                    self.activation if callable(self.activation) else self.activation
                ),
                dtype=tf.float32,
            )
            for dim in hidden_dims
        ]
        if dropout_rate > 0:
            self.encoder_dropout = layers.Dropout(dropout_rate)

        # Latente Schicht im Encoder
        self.latent_layer = layers.Dense(
            latent_dim, activation=None, dtype=tf.float32
        )  # Keine Aktivierung für latenten Raum

        # Decoder-Schichten mit optionalem Dropout
        self.decoder_layers = [
            layers.Dense(
                dim,
                activation=(
                    self.activation if callable(self.activation) else self.activation
                ),
                dtype=tf.float32,
            )
            for dim in reversed(hidden_dims)
        ]
        if dropout_rate > 0:
            self.decoder_dropout = layers.Dropout(dropout_rate)

        self.decoder_output = layers.Dense(
            input_dim, activation="linear", dtype=tf.float32
        )

        # Klassifikator-Schichten mit L2-Regularisierung und optionalem Dropout
        self.classifier_layers = [
            layers.Dense(
                dim,
                activation=self.activation,
                kernel_regularizer=regularizers.l2(l2_lambda),
                dtype=tf.float32,
            )
            for dim in classifier_dims
        ]
        if dropout_rate > 0:
            self.classifier_dropout = layers.Dropout(dropout_rate)

        self.classifier_output = layers.Dense(
            1,
            activation="sigmoid",
            kernel_regularizer=regularizers.l2(l2_lambda),
            dtype=tf.float32,
        )

        # Metriken
        self.reconstruction_loss_tracker = metrics.Mean(
            name="reconstruction_loss", dtype=tf.float32
        )
        self.classification_loss_tracker = metrics.Mean(
            name="classification_loss", dtype=tf.float32
        )
        self.total_loss_tracker = metrics.Mean(name="total_loss", dtype=tf.float32)
        self.regularization_loss_tracker = metrics.Mean(
            name="regularization_loss", dtype=tf.float32
        )  # Neue Metrik für Regularisierung

    @property
    def metrics(self) -> List[metrics.Metric]:
        """Gibt die Liste der Metriken zurück."""
        return [
            self.reconstruction_loss_tracker,
            self.classification_loss_tracker,
            self.total_loss_tracker,
            self.regularization_loss_tracker,
        ]

    def encode(self, x: tf.Tensor) -> tf.Tensor:
        """
        Kodierung der Eingabedaten in den latenten Raum.

        Args:
            x: Eingabetensor der Form (Batch-Größe, input_dim), erwartet tf.float32.

        Returns:
            Latenter Tensor der Form (Batch-Größe, latent_dim), als tf.float32.
        """
        if x.dtype != tf.float32:
            x = tf.cast(x, tf.float32)  # Konvertiere explizit zu float32
        for layer in self.encoder_layers:
            x = layer(x)
            if self.dropout_rate > 0:
                x = self.encoder_dropout(x, training=True)
        return self.latent_layer(x)

    def decode(self, z: tf.Tensor) -> tf.Tensor:
        """
        Dekodierung der latenten Variablen in rekonstruierte Daten.

        Args:
            z: Latenter Tensor der Form (Batch-Größe, latent_dim), erwartet tf.float32.

        Returns:
            Rekonstruierter Tensor der Form (Batch-Größe, input_dim), als tf.float32.
        """
        if z.dtype != tf.float32:
            z = tf.cast(z, tf.float32)  # Konvertiere explizit zu float32
        x = z
        for layer in self.decoder_layers:
            x = layer(x)
            if (
                self.dropout_rate > 0 and layer != self.decoder_layers[-1]
            ):  # Kein Dropout in der letzten Schicht
                x = self.decoder_dropout(x, training=True)
        return self.decoder_output(x)

    def classify(self, z: tf.Tensor) -> tf.Tensor:
        """
        Klassifikation der latenten Variablen in Betrugswahrscheinlichkeiten.

        Args:
            z: Latenter Tensor der Form (Batch-Größe, latent_dim), erwartet tf.float32.

        Returns:
            Tensor der Form (Batch-Größe, 1) mit Wahrscheinlichkeiten (0-1), als tf.float32.
        """
        if z.dtype != tf.float32:
            z = tf.cast(z, tf.float32)  # Konvertiere explizit zu float32
        x = z
        for layer in self.classifier_layers:
            x = layer(x)
            if self.dropout_rate > 0:
                x = self.classifier_dropout(x, training=True)
        return self.classifier_output(x)

    def call(
        self, inputs: tf.Tensor, training: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Vorwärtspass durch den SupervisedAE.

        Args:
            inputs: Eingabetensor der Form (Batch-Größe, input_dim), erwartet tf.float32.
            training: Boolean, der angibt, ob Trainings- oder Inferenzmodus vorliegt.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: (rekonstruiert, klassifiziert) – Rekonstruierte Daten und Klassifikationswahrscheinlichkeiten, beide als tf.float32.
        """
        if inputs.dtype != tf.float32:
            inputs = tf.cast(inputs, tf.float32)  # Konvertiere explizit zu float32
        encoded = self.encode(inputs)
        reconstructed = self.decode(encoded)
        classification = self.classify(encoded)
        return reconstructed, classification

    def train_step(self, data: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """
        Trainings-Schritt für den SupervisedAE mit MSE- und Binary Crossentropy-Verlust sowie L2-Regularisierung.

        Args:
            data: Tuple (x, y) mit Eingabedaten und Labels für TensorFlow 2.x, beide als tf.float32.

        Returns:
            Dictionary mit Metrikwerten.
        """
        x, y = data

        if x.dtype != tf.float32:
            x = tf.cast(x, tf.float32)  # Konvertiere explizit zu float32
        if y.dtype != tf.float32:
            y = tf.cast(y, tf.float32)  # Konvertiere explizit zu float32

        # Formatiere y zu Shape (batch_size, 1), um mit der Klassifikationsausgabe übereinzustimmen
        if len(y.shape) == 1:
            y = tf.expand_dims(
                y, axis=-1
            )  # Füge eine Dimension hinzu: (batch_size,) -> (batch_size, 1)

        with tf.GradientTape() as tape:
            reconstructed, classification = self(x, training=True)

            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(x - reconstructed), axis=-1)
            )
            classification_loss = tf.keras.losses.binary_crossentropy(y, classification)

            # Berechne Regularisierungsloss für den Klassifikator
            regularization_loss = tf.reduce_sum(
                [
                    tf.reduce_sum(tf.nn.l2_loss(w))
                    for w in self.trainable_weights
                    if "classifier" in w.name
                ]
            )
            total_loss = (
                reconstruction_loss
                + tf.reduce_mean(classification_loss)
                + (self.l2_lambda * regularization_loss)
            )

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.classification_loss_tracker.update_state(classification_loss)
        self.total_loss_tracker.update_state(total_loss)
        self.regularization_loss_tracker.update_state(regularization_loss)

        return {
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "classification_loss": self.classification_loss_tracker.result(),
            "total_loss": self.total_loss_tracker.result(),
            "regularization_loss": self.regularization_loss_tracker.result(),
        }

    def test_step(self, data: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """
        Test-Schritt für den SupervisedAE mit MSE- und Binary Crossentropy-Verlust sowie L2-Regularisierung.

        Args:
            data: Tuple (x, y) mit Eingabedaten und Labels für TensorFlow 2.x, beide als tf.float32.

        Returns:
            Dictionary mit Metrikwerten.
        """
        x, y = data

        if x.dtype != tf.float32:
            x = tf.cast(x, tf.float32)  # Konvertiere explizit zu float32
        if y.dtype != tf.float32:
            y = tf.cast(y, tf.float32)  # Konvertiere explizit zu float32

        # Formatiere y zu Shape (batch_size, 1), um mit der Klassifikationsausgabe übereinzustimmen
        if len(y.shape) == 1:
            y = tf.expand_dims(
                y, axis=-1
            )  # Füge eine Dimension hinzu: (batch_size,) -> (batch_size, 1)

        reconstructed, classification = self(x, training=False)

        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(x - reconstructed), axis=-1)
        )
        classification_loss = tf.keras.losses.binary_crossentropy(y, classification)

        # Berechne Regularisierungsloss für den Klassifikator
        regularization_loss = tf.reduce_sum(
            [
                tf.reduce_sum(tf.nn.l2_loss(w))
                for w in self.trainable_weights
                if "classifier" in w.name
            ]
        )
        total_loss = (
            reconstruction_loss
            + tf.reduce_mean(classification_loss)
            + (self.l2_lambda * regularization_loss)
        )

        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.classification_loss_tracker.update_state(classification_loss)
        self.total_loss_tracker.update_state(total_loss)
        self.regularization_loss_tracker.update_state(regularization_loss)

        return {
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "classification_loss": self.classification_loss_tracker.result(),
            "total_loss": self.total_loss_tracker.result(),
            "regularization_loss": self.regularization_loss_tracker.result(),
        }

    def print_summary(self):
        """
        Gibt eine Zusammenfassung der Modellarchitektur aus.

        Diese Methode verwendet die `summary()`-Methode von TensorFlow/Keras, um die Layer, Parameter und Ausgabetypen anzuzeigen.
        """
        self.summary()

    def build(self, input_shape: Tuple[int, int]) -> None:
        """
        Baut das Modell explizit mit der angegebenen Eingabeform.

        Args:
            input_shape: Tuple der Form (Batch-Größe, input_dim), wobei Batch-Größe optional ist.

        Raises:
            ValueError: Wenn input_shape kein gültiges Tuple ist.
        """
        if isinstance(input_shape, tuple) and len(input_shape) >= 1:
            input_dim = input_shape[-1]  # Nehme die letzte Dimension als input_dim
        else:
            raise ValueError("input_shape muss ein Tuple sein, z. B. (None, input_dim)")

        # Erstelle ein Dummy-Input-Tensor, um das Modell zu bauen
        dummy_input = tf.zeros((1, input_dim), dtype=tf.float32)
        self.call(
            dummy_input, training=False
        )  # Baut das Modell, indem es auf Dummy-Daten angewendet wird
