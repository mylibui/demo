import tensorflow as tf
from tensorflow.keras import layers, Model, metrics
from typing import List, Optional, Union, Tuple, Dict
class SupervisedVAE(Model):
    """
    Überwachter Variational Autoencoder für Anomalieerkennung mit Klassifikation.

    Args:
        input_dim (int): Dimension der Eingabedaten.
        hidden_dims (List[int]): Liste der Dimensionen der versteckten Schichten im Encoder/Decoder (Standard: [64, 32]).
        latent_dim (int): Dimension des latenten Raums (Standard: 16).
        classifier_dims (List[int]): Liste der Dimensionen der versteckten Schichten im Klassifikator (Standard: [16]).
        activation (str, optional): Aktivierungsfunktion ('relu', 'mish', 'swish'). Standardwert ist 'relu'.
        dropout_rate (float, optional): Dropout-Rate zwischen 0 und 1. Standardwert ist 0.0.
        kl_weight (float, optional): Gewichtungsfaktor für den KL-Verlust. Standardwert ist 1.0.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32],
        latent_dim: int = 16,
        classifier_dims: List[int] = [16,8],
        activation: str = 'relu',
        dropout_rate: float = 0.0,
        kl_weight: float = 1.0
    ):
        super(SupervisedVAE, self).__init__()
        self.type = 'svae'

        # Aktivierungsfunktion direkt definieren
        if activation == 'mish':
            self.activation = lambda x: x * tf.math.tanh(tf.math.softplus(x))
        elif activation == 'swish':
            self.activation = lambda x: x * tf.nn.sigmoid(x)
        elif activation == 'relu':
            self.activation = 'relu'
        else:
            raise ValueError(f"Unbekannte Aktivierungsfunktion: {activation}")

        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.kl_weight = kl_weight

        # Encoder-Schichten mit optionalem Dropout
        self.encoder_layers = [
            layers.Dense(dim, activation=self.activation if callable(self.activation) else self.activation, dtype=tf.float32)
            for dim in hidden_dims
        ]
        if dropout_rate > 0:
            self.encoder_dropout = layers.Dropout(dropout_rate)

        self.z_mean = layers.Dense(latent_dim, dtype=tf.float32)
        self.z_log_var = layers.Dense(latent_dim, dtype=tf.float32)

        class Sampling(layers.Layer):
            def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
                z_mean, z_log_var = inputs
                batch = tf.shape(z_mean)[0]
                dim = tf.shape(z_mean)[1]
                epsilon = tf.random.normal(shape=(batch, dim), dtype=tf.float32)
                return z_mean + tf.exp(0.5 * z_log_var) * epsilon

        self.sampling = Sampling()

        # Decoder-Schichten mit optionalem Dropout
        self.decoder_layers = [
            layers.Dense(dim, activation=self.activation if callable(self.activation) else self.activation, dtype=tf.float32)
            for dim in reversed(hidden_dims)
        ]
        if dropout_rate > 0:
            self.decoder_dropout = layers.Dropout(dropout_rate)

        self.decoder_output = layers.Dense(input_dim, activation='linear', dtype=tf.float32)

        # Klassifikator-Schichten mit optionalem Dropout
        #self.classifier_layers = [
        #    layers.Dense(dim, activation=self.activation if callable(self.activation) else self.activation, dtype=tf.float32)
        #    for dim in classifier_dims
        #]
        #if dropout_rate > 0:
        #    self.classifier_dropout = layers.Dropout(dropout_rate)

        self.classifier_output = layers.Dense(1, activation='sigmoid', dtype=tf.float32)

        # Metriken
        self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss", dtype=tf.float32)
        self.classification_loss_tracker = metrics.Mean(name="classification_loss", dtype=tf.float32)
        self.kl_loss_tracker = metrics.Mean(name="kl_loss", dtype=tf.float32)
        self.total_loss_tracker = metrics.Mean(name="total_loss", dtype=tf.float32)

    @property
    def metrics(self) -> List[metrics.Metric]:
        """Gibt die Liste der Metriken zurück."""
        return [
            self.reconstruction_loss_tracker,
            self.classification_loss_tracker,
            self.kl_loss_tracker,
            self.total_loss_tracker,
        ]

    def encode(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Kodierung der Eingabedaten in den latenten Raum.

        Args:
            x: Eingabetensor der Form (Batch-Größe, input_dim), erwartet tf.float32.

        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: (z_mean, z_log_var, z) – Mittelwert, Log-Varianz und gestochastische latente Variable, alle als tf.float32.
        """
        if x.dtype != tf.float32:
            x = tf.cast(x, tf.float32)  # Konvertiere explizit zu float32
        for layer in self.encoder_layers:
            x = layer(x)
            if self.dropout_rate > 0:
                x = self.encoder_dropout(x, training=True)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.sampling([z_mean, z_log_var])
        return z_mean, z_log_var, z

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
            if self.dropout_rate > 0 and layer != self.decoder_layers[-1]:  # Kein Dropout in der letzten Schicht
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
        #if self.dropout_rate > 0:
         #       x = self.classifier_dropout(x, training=True)
        return self.classifier_output(x)

    def call(self, inputs: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Vorwärtspass durch den SupervisedVAE.

        Args:
            inputs: Eingabetensor der Form (Batch-Größe, input_dim), erwartet tf.float32.
            training: Boolean, der angibt, ob Trainings- oder Inferenzmodus vorliegt.

        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]: (rekonstruiert, z_mean, z_log_var, klassifiziert) – Rekonstruierte Daten, Mittelwert, Log-Varianz des latenten Raums und Klassifikationswahrscheinlichkeiten, alle als tf.float32.
        """
        if inputs.dtype != tf.float32:
            inputs = tf.cast(inputs, tf.float32)  # Konvertiere explizit zu float32
        z_mean, z_log_var, z = self.encode(inputs)
        reconstructed = self.decode(z)
        classification = self.classify(z_mean)  # Verwende z_mean für Klassifikation
        return reconstructed, z_mean, z_log_var, classification

    def train_step(self, data: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, float]:
        """
        Trainings-Schritt für den SupervisedVAE mit MSE-, Binary Crossentropy- und KL-Verlust.
        """
        # Debugging und Datenextraktion (wie zuvor)
        x, y = data
        if x.dtype != tf.float32:
            x = tf.cast(x, tf.float32)
        if y.dtype != tf.float32:
            y = tf.cast(y, tf.float32)
        if len(y.shape) == 1:
            y = tf.expand_dims(y, axis=-1)

        with tf.GradientTape() as tape:
            reconstructed, z_mean, z_log_var, classification = self.call(x, training=True)

            # Rekonstruktionsverlust (MSE)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x - reconstructed), axis=-1))

            # Klassifikationsverlust (Binary Crossentropy)
            classification_loss = tf.keras.losses.binary_crossentropy(y, classification)

            # KL-Verlust
            kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1))

            # Gesamtverlust
            total_loss = reconstruction_loss + tf.reduce_mean(classification_loss) + self.kl_weight * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Aktualisiere Metriken, einschließlich 'loss'
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.classification_loss_tracker.update_state(classification_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.total_loss_tracker.update_state(total_loss)  # Speichere 'loss' als total_loss

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "classification_loss": self.classification_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, float]:
        """
        Test-Schritt für den SupervisedVAE mit MSE-, Binary Crossentropy- und KL-Verlust.

        Args:
            data: Tuple (x, y) mit Eingabedaten und Labels für TensorFlow 2.18, beide als tf.float32.

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
            y = tf.expand_dims(y, axis=-1)  # Füge eine Dimension hinzu: (batch_size,) -> (batch_size, 1)

        reconstructed, z_mean, z_log_var, classification = self.call(x, training=False)

        # Rekonstruktionsverlust (MSE)
        reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x - reconstructed), axis=-1))

        # Klassifikationsverlust (Binary Crossentropy)
        classification_loss = tf.keras.losses.binary_crossentropy(y, classification)

        # KL-Verlust
        kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1))

        # Gesamtverlust
        total_loss = reconstruction_loss + tf.reduce_mean(classification_loss) + self.kl_weight * kl_loss

        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.classification_loss_tracker.update_state(classification_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.total_loss_tracker.update_state(total_loss)  # Speichere 'loss' als total_loss

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "classification_loss": self.classification_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
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
        self.call(dummy_input, training=False)  # Baut das Modell, indem es auf Dummy-Daten angewendet wird
