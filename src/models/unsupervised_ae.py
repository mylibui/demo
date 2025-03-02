import tensorflow as tf
from tensorflow.keras import layers, Model, metrics
from typing import List, Optional, Union, Tuple, Dict

class UnsupervisedAE(Model):
    """
    Unüberwachter Autoencoder für Anomalieerkennung mit anpassbarer latenter Dimension.

    Args:
        input_dim (int): Dimension der Eingabedaten.
        hidden_dims (List[int]): Liste der Dimensionen der versteckten Schichten im Encoder/Decoder (Standard: [64, 32]).
        latent_dim (int): Dimension des latenten Raums (anpassbar, Standard: 32).
        activation (str, optional): Aktivierungsfunktion ('relu', 'mish', 'swish'). Standardwert ist 'relu'.
        dropout_rate (float, optional): Dropout-Rate zwischen 0 und 1. Standardwert ist 0.0.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32],
        latent_dim: int = 16,
        activation: str = 'relu',
        dropout_rate: float = 0.0
    ):
        super(UnsupervisedAE, self).__init__()

        # Eingabevalidierung
        if not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError("input_dim muss eine positive Ganzzahl sein")
        if not isinstance(hidden_dims, (list, tuple)) or not all(isinstance(d, int) and d > 0 for d in hidden_dims):
            raise ValueError("hidden_dims muss eine Liste positiver Ganzzahlen sein")
        if not isinstance(latent_dim, int) or latent_dim <= 0:
            raise ValueError("latent_dim muss eine positive Ganzzahl sein")
        if dropout_rate < 0 or dropout_rate >= 1:
            raise ValueError("dropout_rate muss zwischen 0 und 1 liegen")
        if activation not in ['relu', 'mish', 'swish']:
            raise ValueError("activation muss 'relu', 'mish' oder 'swish' sein")

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

        # Encoder-Schichten mit optionalem Dropout
        self.encoder_layers = [
            layers.Dense(dim, activation=self.activation if callable(self.activation) else self.activation)
            for dim in hidden_dims
        ]
        if dropout_rate > 0:
            self.encoder_dropout = layers.Dropout(dropout_rate)

        # Latente Schicht im Encoder
        self.latent_layer = layers.Dense(latent_dim, activation=None)  # Keine Aktivierung für latenten Raum

        # Decoder-Schichten mit optionalem Dropout
        self.decoder_layers = [
            layers.Dense(dim, activation=self.activation if callable(self.activation) else self.activation)
            for dim in reversed(hidden_dims)
        ]
        if dropout_rate > 0:
            self.decoder_dropout = layers.Dropout(dropout_rate)

        self.decoder_output = layers.Dense(input_dim, activation='sigmoid')

        # Metriken
        self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")

    @property
    def metrics(self) -> List[metrics.Metric]:
        """Gibt die Liste der Metriken zurück."""
        return [self.reconstruction_loss_tracker]

    def encode(self, x: tf.Tensor) -> tf.Tensor:
        """
        Kodierung der Eingabedaten in den latenten Raum.

        Args:
            x: Eingabetensor der Form (Batch-Größe, input_dim).

        Returns:
            Latenter Tensor der Form (Batch-Größe, latent_dim).
        """
        for layer in self.encoder_layers:
            x = layer(x)
            if self.dropout_rate > 0:
                x = self.encoder_dropout(x, training=True)
        return self.latent_layer(x)

    def decode(self, z: tf.Tensor) -> tf.Tensor:
        """
        Dekodierung der latenten Variablen in rekonstruierte Daten.

        Args:
            z: Latenter Tensor der Form (Batch-Größe, latent_dim).

        Returns:
            Rekonstruierter Tensor der Form (Batch-Größe, input_dim).
        """
        x = z
        for layer in self.decoder_layers:
            x = layer(x)
            if self.dropout_rate > 0 and layer != self.decoder_layers[-1]:  # Kein Dropout in der letzten Schicht
                x = self.decoder_dropout(x, training=True)
        return self.decoder_output(x)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Vorwärtspass durch den Autoencoder.

        Args:
            inputs: Eingabetensor der Form (Batch-Größe, input_dim).
            training: Boolean, der angibt, ob Trainings- oder Inferenzmodus vorliegt.

        Returns:
            Rekonstruierter Tensor der Form (Batch-Größe, input_dim).
        """
        encoded = self.encode(inputs)
        reconstructed = self.decode(encoded)
        return reconstructed

    def train_step(self, data: Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]) -> Dict[str, float]:
        """
        Trainings-Schritt für den Autoencoder mit MSE-Verlust.

        Args:
            data: Eingabetensor der Trainingsdaten oder Tuple (x, y) für TensorFlow 2.18.

        Returns:
            Dictionary mit Metrikwerten.
        """
        if isinstance(data, tuple):
            if len(data) == 2:
                x, _ = data  # Ignoriere y, da es unüberwacht ist
            else:
                x = data[0]
        else:
            x = data

        with tf.GradientTape() as tape:
            reconstruction = self.call(x, training=True)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x - reconstruction), axis=-1))

        grads = tape.gradient(reconstruction_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.reconstruction_loss_tracker.update_state(reconstruction_loss)

        return {"reconstruction_loss": self.reconstruction_loss_tracker.result()}

    def test_step(self, data: Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]) -> Dict[str, float]:
        """
        Test-Schritt für den Autoencoder mit MSE-Verlust.

        Args:
            data: Eingabetensor der Testdaten oder Tuple (x, y) für TensorFlow 2.18.

        Returns:
            Dictionary mit Metrikwerten.
        """
        if isinstance(data, tuple):
            if len(data) == 2:
                x, _ = data  # Ignoriere y, da es unüberwacht ist
            else:
                x = data[0]
        else:
            x = data

        reconstruction = self.call(x, training=False)
        reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x - reconstruction), axis=-1))

        self.reconstruction_loss_tracker.update_state(reconstruction_loss)

        return {"reconstruction_loss": self.reconstruction_loss_tracker.result()}

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
        