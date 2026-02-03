# models/neural_network.py
from tensorflow import keras
from tensorflow.keras import layers

class NeuralNetworkModel:

    def __init__(self, input_dim, learning_rate="adam"):
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        model = keras.Sequential([
            keras.Input(shape=(self.input_dim,)),

            layers.Dense(1024, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Dense(1024, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Dense(1024, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Dense(1, activation="sigmoid")
        ])

        model.compile(
            optimizer=self.learning_rate,
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        return model

    def train(
        self,
        x_train,
        y_train,
        x_validation,
        y_validation,
        epochs=10,
        batch_size=256,
        verbose=1
    ):
        history = self.model.fit(
            x_train,
            y_train,
            validation_data=(x_validation, y_validation),
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )
        return history

    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype("int32")

    def predict_proba(self, X):
        return self.model.predict(X)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = keras.models.load_model(path)
