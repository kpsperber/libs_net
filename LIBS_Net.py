import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

class LIBS_Net:
    def __init__(self):
        self.model = None
        self.optimizer = None

        self.X_scale = None
        self.y_mean = None
        self.y_std = None

        self.X_train = None
        self.X_val = None
        self.X_test = None
        
        self.y_train = None
        self.y_val = None
        self.y_test = None

    def read_data(self, root_folder):
        self.root_folder = root_folder

        self.shape = tuple(np.load(os.path.join(root_folder, "shape.npy")).astype(int))

        labeled_data = pd.read_csv(os.path.join(root_folder, "labeled_data.csv"))

        labels = []
        for label in labeled_data.keys():
            labels.append(label.split("p")[0])

        labels = np.float32(labels)

        data = labeled_data.to_numpy().T.reshape(
            len(labels), *self.shape
        )

        self.X = []
        self.y = []

        for i in range(0, len(labels)):
            for j in range(0, len(data[i])):
                self.X.append(data[i, j])
                self.y.append(labels[i])

        self.X = np.array(self.X)
        self.y = np.array(self.y)

        fig, ax = plt.subplots()

        im = ax.imshow(data[0])
        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Image")

        fig.colorbar(im, ax = ax)

        plt.show()

        
        print("X shape:", self.X.shape)
        print("y shape:", self.y.shape)
        print("Labels:", np.unique(self.y, return_counts=True))


    def build_model(self):
        number_of_wavelengths = self.shape[1]

        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(
                shape=(number_of_wavelengths, 1)
            ),

            tf.keras.layers.Conv1D(
                filters=32,
                kernel_size=7,
                padding="same",
                activation="relu"
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(pool_size=2),

            tf.keras.layers.Conv1D(
                filters=64,
                kernel_size=7,
                padding="same",
                activation="relu"
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(pool_size=2),

            tf.keras.layers.Conv1D(
                filters=128,
                kernel_size=5,
                padding="same",
                activation="relu"
            ),

            # Converts the entire spectrum into one feature vector.
            tf.keras.layers.GlobalAveragePooling1D(),

            tf.keras.layers.Dense(
                64,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(1e-4)
            ),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Dense(
                16,
                activation="relu"
            ),

            # One scalar concentration.
            tf.keras.layers.Dense(1)
        ])

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=1e-3
        )

        self.model.compile(
            optimizer=self.optimizer,
            loss=tf.keras.losses.Huber(),
            metrics=[
                tf.keras.metrics.RootMeanSquaredError(
                    name="rmse"
                ),
                tf.keras.metrics.MeanAbsoluteError(
                    name="mae"
                )
            ]
        )

        self.model.summary()

    def train(self, epochs=300, batch_size=32):
        if self.model is None:
            raise RuntimeError(
                "Call build_model() before train()."
            )

        # Training parameters. This helps reduce the amount of epochs done while training and sets adaptive learning rate and more.
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=25,
                restore_best_weights=True
            ),

            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=8,
                min_lr=1e-6,
                verbose=1
            )
        ]

        # Train the model
        history = self.model.fit(
            self.X_train,
            self.y_train_scaled,
            validation_data=(
                self.X_val,
                self.y_val_scaled
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        return history
    
    def split_data(self, test_size = 0.2, validation_size = 0.2, random_state = 42):

        # Using scikit learns auto splitter because I'm lazy
        X_train_val, self.X_test, y_train_val, self.y_test = (
            train_test_split(
                self.X,
                self.y,
                test_size=test_size,
                random_state=random_state,
                shuffle=True,
                stratify=self.y
            )
        )

        relative_validation_size = validation_size / (1.0 - test_size)

        self.X_train, self.X_val, self.y_train, self.y_val = (
            train_test_split(
                X_train_val,
                y_train_val,
                test_size=relative_validation_size,
                random_state=random_state,
                shuffle=True,
                stratify=y_train_val
            )
        )

        # Establish this to normalize data. This helps regression not go crazy
        self.X_scale = np.max(np.abs(self.X_train))

        if self.X_scale == 0:
            raise ValueError("Training spectra contain only zeros.")

        # Rescale training and testing data
        self.X_train = self.X_train / self.X_scale
        self.X_val = self.X_val / self.X_scale
        self.X_test = self.X_test / self.X_scale

        # Rescale y to further help the loss function not explode
        self.y_mean = np.mean(self.y_train)
        self.y_std = np.std(self.y_train)

        if self.y_std == 0:
            raise ValueError(
                "Training targets all have the same concentration."
            )

        self.y_train_scaled = (
            self.y_train - self.y_mean
        ) / self.y_std

        self.y_val_scaled = (
            self.y_val - self.y_mean
        ) / self.y_std

        self.y_test_scaled = (
            self.y_test - self.y_mean
        ) / self.y_std

        # Resizing for tensor flow
        self.X_train = self.X_train[..., np.newaxis]
        self.X_val = self.X_val[..., np.newaxis]
        self.X_test = self.X_test[..., np.newaxis]

        print("Training shape:", self.X_train.shape)
        print("Validation shape:", self.X_val.shape)
        print("Test shape:", self.X_test.shape)


    def predict(self, X):
        if self.model is None:
            raise RuntimeError("The model has not been built.")

        if self.X_scale is None:
            raise RuntimeError(
                "The preprocessing scale has not been fitted."
            )

        X = np.asarray(X, dtype=np.float32)

        if X.ndim == 1:
            X = X[np.newaxis, :]

        if X.ndim != 2:
            raise ValueError(
                "X must have shape (wavelengths,) or "
                "(samples, wavelengths)."
            )

        if X.shape[1] != self.shape[1]:
            raise ValueError(
                f"Expected {self.shape[1]} wavelength bins, "
                f"but received {X.shape[1]}."
            )

        X = X / self.X_scale
        X = X[..., np.newaxis]

        predictions_scaled = self.model.predict(
            X,
            verbose=0
        ).reshape(-1)

        predictions = (
            predictions_scaled * self.y_std
            + self.y_mean
        )

        return predictions
    
    def save(self):
        self.model.save("libs_net.keras")

    def load(self):
        root = os.getcwd()
        model_file = os.join(root, "Model/libs_net.keras")

        self.model = tf.keras.models.load_model(model_file)
        
    
    def evaluate(self):
        predictions = self.predict(
            self.X_test[..., 0] * self.X_scale
        )

        mae = mean_absolute_error(
            self.y_test,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                self.y_test,
                predictions
            )
        )

        r2 = r2_score(
            self.y_test,
            predictions
        )

        print(f"Test MAE:  {mae:.4f}")
        print(f"Test RMSE: {rmse:.4f}")
        print(f"Test R²:   {r2:.4f}")

        fig, ax = plt.subplots()

        ax.scatter(
            self.y_test,
            predictions,
            alpha=0.7
        )

        minimum = min(
            self.y_test.min(),
            predictions.min()
        )
        maximum = max(
            self.y_test.max(),
            predictions.max()
        )

        ax.plot(
            [minimum, maximum],
            [minimum, maximum],
            linestyle="--"
        )

        ax.set_xlabel("Actual concentration")
        ax.set_ylabel("Predicted concentration")
        ax.set_title(
            f"LIBS regression: $R^2={r2:.3f}$"
        )

        plt.tight_layout()
        plt.show()

        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "actual": self.y_test,
            "predicted": predictions
        }
    
if __name__ == "__main__":
    net = LIBS_Net()

    # Reads in the processed data
    net.read_data("DP")

    # Splits up the data into test and training
    net.split_data()

    # Initializes the model
    net.build_model()

    # Trains and evaluates the model
    history = net.train()
    results = net.evaluate()
