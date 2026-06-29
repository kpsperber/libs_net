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

def heteroscedastic_gaussian_nll(y_true, y_pred):
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true = tf.reshape(y_true, (-1,))

    mean = y_pred[:, 0]
    log_variance = y_pred[:, 1]

    # Prevent numerical overflow or an unrealistically tiny variance.
    log_variance = tf.clip_by_value(
        log_variance,
        clip_value_min=-10.0,
        clip_value_max=10.0
    )

    squared_error = tf.square(y_true - mean)

    nll = 0.5 * (
        log_variance
        + squared_error * tf.exp(-log_variance)
    )

    return tf.reduce_mean(nll)

def concentration_rmse(y_true, y_pred):
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true = tf.reshape(y_true, (-1,))
    mean = y_pred[:, 0]

    return tf.sqrt(tf.reduce_mean(tf.square(y_true - mean)))


def concentration_mae(y_true, y_pred):
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true = tf.reshape(y_true, (-1,))
    mean = y_pred[:, 0]

    return tf.reduce_mean(tf.abs(y_true - mean))

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

        self.cnn_model = None
        self.cnn_file = "Model/cnn.keras"
        self.physics_model = None
        self.physics_file = "Model/physics.keras"
        self.net = None

        self.metrics_folder = os.getcwd() + "/Training Metrics"

    def read_cnn_data(self, root_folder):
        self.root_folder = os.path.join(root_folder, "Training Data")

        self.shape = tuple(np.load(os.path.join(self.root_folder, "shape.npy")).astype(int))

        labeled_data = pd.read_csv(os.path.join(self.root_folder, "cnn_global_data.csv"))

        labels = []
        for label in labeled_data.keys():
            labels.append(label.split("p")[0])

        labels = np.float32(labels)

        data = labeled_data.to_numpy().T.reshape(
            len(labels), *self.shape
        )

        self.cnn_X = []
        self.cnn_y = []

        for i in range(0, len(labels)):
            for j in range(0, len(data[i])):
                self.cnn_X.append(data[i, j])
                self.cnn_y.append(labels[i])

        self.cnn_X = np.array(self.cnn_X)
        self.cnn_y = np.array(self.cnn_y)
        
        print("X shape:", self.cnn_X.shape)
        print("y shape:", self.cnn_y.shape)
        print("Labels:", np.unique(self.cnn_y, return_counts=True))

    def read_physics_data(self, root_folder):
        pass

    def build_cnn_model(self):
        number_of_wavelengths = self.shape[1]

        model = tf.keras.Sequential([
            tf.keras.layers.Input(
                shape=(number_of_wavelengths, 1),
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
        ],
        name = "cnn_model")

        model.optimizer = tf.keras.optimizers.Adam(
            learning_rate=1e-3
        )

        model.compile(
            optimizer=model.optimizer,
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

        model.summary()
        
        self.cnn_model = model

    def build_physics_model(self):
        pass

    def build_model(self):
        pass

    def train_cnn(self, epochs=300, batch_size=32):
        if self.cnn_model is None:
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
        history = self.cnn_model.fit(
            self.cnn_X_train,
            self.cnn_y_train_scaled,
            validation_data=(
                self.cnn_X_val,
                self.cnn_y_val_scaled
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        self.plot_training_history(history, "CNN")

        return history
    
    def train(self, epochs = 300, batch_size = 32):
        return

    def plot_training_history(self, history, model):
        output_folder = os.path.join(self.metrics_folder, model)
        os.makedirs(output_folder, exist_ok=True)

        history_data = history.history
        epochs = np.arange(1, len(history.epoch) + 1)

        # Get metric names while excluding validation duplicates.
        metric_names = [
            metric
            for metric in history_data.keys()
            if not metric.startswith("val_")
        ]

        for metric in metric_names:
            validation_metric = f"val_{metric}"

            fig, ax = plt.subplots(figsize=(8, 5))

            ax.plot(
                epochs,
                history_data[metric],
                label=f"Training {metric}"
            )

            if validation_metric in history_data:
                ax.plot(
                    epochs,
                    history_data[validation_metric],
                    label=f"Validation {metric}"
                )

            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_title(
                f"Training and Validation "
                f"{metric.replace('_', ' ').title()}"
            )
            ax.grid(True)
            ax.legend()

            fig.tight_layout()

            file_path = os.path.join(
                output_folder,
                f"{metric}.png"
            )

            fig.savefig(
                file_path,
                dpi=300,
                bbox_inches="tight"
            )

            plt.close(fig)        
    
    def train_physics(self, epochs=300, batch_size=32):
        pass
    
    def split_cnn_data(self, test_size = 0.2, validation_size = 0.2, random_state = 42):

        # Using scikit learns auto splitter because I'm lazy
        X_train_val, self.cnn_X_test, y_train_val, self.cnn_y_test = (
            train_test_split(
                self.cnn_X,
                self.cnn_y,
                test_size=test_size,
                random_state=random_state,
                shuffle=True,
                stratify=self.cnn_y
            )
        )

        relative_validation_size = validation_size / (1.0 - test_size)

        self.cnn_X_train, self.cnn_X_val, self.cnn_y_train, self.cnn_y_val = (
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
        self.cnn_X_scale = np.max(np.abs(self.cnn_X_train))

        if self.cnn_X_scale == 0:
            raise ValueError("Training spectra contain only zeros.")

        # Rescale training and testing data
        self.cnn_X_train = self.cnn_X_train / self.cnn_X_scale
        self.cnn_X_val = self.cnn_X_val / self.cnn_X_scale
        self.cnn_X_test = self.cnn_X_test / self.cnn_X_scale

        # Rescale y to further help the loss function not explode
        self.cnn_y_mean = np.mean(self.cnn_y_train)
        self.cnn_y_std = np.std(self.cnn_y_train)

        if self.cnn_y_std == 0:
            raise ValueError(
                "Training targets all have the same concentration."
            )

        self.cnn_y_train_scaled = (
            self.cnn_y_train - self.cnn_y_mean
        ) / self.cnn_y_std

        self.cnn_y_val_scaled = (
            self.cnn_y_val - self.cnn_y_mean
        ) / self.cnn_y_std

        self.y_test_scaled = (
            self.cnn_y_test - self.cnn_y_mean
        ) / self.cnn_y_std

        # Resizing for tensor flow
        self.cnn_X_train = self.cnn_X_train[..., np.newaxis]
        self.cnn_X_val = self.cnn_X_val[..., np.newaxis]
        self.cnn_X_test = self.cnn_X_test[..., np.newaxis]

        print("Training shape:", self.cnn_X_train.shape)
        print("Validation shape:", self.cnn_X_val.shape)
        print("Test shape:", self.cnn_X_test.shape)

    def split_physics_data(self, test_size = 0.2, validation = 0.2, random_state = 42):
        pass

    def split_data(self, test_size = 0.2, validation = 0.2, random_state = 42):
        pass

    def predict_cnn(self, X):
        if self.cnn_model is None:
            raise RuntimeError("The model has not been built.")

        if self.cnn_X_scale is None:
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

        X = X / self.cnn_X_scale
        X = X[..., np.newaxis]

        predictions_scaled = self.cnn_model.predict(
            X,
            verbose=0
        ).reshape(-1)

        predictions = (
            predictions_scaled * self.cnn_y_std
            + self.cnn_y_mean
        )

        return predictions
    
    def save_cnn(self):
        root = os.getcwd()
        model_file = os.path.join(root, self.cnn_file)
        self.cnn_model.save(model_file)

    def save_physics(self):
        root = os.getcwd()
        model_file = os.path.join(root, self.physics_file)
        self.cnn_model.save(model_file)

    def save(self):
        self.model.save("libs_net.keras")

    def load_cnn(self):
        root = os.getcwd()
        model_file = os.path.join(root, self.cnn_file)

        self.cnn_model = tf.keras.models.load_model(model_file)

    def load_physics(self):
        root = os.getcwd()
        model_file = os.path.join(root, self.physics_file)

        self.physics_model = tf.keras.models.load_model(model_file)
          
    def evaluate_cnn(self):
        output_folder = os.path.join(self.metrics_folder, "CNN")
        os.makedirs(output_folder, exist_ok = True)
        
        predictions = self.predict_cnn(
            self.cnn_X_test[..., 0] * self.cnn_X_scale
        )

        mae = mean_absolute_error(
            self.cnn_y_test,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                self.cnn_y_test,
                predictions
            )
        )

        r2 = r2_score(
            self.cnn_y_test,
            predictions
        )

        print(f"Test MAE:  {mae:.4f}")
        print(f"Test RMSE: {rmse:.4f}")
        print(f"Test R²:   {r2:.4f}")

        fig, ax = plt.subplots()

        ax.scatter(
            self.cnn_y_test,
            predictions,
            alpha=0.7
        )

        minimum = min(
            self.cnn_y_test.min(),
            predictions.min()
        )
        maximum = max(
            self.cnn_y_test.max(),
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

        plt.savefig(output_folder + "/accuracy.png")

        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "actual": self.cnn_y_test,
            "predicted": predictions
        }


