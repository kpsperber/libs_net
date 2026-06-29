import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

class LIBS_Net:
    def __init__(self):
        self.root = os.getcwd()

        # Final combined model
        self.model = None
        self.model_file = "Model/libs_net.keras"

        # Global CNN
        self.global_model = None
        self.global_file = "Model/global.keras"

        self.global_X = None
        self.global_y = None

        self.global_X_scale = None
        self.global_y_mean = None
        self.global_y_std = None

        self.global_X_train = None
        self.global_X_val = None
        self.global_X_test = None

        self.global_y_train = None
        self.global_y_val = None
        self.global_y_test = None

        self.global_y_train_scaled = None
        self.global_y_val_scaled = None
        self.global_y_test_scaled = None

        # Local peak CNN
        self.local_model = None
        self.local_file = "Model/local.keras"

        self.local_X = None
        self.local_y = None

        self.local_X_scale = None
        self.local_y_mean = None
        self.local_y_std = None

        self.local_X_train = None
        self.local_X_val = None
        self.local_X_test = None

        self.local_y_train = None
        self.local_y_val = None
        self.local_y_test = None

        self.local_y_train_scaled = None
        self.local_y_val_scaled = None
        self.local_y_test_scaled = None

        # Physics / Voigt model
        self.physics_model = None
        self.physics_file = "Model/physics.keras"

        self.physics_X = None
        self.physics_y = None

        self.physics_X_mean = None
        self.physics_X_std = None

        self.physics_y_mean = None
        self.physics_y_std = None

        self.physics_X_train = None
        self.physics_X_val = None
        self.physics_X_test = None

        self.physics_y_train = None
        self.physics_y_val = None
        self.physics_y_test = None

        self.physics_y_train_scaled = None
        self.physics_y_val_scaled = None
        self.physics_y_test_scaled = None

        # Super-model data
        self.net_X_train = None
        self.net_X_val = None
        self.net_X_test = None

        self.net_y_train = None
        self.net_y_val = None
        self.net_y_test = None

        self.net_X_mean = None
        self.net_X_std = None

        self.metrics_folder = os.path.join(
            os.getcwd(),
            "Training Metrics"
        )

    # -------------------------
    # Reading data
    # -------------------------

    def read_global_data(self, root_folder):
        training_folder = os.path.join(self.root, root_folder)
        training_folder = os.path.join(training_folder, "Training Data")
        data_file = os.path.join(training_folder, "cnn_global_data.csv")
        shape_file = os.path.join(training_folder, "shape.npy")

        self.global_shape = tuple(np.load(shape_file).astype(int))
        labeled_data = pd.read_csv(data_file)

        labels = []
        for label in labeled_data.keys():
            labels.append(label.split("p")[0])

        labels = np.float32(labels)

        data = labeled_data.to_numpy().T.reshape((len(labels), *self.global_shape))

        self.global_X = []
        self.global_y = []

        for i in range(0, len(labels)):
            for j in range(0, len(data[i])):
                self.global_X.append(data[i, j])
                self.global_y.append(labels[i])

        self.global_X = np.array(self.global_X)
        self.global_y = np.array(self.global_y)

        print("=" * 50)
        print("Global CNN Data")
        print("=" * 50)
        print(f"X shape: {self.global_X.shape}")
        print(f"y shape: {self.global_y.shape}") 


    def read_local_data(self, root_folder):
        training_folder = os.path.join(self.root, root_folder)
        training_folder = os.path.join(training_folder, "Training Data")
        data_file = os.path.join(training_folder, "cnn_local_data.csv")
        shape_file = os.path.join(training_folder, "shape_local.npy")

        self.local_shape = tuple(np.load(shape_file).astype(int))
        labeled_data = pd.read_csv(data_file)

        labels = []
        for label in labeled_data.keys():
            labels.append(label.split("p")[0])

        labels = np.float32(labels)

        data = labeled_data.to_numpy().T.reshape((len(labels), *self.local_shape))

        self.local_X = []
        self.local_y = []

        for i in range(0, len(labels)):
            for j in range(0, len(data[i])):
                self.local_X.append(data[i, j])
                self.local_y.append(labels[i])

        self.local_X = np.array(self.local_X)
        self.local_y = np.array(self.local_y)

        print("=" * 50)
        print("Local CNN Data")
        print("=" * 50)
        print(f"X shape: {self.local_X.shape}")
        print(f"y shape: {self.local_y.shape}")

    def read_physics_data(self, root_folder):
        pass

    # -------------------------
    # Splitting and scaling
    # -------------------------

    def split_global_data(self, test_size = 0.2, validation_size = 0.2, random_state = 42):
        X_train_val, self.global_X_test, y_train_val, self.global_y_test = train_test_split(
            self.global_X,
            self.global_y,
            test_size = test_size,
            random_state = random_state,
            shuffle = True,
            stratify = self.global_y
        )

        relative_validation_size = validation_size / (1.0 - test_size)

        self.global_X_train, self.global_X_val, self.global_y_train, self.global_y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size = relative_validation_size,
            random_state = random_state,
            shuffle = True,
            stratify = y_train_val
        )

        self.global_X_scale = np.max(np.abs(self.global_X_train))

        self.global_X_train = self.global_X_train / self.global_X_scale
        self.global_X_val = self.global_X_val / self.global_X_scale
        self.global_X_test = self.global_X_test / self.global_X_scale

        self.global_y_mean = np.mean(self.global_y_train)
        self.global_y_std = np.std(self.global_y_train)

        self.global_y_train_scaled = (self.global_y_train - self.global_y_mean) / self.global_y_std
        self.global_y_val_scaled = (self.global_y_val - self.global_y_mean) / self.global_y_std
        self.global_y_test_scaled = (self.global_y_test - self.global_y_mean) / self.global_y_std

        self.global_X_train = self.global_X_train[..., np.newaxis]
        self.global_X_val = self.global_X_val[..., np.newaxis]
        self.global_X_test = self.global_X_test[..., np.newaxis]

        print("=" * 50)
        print("Splitting Global CNN Data")
        print(f"Training Shape: {self.global_X_train.shape}")
        print(f"Validation Shape: {self.global_X_val.shape}")
        print(f"Test Shape: {self.global_X_test.shape}")

    def split_local_data(self, test_size = 0.2, validation_size = 0.2, random_state = 42):
        X_train_val, self.local_X_test, y_train_val, self.local_y_test = train_test_split(
            self.local_X,
            self.local_y,
            test_size = test_size,
            random_state = random_state,
            shuffle = True,
            stratify = self.local_y
        )

        relative_validation_size = validation_size / (1.0 - test_size)

        self.local_X_train, self.local_X_val, self.local_y_train, self.local_y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size = relative_validation_size,
            random_state = random_state,
            shuffle = True,
            stratify = y_train_val
        )

        self.local_X_scale = np.max(np.abs(self.local_X_train))

        self.local_X_train = self.local_X_train / self.local_X_scale
        self.local_X_val = self.local_X_val / self.local_X_scale
        self.local_X_test = self.local_X_test / self.local_X_scale

        self.local_y_mean = np.mean(self.local_y_train)
        self.local_y_std = np.std(self.local_y_train)

        self.local_y_train_scaled = (self.local_y_train - self.local_y_mean) / self.local_y_std
        self.local_y_val_scaled = (self.local_y_val - self.local_y_mean) / self.local_y_std
        self.local_y_test_scaled = (self.local_y_test - self.local_y_mean) / self.local_y_std

        self.local_X_train = self.local_X_train[..., np.newaxis]
        self.local_X_val = self.local_X_val[..., np.newaxis]
        self.local_X_test = self.local_X_test[..., np.newaxis]

        print("=" * 50)
        print("Splitting local CNN Data")
        print(f"Training Shape: {self.local_X_train.shape}")
        print(f"Validation Shape: {self.local_X_val.shape}")
        print(f"Test Shape: {self.local_X_test.shape}")

    def split_physics_data(self, test_size = 0.2, validation_size = 0.2, random_state = 42):
        pass

    def split_data(self, test_size = 0.2, validation_size = 0.2, random_state = 42):

        self.split_cnn_data(test_size = test_size, validation_size = validation_size, random_state = random_state)

        self.split_local_data(test_size = test_size, validation_size = validation_size, random_state = random_state)

        self.split_physics_data(test_size = test_size, validation_size = validation_size, random_state=random_state        )

    # -------------------------
    # Building models
    # -------------------------

    def build_global_model(self):
        number_of_wavelengths = self.global_shape[1]

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape = (number_of_wavelengths, 1)),
            tf.keras.layers.Conv1D(
                filters = 32,
                kernel_size = 7,
                padding = "same",
                activation = "relu"
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(pool_size = 2),
            tf.keras.layers.Conv1D(
                filters = 64,
                kernel_size = 7,
                padding = "same",
                activation = "relu"
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(pool_size = 2),
            tf.keras.layers.Conv1D(
                filters = 128,
                kernel_size = 5,
                padding = "same",
                activation = "relu"
            ),
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
        ], name = "global_cnn")

        model.optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3)
        model.compile(
            optimizer = model.optimizer,
            loss = tf.keras.losses.Huber(),
            metrics = [
                tf.keras.metrics.RootMeanSquaredError(name = "rmse"),
                tf.keras.metrics.MeanAbsoluteError(name = "mae")
            ]
        )

        model.summary()
        self.global_model = model

    def build_local_model(self):
        number_of_wavelengths = self.local_shape[1]

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape = (number_of_wavelengths, 1)),
            tf.keras.layers.Conv1D(
                filters = 32,
                kernel_size = 7,
                padding = "same",
                activation = "relu"
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(pool_size = 2),
            tf.keras.layers.Conv1D(
                filters = 64,
                kernel_size = 7,
                padding = "same",
                activation = "relu"
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(pool_size = 2),
            tf.keras.layers.Conv1D(
                filters = 128,
                kernel_size = 5,
                padding = "same",
                activation = "relu"
            ),
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
        ], name = "local_cnn")

        model.optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3)
        model.compile(
            optimizer = model.optimizer,
            loss = tf.keras.losses.Huber(),
            metrics = [
                tf.keras.metrics.RootMeanSquaredError(name = "rmse"),
                tf.keras.metrics.MeanAbsoluteError(name = "mae")
            ]
        )

        model.summary()
        self.local_model = model

    def build_physics_model(self):
        number_of_features = self.physics_X_train.shape[1]

        model = tf.keras.Sequential([
            tf.keras.layers.Input(
                shape = (number_of_features,)
            ),
            tf.keras.layers.Dense(
                64,
                activation = "relu",
                kernel_regularizer = tf.keras.regularizers.l2(1e-4)
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(
                32,
                activation = "relu",
                kernel_regularizer = tf.keras.regularizers.l2(1e-4)
            ),
            tf.keras.layers.Dropout(0.1),

            tf.keras.layers.Dense(
                16,
                activation = "relu",
            ),

            tf.keras.layers.Dense(1)
        ], name = "physics_model")

        optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3)

        model.compile(
            optimizer = optimizer,
            loss = tf.keras.losses.Huber(),
            metrics = [
                tf.keras.metrics.RootMeanSquaredError(
                    name = "rmse"
                ),
                tf.keras.metrics.MeanAbsoluteError(
                    name = "mae"
                )
            ]
        )

        model.summary()
        self.physics_model = model

    def build_model(self):
        pass

    # -------------------------
    # Training
    # -------------------------

    def train_global(self, epochs = 300, batch_size = 32):
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor = "val_loss",
                patience = 25,
                restore_best_weights = True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor = "val_loss",
                factor = 0.5,
                patience = 8,
                min_lr = 1e-6,
                verbose = 1
            )
        ]

        history = self.global_model.fit(
            self.global_X_train,
            self.global_y_train_scaled,
            validation_data = (
                self.global_X_val,
                self.global_y_val_scaled
            ),
            epochs = epochs,
            batch_size = batch_size,
            callbacks = callbacks,
            verbose = 1
        )

        self.plot_training_history(history, "Global CNN")
        return history

    def train_local(self, epochs = 300, batch_size = 32):
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor = "val_loss",
                patience = 25,
                restore_best_weights = True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor = "val_loss",
                factor = 0.5,
                patience = 8,
                min_lr = 1e-6,
                verbose = 1
            )
        ]

        history = self.local_model.fit(
            self.local_X_train,
            self.local_y_train_scaled,
            validation_data = (
                self.local_X_val,
                self.local_y_val_scaled
            ),
            epochs = epochs,
            batch_size = batch_size,
            callbacks = callbacks,
            verbose = 1
        )

        self.plot_training_history(history, "Local CNN")
        return history

    def train_physics(self, epochs = 300, batch_size = 32):
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor = "val_loss",
                patience = 25,
                restore_best_weights = True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor = "val_loss",
                factor = 0.5,
                patience = 8,
                min_lr = 1e-6,
                verbose = 1
            )
        ]

        history = self.physics_model.fit(
            self.physics_X_train,
            self.physics_y_train_scaled,
            validation_data = (
                self.physics_X_val,
                self.physics_y_val_scaled
            ),
            epochs = epochs,
            batch_size = batch_size,
            callbacks = callbacks,
            verbose = 1
        )

        self.plot_training_history(history, "Local CNN")
        return history

    def train(self, epochs = 300, batch_size = 32):
        pass

    # -------------------------
    # Prediction
    # -------------------------

    def predict_global(self, X):
        X = np.asarray(X, dtype = np.float32)
        X = X / self.global_X_scale
        X = X[..., np.newaxis]

        prediction_scaled = self.global_model.predict(X, verbose = 0).reshape(-1)
        prediction = (prediction_scaled * self.global_y_std + self.global_y_mean)

        return prediction

    def predict_local(self, X):
        X = np.asarray(X, dtype = np.float32)
        X = X / self.local_X_scale
        X = X[..., np.newaxis]

        prediction_scaled = self.local_model.predict(X, verbose = 0).reshape(-1)
        prediction = (prediction_scaled * self.local_y_std + self.local_y_mean)

        return prediction

    def predict_physics(self, X):
        X = np.asarray(X, dtype = np.float32)
        X = X / self.physics_X_scale
        X = X[..., np.newaxis]

        prediction_scaled = self.physics_model.predict(X, verbose = 0).reshape(-1)
        prediction = (prediction_scaled * self.physics_y_std + self.physics_y_mean)

        return prediction

    def predict(self, cnn_X, local_X, physics_X):
        pass

    # -------------------------
    # Evaluation
    # -------------------------

    def evaluate_global(self, root_folder):
        output_folder = os.path.join(self.root, root_folder)
        output_folder = os.path.join(output_folder, "Global CNN")
        os.makedirs(output_folder, exist_ok = True)

        predictions = self.predict_global(self.global_X_test[..., 0] * self.global_X_scale)
        mae = mean_absolute_error(self.global_y_test, predictions)
        rmse = np.sqrt(mean_squared_error(self.global_y_test, predictions))
        r2 = r2_score(self.global_y_test, predictions)

        print(f"Test MAE:  {mae:.4f}")
        print(f"Test RMSE: {rmse:.4f}")
        print(f"Test R²:   {r2:.4f}")

        fig, ax = plt.subplots()

        ax.scatter(
            self.global_y_test,
            predictions,
            alpha=0.7
        )

        minimum = min(
            self.global_y_test.min(),
            predictions.min()
        )
        maximum = max(
            self.global_y_test.max(),
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

    def evaluate_local(self, root_folder):
        output_folder = os.path.join(self.root, root_folder)
        output_folder = os.path.join(output_folder, "Local CNN")
        os.makedirs(output_folder, exist_ok = True)

        predictions = self.predict_local(self.local_X_test[..., 0] * self.local_X_scale)
        mae = mean_absolute_error(self.local_y_test, predictions)
        rmse = np.sqrt(mean_squared_error(self.local_y_test, predictions))
        r2 = r2_score(self.local_y_test, predictions)

        print(f"Test MAE:  {mae:.4f}")
        print(f"Test RMSE: {rmse:.4f}")
        print(f"Test R²:   {r2:.4f}")

        fig, ax = plt.subplots()

        ax.scatter(
            self.local_y_test,
            predictions,
            alpha=0.7
        )

        minimum = min(
            self.local_y_test.min(),
            predictions.min()
        )
        maximum = max(
            self.local_y_test.max(),
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

    def evaluate_physics(self, root_folder):
        output_folder = os.path.join(self.root, root_folder)
        output_folder = os.path.join(output_folder, "Physics")
        os.makedirs(output_folder, exist_ok = True)

        predictions = self.predict_physics(self.physics_X_test[..., 0] * self.physics_X_scale)
        mae = mean_absolute_error(self.physics_y_test, predictions)
        rmse = np.sqrt(mean_squared_error(self.physics_y_test, predictions))
        r2 = r2_score(self.physics_y_test, predictions)

        print(f"Test MAE:  {mae:.4f}")
        print(f"Test RMSE: {rmse:.4f}")
        print(f"Test R²:   {r2:.4f}")

        fig, ax = plt.subplots()

        ax.scatter(
            self.physics_y_test,
            predictions,
            alpha=0.7
        )

        minimum = min(
            self.physics_y_test.min(),
            predictions.min()
        )
        maximum = max(
            self.physics_y_test.max(),
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

    def evaluate(self):
        pass

    # -------------------------
    # Saving
    # -------------------------

    def save_global(self):
        output_file = os.path.join(self.root, self.global_file)
        self.global_model.save(output_file)

    def save_local(self):
        output_file = os.path.join(self.root, self.local_file)
        self.local_model.save(output_file)

    def save_physics(self):
        output_file = os.path.join(self.root, self.physics_file)
        self.physics_model.save(output_file)

    def save(self):
        output_file = os.path.join(self.root, self.model_file)
        self.model.save(output_file)

    # -------------------------
    # Loading
    # -------------------------

    def load_global(self):
        input_file = os.path.join(self.root, self.global_file)
        
        self.global_model = tf.keras.models.load_model(input_file)

    def load_local(self):
        input_file = os.path.join(self.root, self.local_file)

        self.local_model = tf.keras.models.load_model(input_file)

    def load_physics(self):
        input_file = os.path.join(self.root, self.physics_file)

        self.physics_model = tf.keras.models.load_model(input_file)

    def load(self):
        input_file = os.path.join(self.root, self.model_file)

        self.model = tf.keras.models.load_model(input_file)

    # -------------------------
    # Plotting
    # -------------------------

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