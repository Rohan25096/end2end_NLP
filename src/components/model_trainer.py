import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, load_object

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

from dataclasses import dataclass, field


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model", "model.pkl")
    trained_history_plot_file_path: str = os.path.join("artifacts", "plots", "training_history.jpg")
    metadata_path = str = os.path.join("artifacts", "preprocessor", "metadata.pkl")

    output_dim: int = 15
    lstm_units: int = 128
    activation = 'softmax'
    loss = 'sparse_categorical_crossentropy'
    optimizer = 'adam'
    metrics: list = field(default_factory=lambda: ["accuracy"])
    epochs: int = 50

    vocab_size: int = field(init=False)
    max_length: int = field(init=False)
    num_classes: int = field(init=False)

    def __post_init__(self):
        metadata = load_object(self.metadata_path)
        self.vocab_size = metadata["vocab_size"]
        self.max_length = metadata["max_length"]
        self.num_classes = metadata["num_classes"]


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def build_model(self):
        try:

            model = Sequential([
                Embedding(
                    input_dim=self.model_trainer_config.vocab_size,
                    output_dim=self.model_trainer_config.output_dim,
                    input_length=self.model_trainer_config.max_length
                ),
                LSTM(self.model_trainer_config.lstm_units),
                Dense(
                    units=self.model_trainer_config.num_classes,
                    activation=self.model_trainer_config.activation)
            ])
            logging.info("LSTM architecture created")
            return model
        
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self,padded_sequences,emotion_class_arr):
        try:
            logging.info('Splitting train and test data')
            X_train, X_test, y_train, y_test = train_test_split(padded_sequences, emotion_class_arr, test_size=0.2, random_state=42)

            logging.info("Building Model")
            Model = self.build_model()

            logging.info("Compiling Model")
            Model.compile(
                loss = self.model_trainer_config.loss,
                optimizer = self.model_trainer_config.optimizer,
                metrics = self.model_trainer_config.metrics
            )

            early_stopping = EarlyStopping(
                monitor="val_loss",
                patience=10,
                verbose=True
            )

            logging.info("Training model")
            history = Model.fit(
                X_train,
                y_train,
                validation_data = (X_test, y_test),
                epochs=self.model_trainer_config.epochs,
                callbacks=[early_stopping],
                verbose=1
            )
            
            self.evaluate_model(Model, X_test, y_test)
            self.plot_training_charts(history)

            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=Model)

            return (
                Model
            )
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def evaluate_model(self, Model, X_test, y_test):
            try:
                logging.info("Evaluating scores")
                test_loss, test_accuracy = Model.evaluate(X_test, y_test)
                logging.info(f"Test accuracy {test_accuracy} and Test loss {test_loss}")

            except Exception as e:
                raise CustomException(e, sys)

    def plot_training_charts(self, history):
        try:
            plot_path = self.model_trainer_config.trained_history_plot_file_path
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_history_plot_file_path), exist_ok=True)


            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.plot(history.history["loss"], label="train_loss")
            plt.plot(history.history["val_loss"], label="val_loss")
            plt.title("Training and Validation Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(history.history["accuracy"], label="train_accuracy")
            plt.plot(history.history["val_accuracy"], label="val_accuracy")
            plt.title("Training and Validation Accuracy")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.legend()

            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            logging.info("Training history saved")

        except Exception as e:
            raise CustomException(e, sys)