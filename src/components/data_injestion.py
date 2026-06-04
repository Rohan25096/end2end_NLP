import os
import sys
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig

@dataclass
class DataInjestionConfig:
    train_data_path: str=os.path.join('artifacts/raw','train.csv')
    test_data_path: str=os.path.join('artifacts/raw','test.csv')
    validation_data_path: str=os.path.join('artifacts/raw','validation.csv')

class DataInjestion:
    def __init__(self):
        self.injestion_config = DataInjestionConfig()

    def initiate_data_injestion(self):
        logging.info('Entered data injestion method or component.')

        train_df = pd.read_csv(r'C:\Users\rohan\OneDrive\Desktop\jupytr_proj\emotion-detection\train.csv')
        test_df = pd.read_csv(r'C:\Users\rohan\OneDrive\Desktop\jupytr_proj\emotion-detection\test.csv')
        validation_df = pd.read_csv(r'C:\Users\rohan\OneDrive\Desktop\jupytr_proj\emotion-detection\validation.csv')

        logging.info('Reading the dataset')

        try:
            os.makedirs(os.path.dirname(self.injestion_config.train_data_path),exist_ok=True)

            train_df.to_csv(self.injestion_config.train_data_path,index=False,header=True)
            test_df.to_csv(self.injestion_config.test_data_path,index=False,header=True)
            validation_df.to_csv(self.injestion_config.validation_data_path,index=False,header=True)

            logging.info('Injestion of Data is completed')

            return(
                self.injestion_config.train_data_path,
                self.injestion_config.test_data_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__=="__main__":
    obj = DataInjestion()
    train_path,test_path = obj.initiate_data_injestion()

    data_transform = DataTransformation()
    padded_sequences, emotion_class_arr, metadata = data_transform.initiate_train_data_transform(train_path)

    model_trainer = ModelTrainer()
    Model = model_trainer.initiate_model_trainer(padded_sequences, emotion_class_arr)