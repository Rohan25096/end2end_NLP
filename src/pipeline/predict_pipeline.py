import os
import sys
import numpy as np

from src.logger import logging
from src.exception import CustomException
from src.utils import load_object, label_mapper
from src.components.data_transformation import DataTransformation
from dataclasses import dataclass


@dataclass
class PredictPipelineConfig:

    model_path: str = os.path.join("artifacts", "model", "model.pkl")
    metadata_path: str = os.path.join("artifacts", "preprocessor", "metadata.pkl")
    tokenizer_path: str = os.path.join("artifacts", "preprocessor", "tokenizer.pkl")

class PredictPipeline:

    def __init__(self):
        try:

            self.pipeline_config = PredictPipelineConfig()
            self.transformer_config = DataTransformation()

            self.model = load_object(self.pipeline_config.model_path)
        
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_predictor(self, Model, sentence):
    
        try:
            padded_sequence = self.transformer_config.initiate_data_transform(sentence,
                                                                                self.pipeline_config.tokenizer_path,
                                                                                self.pipeline_config.metadata_path)
            
            pred = self.model.predict(padded_sequence)
            pred_class = int(np.argmax(pred, axis=1)[0])
            pred_emotion_class = label_mapper(pred_class)

            return pred, pred_emotion_class
            
        except Exception as e:
            raise CustomException(e, sys)


