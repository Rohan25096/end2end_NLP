import os
import sys
import pickle

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object, save_object

import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    tokenizer_path: str = os.path.join('artifacts/preprocessor','tokenizer.pkl')
    metadata_path: str = os.path.join('artifacts/preprocessor',"metadata.pkl")
    

class DataTransformation:
    def __init__(self):
        self.preprocessing_config = DataTransformationConfig()

    def initiate_train_data_transform(self,train_path):
        try:
            train_df = pd.read_csv(train_path)
            X_train = train_df['text']
            y_train = train_df['label']

            logging.info("Tokenization in progress")
            T = Tokenizer()
            T.fit_on_texts(X_train)

            vocab_size = len(T.word_index) + 1
            sentences_list = X_train.tolist()
            max_length = max(len(word.split()) for word in sentences_list)
            num_classes = int(y_train.value_counts().count())
            
            logging.info("Converting text to sequences")
            sequences = T.texts_to_sequences(sentences_list)
            padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')

            os.makedirs(os.path.dirname(self.preprocessing_config.tokenizer_path), exist_ok=True)

            logging.info("Saving tokenizer")
            save_object(self.preprocessing_config.tokenizer_path, T)

            metadata = {
                "vocab_size": vocab_size,
                "max_length": max_length,
                "num_classes": num_classes
            }

            logging.info(f"Saving metadata as {vocab_size} {max_length} {num_classes}")
            save_object(self.preprocessing_config.metadata_path, metadata)
            
            return(
                padded_sequences,
                y_train.values,
                metadata
            )
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transform(self, sentence, tokenizer_path, metadata_path):
        try:
            T = load_object(file_path=tokenizer_path)
            metadata = load_object(file_path=metadata_path)

            max_length = metadata['max_length']

            sequence = T.texts_to_sequences(sentence)
            padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='pre')

            return padded_sequence
        except Exception as e:
            raise CustomException(e, sys)