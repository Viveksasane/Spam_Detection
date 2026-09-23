import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy
from src.exception import CustomException
from src.logger import logging
import sys
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download("stopwords")
stopword=stopwords.words("english")
from sklearn.feature_extraction.text import CountVectorizer
import re
from dataclasses import dataclass
import os
ps=PorterStemmer()
cv=CountVectorizer()
from sklearn.model_selection import train_test_split
import pickle



@dataclass
class PreprocessConfig:
        preprocessed_data: str = os.path.join("artifacts", "preprocessing.csv")
        train_path : str = os.path.join("artifacts","train_data.csv")
        test_path : str = os.path.join("artifacts","test_data.csv")


class Preprocessing:
    def __init__(self):
        self.preprocessing_config=PreprocessConfig()


    def clean(self,data):
        try:
            preprocess=re.sub("[^A-Za-z0-9]"," ",str(data))
            preprocess=preprocess.lower().split()
            preprocess=[ps.stem(i) for i in preprocess if i not in stopword]
            return " ".join(preprocess)
        except Exception as e:
            raise e




    def cleaned(self,data):
        try:
            logging.info("Data preprocessing step in process")
            data["cleaned"]=data["cleaned"].apply(self.clean)
            os.makedirs(
                os.path.dirname(self.preprocessing_config.preprocessed_data),exist_ok=True
            )
            data.to_csv(
                self.preprocessing_config.preprocessed_data,index=False
            )
            logging.info("Data preprocessing step completed successfully")

            return data
        except Exception as e:
            raise e

    def splitting_dataset(self,data):
        try:
            logging.info("Training and Testing step starts")
            data=pd.read_csv(self.preprocessing_config.preprocessed_data)

            train_data,test_data=train_test_split(data,
                                        random_state=42,
                                        test_size=0.25)

            os.makedirs(
                os.path.dirname(self.preprocessing_config.train_path),exist_ok=True
            )
            train_data.to_csv(
                self.preprocessing_config.train_path,index=False
            )
            os.makedirs(
                os.path.dirname(self.preprocessing_config.test_path),exist_ok=True
            )
            test_data.to_csv(
                self.preprocessing_config.test_path,index=False
            )
            logging.info("Splitting data into training and testing completed successfully")
            return train_data,test_data
        except Exception as e:
            raise e
