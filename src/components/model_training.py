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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


@dataclass
class ModelTrainingConfig:
    train_path : str = os.path.join("artifacts","train_data.csv")
    test_path : str = os.path.join("artifacts","test_data.csv")
    model_path : str = os.path.join("artifacts","model.pkl")
    vector_path : str = os.path.join("artifacts","vectorizer.pkl")

class ModelTraining :

    def __init__(self):
        self.model_trainer_config=ModelTrainingConfig()


    def get_data_initialized(self):
        try:
            logging.info("Training and testing Initialization step")
            train_data=pd.read_csv(self.model_trainer_config.train_path)
            test_data=pd.read_csv(self.model_trainer_config.test_path)
            X_train=train_data["cleaned"]
            y_train=train_data["label"]
            X_test=test_data["cleaned"]
            y_test=test_data["label"]
            x_train=cv.fit_transform(X_train)
            x_test=cv.transform(X_test)

            os.makedirs(
                os.path.dirname(self.model_trainer_config.vector_path),
                exist_ok=True
                )
            with open(self.model_trainer_config.vector_path,"wb") as f:
                pickle.dump(cv,f)

            logging.info("Initializing of training and testing data completed successfully")
            return x_train,y_train,x_test,y_test
        except Exception as e:
            raise e

    def model_evaluation(self):
        try:
            logging.info("Model Evaluation Step starts")
            x_train,y_train,x_test,y_test=self.get_data_initialized()
            log=LogisticRegression()
            log.fit(x_train,y_train)
            ypre=log.predict(x_test)
            accuracy=accuracy_score(y_test,ypre)*100
            logging.info(f"Accuracy Score of {log} is : {accuracy}")
            os.makedirs(
                os.path.dirname(self.model_trainer_config.model_path),exist_ok=True
            )
            with open(self.model_trainer_config.model_path,"wb") as file:
                pickle.dump(log,file)
            logging.info("Model Evaluation Step Completed successfully")
        except Exception as e:
            logging.info("Error occur during model execution")
            raise e
