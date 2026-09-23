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

from src.components.preprocessing import *
from src.components.model_training import *


@dataclass
class PredictingConfig :
    model_path : str = os.path.join("artifacts","model.pkl")
    preprocess_path : str = os.path.join("artifacts","preprocessing.csv")
    vector_path : str = os.path.join("artifacts","vectorizer.pkl")

class Predicting :
    
    def __init__(self):

        self.prediction=PredictingConfig()
        self.process=Preprocessing()

    def Pred(self,data):

        try :
            logging.info("Prediction Stage is in Process ")

            with open(self.prediction.model_path,"rb") as f:
                model=pickle.load(f)
            logging.info("Model loaded successfully")

            # Data Cleaning Step

            new_cleaning=self.process.clean(data)

            # Data Converting into numeric
            with open(self.prediction.vector_path,"rb") as f:
                cv=pickle.load(f)

            new_data=cv.transform([new_cleaning])

            # Now Predicting the data

            pre=model.predict(new_data)

            logging.info("Prediction Stage is completed successfully")
            
            return pre

        except Exception as e:
            raise e