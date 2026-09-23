import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy
from src.exception import CustomException
from src.logger import logging
import sys
import re
from dataclasses import dataclass
import os



@dataclass
class DataLoaderConfig:
    data_path : str=os.path.join("artifacts","data.csv")

class DataLoading:
    
    def __init__(self):
        self.data_loader_config=DataLoaderConfig()

    def data_loader(self):
        try:
            logging.info("Data Loading in process")

            data=pd.read_csv(r"D:\Machine_Learning_Projects\Spam_detection\data\data_clean.csv")

            os.makedirs(
                os.path.dirname(self.data_loader_config.data_path),
                exist_ok=True
                )
            data.to_csv(
                self.data_loader_config.data_path,index=False
            )
            logging.info("Data Loading Successfully completed")
            return data
        except Exception as e:
            raise e

