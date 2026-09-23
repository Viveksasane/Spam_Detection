import pandas as pd
from src.components.data_ingestion import DataLoading
from src.components.preprocessing import Preprocessing
from src.components.model_training import ModelTraining


# Data Loading

loader=DataLoading()
data=loader.data_loader()

# Data Cleaning

preprocess=Preprocessing()
processed=preprocess.cleaned(data)
train_data,test_data=preprocess.splitting_dataset(processed)

# Data Modeling

models=ModelTraining()
model=models.model_evaluation()
