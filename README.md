# Spam Detection Project 🚀

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/) 
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Project Overview
This project is a **Spam Detection system** that classifies messages (emails or text) as **spam** or **non-spam**. It leverages **Machine Learning** and **Natural Language Processing (NLP)** techniques to provide accurate and fast predictions.

---

## Key Features
- **Text Preprocessing**: cleaning, tokenization, stopword removal
- **Feature Extraction**:  CountVectorizer
- **Machine Learning Models**:
  - Logistic Regression
  - Random Forest
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score
- **Web Interface**: Streamlit for real-time prediction

---

## Project Structure
```
Spam_Detection/
│──artifacts
│
├── Data/                  # Dataset and notebooks
│   └── spam.csv
│   └── EDA.ipynb
│
│── Logs
│
│── myenv
│
│── research
│
├── src/                   # Source code
│   ├── components/        # Modular functions
│   │   └── __init__.py
│   │   └── data_ingestion.py
│   │   └── model_training.py
│   │   └── preprocessing.py
│   ├── Pipeline/          # ML pipeline scripts
│   │   └── __init__.py
│   │   └── predictionpipeline.py 
│   ├── logger.py          # Logging utility
│   └── exception.py       # Custom exceptions
│
├── app.py                 # Main web application
└── main.py                # Pipeline
├── requirements.txt       # Python dependencies
├── setup.py               # Package setup
└── README.md              # Project documentation
```
