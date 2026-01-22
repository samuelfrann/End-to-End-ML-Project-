import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from dataclasses import dataclass
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, preprocessor_path):
        try:
            logging.info('Splitting training and testing input data')
            X = train_array[:, :-1]
            y = train_array[:, -1].astype(int)

            x_train, x_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            models = {
                'xgboost': XGBClassifier(),
                'tree': DecisionTreeClassifier()
            }

            params = {
                'xgboost':{
                'n_estimators': [32,64, 100, 128],
                'random_state': [42]},
                
                'tree':{
                    "criterion": ["gini", "entropy", "log_loss"],
                    "max_depth": [None, 5, 10, 20, 30]}
            }

            model_report:dict = evaluate_model(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, models=models, params=params)
        
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            logging.info('Best found model')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model_name, best_model_score
        
        except Exception as e:
            raise CustomException(e,sys)
