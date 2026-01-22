import os 
import sys
import numpy as np
import pandas as pd
import dill 

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e,sys)


def evaluate_model(x_train, x_test, y_train, y_test, models, params):
    try:
        report = {}

        
        for model_name, model in models.items():

            param_grid = params.get(model_name, {})

            if param_grid:
                gs = GridSearchCV(
                    model,
                    param_grid,
                    cv=3,
                    n_jobs=-1
                )
            gs.fit(x_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(x_train, y_train)

            y_pred = model.predict(x_test)

            test_model_score = accuracy_score(y_test, y_pred)

            report[model_name] = test_model_score
        return report 


    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)