import os 
import sys
import numpy as np 
import pandas as pd 
import dill


from src.exception import CustomExecption
from sklearn.metrics import r2_score
def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomExecption(e,sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for model_name, model in models.items():  # loop with both name and model
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)
            report[model_name] = test_model_score
        return report
    except Exception as e:
        raise CustomExecption(e, sys)

def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:\
            return dill.load(file_obj)
    except Exception as e:
        raise CustomExecption(e,sys)