import sys 
import os
from dataclasses import dataclass

import pandas as pd
import numpy as np 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_transformer_object(self):
        try:
            numerical_columns = ['months_as_customer', 'age', 'policy_number', 'policy_deductable',
            'policy_annual_premium', 'umbrella_limit', 'insured_zip',
            'capital-gains', 'capital-loss', 'incident_hour_of_the_day',
            'number_of_vehicles_involved', 'bodily_injuries', 'witnesses',
            'total_claim_amount', 'injury_claim', 'property_claim', 'vehicle_claim',
            'auto_year']
            categorical_columns = ['policy_state', 'policy_csl', 'insured_sex', 'insured_education_level',
            'insured_occupation', 'insured_hobbies', 'insured_relationship',
            'incident_type', 'collision_type', 'incident_severity',
            'authorities_contacted', 'incident_state', 'incident_city',
            'incident_location', 'property_damage', 'police_report_available',
            'auto_make', 'auto_model']
                    
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                    ('scaler', StandardScaler())
                ]
            )

            logging.info('Numerical columns enocded completed')
            logging.info('Categorical columns enocded completed')

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )
            return preprocessor 
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):
            try:
                train_df = pd.read_csv(train_path)
                test_df = pd.read_csv(test_path)

                logging.info('Read train and test set complete')
                logging.info('Obtaining preprocessor  object')
                
                preprocessing_obj = self.get_transformer_object()

                target_column_name = 'fraud_reported'

                numerical_columns = ['months_as_customer', 'age', 'policy_number', 'policy_deductable',
                'policy_annual_premium', 'umbrella_limit', 'insured_zip',
                'capital-gains', 'capital-loss', 'incident_hour_of_the_day',
                'number_of_vehicles_involved', 'bodily_injuries', 'witnesses',
                'total_claim_amount', 'injury_claim', 'property_claim', 'vehicle_claim',
                'auto_year']

                input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
                target_feature_train_df = train_df[target_column_name]

                input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
                target_feature_test_df = test_df[target_column_name]

                logging.info('Applying preprocessing object on tarining and testing dataframe')

                input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
                input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

                train_arr = np.c_[
                    input_feature_train_arr, np.array(target_feature_train_df)
                ]
                test_arr = np.c_[
                    input_feature_test_arr, np.array(target_feature_test_df)
                ]

                logging.info('Saved preprocessing object')    
            
                save_object(
                    file_path = self.data_transformation_config.preprocessor_obj_file_path,
                    obj=preprocessing_obj
                )

                return(
                    train_arr,
                    test_arr,
                    self.data_transformation_config.preprocessor_obj_file_path,
                )

            except Exception as e:
                raise CustomException(e,sys)



