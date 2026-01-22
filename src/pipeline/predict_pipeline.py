import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            model_path = r'C:\Users\pc\Documents\JUPYTER ML\End-to-End-ML-Project-\artifacts\model.pkl'
            preprocessor_path = r'C:\Users\pc\Documents\JUPYTER ML\End-to-End-ML-Project-\artifacts\preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
        policy_number: str,
        age: int,
        umbrella_limit: int,
        claim_amount: float,
        policy_annual_premium: float,
        number_of_vehicles_involved: int,
        incident_hour_of_the_day: int,
        bodily_injuries: int,
        witnesses: int,
        auto_year: int,
        policy_deductable: int,
        insured_sex: str,
        insured_education_level: str,
        collision_type: str,
        police_report_available: str,
        policy_state: str,
        policy_csl: str,
        insured_occupation: str,
        incident_type: str,
        incident_severity: str,
        authorities_contacted: str,
        property_damage: str):

        self.policy_number = policy_number
        self.age = age
        self.umbrella_limit = umbrella_limit
        self.claim_amount = claim_amount
        self.policy_annual_premium = policy_annual_premium
        self.number_of_vehicles_involved = number_of_vehicles_involved
        self.incident_hour_of_the_day = incident_hour_of_the_day
        self.bodily_injuries = bodily_injuries
        self.witnesses = witnesses
        self.auto_year = auto_year
        self.policy_deductable = policy_deductable
        self.insured_sex = insured_sex
        self.insured_education_level = insured_education_level
        self.collision_type = collision_type
        self.police_report_available = police_report_available
        self.policy_state = policy_state
        self.policy_csl = policy_csl
        self.insured_occupation = insured_occupation
        self.incident_type = incident_type
        self.incident_severity = incident_severity
        self.authorities_contacted = authorities_contacted
        self.property_damage = property_damage

    def get_data_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                'policy_number': [self.policy_number],
                'age': [self.age],
                'umbrella_limit': [self.umbrella_limit],
                'claim_amount': [self.claim_amount],
                'policy_annual_premium': [self.policy_annual_premium],
                'number_of_vehicles_involved': [self.number_of_vehicles_involved],
                'incident_hour_of_the_day': [self.incident_hour_of_the_day],
                'bodily_injuries': [self.bodily_injuries],
                'witnesses': [self.witnesses],
                'auto_year': [self.auto_year],
                'policy_deductable': [self.policy_deductable],
                'insured_sex': [self.insured_sex],
                'insured_education_level': [self.insured_education_level],
                'collision_type': [self.collision_type],
                'police_report_available': [self.police_report_available],
                'policy_state': [self.policy_state],
                'policy_csl': [self.policy_csl],
                'insured_occupation': [self.insured_occupation],
                'incident_type': [self.incident_type],
                'incident_severity': [self.incident_severity],
                'authorities_contacted': [self.authorities_contacted],
                'property_damage': [self.property_damage],

                'total_claim_amount': [self.claim_amount], 
                'months_as_customer': [0],
                'capital-gains': [0],
                'capital-loss': [0],
                'insured_zip': [0],
                'injury_claim': [0],
                'property_claim': [0],
                'vehicle_claim': [0],
                'incident_location': ["Unknown"],
                'incident_city': ["Unknown"],
                'incident_state': [self.policy_state], # Reusing state from form
                'insured_relationship': ["Unknown"],
                'insured_hobbies': ["Unknown"],
                'make_model': ["Unknown"]
                }

            return pd.DataFrame(custom_data_input_dict)
        

        except Exception as e:
            raise CustomException(e,sys)
        