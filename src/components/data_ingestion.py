import os
import sys
import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass 
from pathlib import Path

class CustomException(Exception):
    def __init__(self, original_exception, sys_module):
        super().__init__(f"{original_exception}")
        self.sys_module = sys_module
@dataclass
class DataIngestConfig:
    train_data_path: str=os.path.join('artifacts', 'train.csv')
    test_data_path: str=os.path.join('artifacts', 'test.csv')
    raw_data_path: str=os.path.join('artifacts', 'raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestConfig()

    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion method or component')
        try:
            BASE_DIR = Path(__file__).resolve().parent.parent.parent
            data_file = BASE_DIR / "Datasets" / "Insurance fraud detection" / "US Insurance Claims Data (1).xlsx"
            df=pd.read_excel(r'Datasets\Insurance fraud detection\US Insurance Claims Data (1).xlsx')
            logging.info('Read the dataset as dataframe')

            os.makedirs((self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            logging.info('Train test split initiated')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info('Data Ingestion Completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
            pass

if __name__ == '__main__':
    obj=DataIngestion()
    obj.initiate_data_ingestion()