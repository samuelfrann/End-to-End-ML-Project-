import os
import sys
import pandas as pd
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
@dataclass
class DataIngestConfig:
    # These are paths to FILES
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.xlsx')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestConfig()

    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion method or component')
        try:
            # 1. Read the data
            df = pd.read_excel(r'C:\Users\pc\Documents\JUPYTER ML\End-to-End-ML-Project-\src\notebook\data\US Insurance Claims Data (1).xlsx')
            logging.info('Read the dataset as dataframe')

            # 2. CREATE THE DIRECTORY FIRST
            # This extracts 'artifacts' from the path and creates the folder
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # 3. Save the RAW data first
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # 4. Perform Train-Test Split
            logging.info('Train test split initiated')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # 5. Save the split files
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Data Ingestion Completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    obj = DataIngestion()
    train_data, test_data, = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation( train_data, test_data)