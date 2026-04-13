import os
import zipfile
import gdown
from kidney_disease_classifier import logger
from kidney_disease_classifier.utils.common import get_size
from kidney_disease_classifier.entity.config_entity import (DataIngestionConfig)


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    
    def download_file(self)-> str:
        '''
        Fetch data from the url
        '''

        try: 
            dataset_url = self.config.source_url
            zip_download_dir = self.config.local_data_file

            if not os.path.exists(zip_download_dir):
                os.makedirs(self.config.root_dir, exist_ok=True)
                logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

                file_id = dataset_url.split("/")[-2]
                prefix = 'https://drive.google.com/uc?/export=download&id='
                
                download_url = prefix + file_id
                gdown.download(download_url, zip_download_dir)

                logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")
            else:
                logger.info(f"Dataset already exists, moving on to data extraction.")

        except Exception as e:
            raise e
        
    

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        extraction_dir = os.path.join(unzip_path, "raw")
        if not os.path.exists(extraction_dir):
            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(extraction_dir)
