import os
import zipfile
import gdown
import tempfile
import shutil
from Kidney_Disease_Classifier import logger
from Kidney_Disease_Classifier.utils.common import get_size
from Kidney_Disease_Classifier.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self) -> str:
        '''
        Fetch data from the url
        Download to a temporary location first to avoid DVC cache conflicts
        '''

        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            
            # Download to a temporary file first to avoid gdown's temp file conflicts with DVC cache
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                tmp_path = tmp_file.name
            
            try:
                # Download to the temp file
                gdown.download(prefix+file_id, tmp_path, quiet=False)
                
                # Move the temp file to the final location, replacing if exists
                if os.path.exists(zip_download_dir):
                    os.remove(zip_download_dir)
                shutil.move(tmp_path, zip_download_dir)
                
                logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")
            except Exception as download_error:
                # Clean up temp file if download fails
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise download_error
                
        except Exception as e:
            raise e
        
    def extract_zip_file(self) -> None:
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """

        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)