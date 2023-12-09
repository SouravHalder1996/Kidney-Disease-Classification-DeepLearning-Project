from src.Kidney_Disease_Classifier import logger
from src.Kidney_Disease_Classifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline


STAGE_NAME = "Data Ingestion"

try:
    logger.info(f">>>>> Stage {STAGE_NAME} Started <<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>> Stage {STAGE_NAME} Completed <<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e



