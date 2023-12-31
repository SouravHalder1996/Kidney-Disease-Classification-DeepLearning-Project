from src.Kidney_Disease_Classifier import logger
from src.Kidney_Disease_Classifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.Kidney_Disease_Classifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelPipeline
from src.Kidney_Disease_Classifier.pipeline.stage_03_model_training import ModelTrainingPipeline
from src.Kidney_Disease_Classifier.pipeline.stage_04_model_evaluation import EvaluationPipeline



STAGE_NAME = "Data Ingestion"

try:
    logger.info(f">>>>> Stage {STAGE_NAME} Started <<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>> Stage {STAGE_NAME} Completed <<<<<\n\nx==========x")
    
except Exception as e:
    logger.exception(e)
    raise e



STAGE_NAME = "Prepare Base Model"

try:
    logger.info(f"******************************")
    logger.info(f">>>>> Stage {STAGE_NAME} Started <<<<<")
    prepare_base_model = PrepareBaseModelPipeline()
    prepare_base_model.main()
    logger.info(f">>>>> Stage {STAGE_NAME} Completed <<<<<\n\nx==========x")

except Exception as e:
    logger.exception(e)
    raise e



STAGE_NAME = "Training"

try:
    logger.info(f"******************************")
    logger.info(f">>>>> Stage {STAGE_NAME} Started <<<<<")
    model_trainer = ModelTrainingPipeline()
    model_trainer.main()
    logger.info(f">>>>> Stage {STAGE_NAME} Completed <<<<<\n\nx==========x")

except Exception as e:
    logger.exception(e)
    raise e

    

STAGE_NAME = "Evaluation Stage"

try:
    logger.info(f"******************************")
    logger.info(f">>>>> Stage {STAGE_NAME} Started <<<<<")
    model_evaluation = EvaluationPipeline()
    model_evaluation.main()
    logger.info(f">>>>> Stage {STAGE_NAME} Completed <<<<<\n\nx==========x")

except Exception as e:
    logger.exception(e)
    raise e


