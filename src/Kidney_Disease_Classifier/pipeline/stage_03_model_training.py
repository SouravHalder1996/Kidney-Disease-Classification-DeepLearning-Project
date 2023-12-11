from src.Kidney_Disease_Classifier.config.configuration import ConfigurationManager
from src.Kidney_Disease_Classifier.components.model_training import Training
from src.Kidney_Disease_Classifier import logger


STAGE_NAME = "Training"


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.train_valid_generator()
        training.train()


if __name__ == '__main__':
    try:
        logger.info(f"******************************")
        logger.info(f">>>>> Stage {STAGE_NAME} Started <<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>> Stage {STAGE_NAME} Completed <<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e