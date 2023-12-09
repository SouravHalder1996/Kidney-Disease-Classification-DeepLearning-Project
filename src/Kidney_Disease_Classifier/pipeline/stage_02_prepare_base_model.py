from src.Kidney_Disease_Classifier.config.configuration import ConfigurationManager
from src.Kidney_Disease_Classifier.components.prepare_base_model import PrepareBaseModel
from src.Kidney_Disease_Classifier import logger


STAGE_NAME = "Prepare Base Model"

class PrepareBaseModelPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_base_model_cofig = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=config.get_prepare_base_model_config())
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()


if __name__ == '__main__':
    try:
        logger.info(f">>>>> Stage {STAGE_NAME} Started <<<<<")
        obj = PrepareBaseModelPipeline()
        obj.main()
        logger.info(f">>>>> Stage {STAGE_NAME} Completed <<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e