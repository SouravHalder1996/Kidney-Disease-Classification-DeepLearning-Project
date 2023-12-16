from Kidney_Disease_Classifier.config.configuration import ConfigurationManager
from Kidney_Disease_Classifier.components.model_evaluation_mlflow import Evaluation
from Kidney_Disease_Classifier import logger


STAGE_NAME = "Evaluation Stage"


class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(eval_config)
        evaluation.evaluation()
        evaluation.save_score()
        # evaluation.log_into_mlflow()



if __name__ == '__main__':
    try:
        logger.info(f"******************************")
        logger.info(f">>>>> Stage {STAGE_NAME} Started <<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>> Stage {STAGE_NAME} Completed <<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e