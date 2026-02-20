import os
from dotenv import load_dotenv
import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.tensorflow
from urllib.parse import urlparse
import logging
from Kidney_Disease_Classifier.entity.config_entity import EvaluationConfig
from Kidney_Disease_Classifier.utils.common import read_yaml, create_directories, save_json

load_dotenv()
logger = logging.getLogger(__name__)


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split = 0.30
        )

        dataflow_kwargs = dict(
            target_size = self.config.params_image_size[:-1],
            batch_size = self.config.params_batch_size,
            interpolation = "bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory = self.config.training_data,
            subset = "validation",
            shuffle = False,
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    
    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)


    def log_into_mlflow(self):
        # mlflow.set_registry_uri(self.config.mlflow_uri)
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
        os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

        try:
            mlflow.set_experiment("Kidney-Disease-Classification")
        except:
            mlflow.create_experiment("Kidney-Disease-Classification")
            mlflow.set_experiment("Kidney-Disease-Classification")
            
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )

            try:
                if tracking_url_type_store != "file":
                    mlflow.tensorflow.log_model(self.model, name="VGG16-Kidney-Disease-Classifier", registered_model_name="VGG16-Kidney-Disease-Classifier")
                else:
                    mlflow.tensorflow.log_model(self.model, name="VGG16-Kidney-Disease-Classifier")
            except Exception as e:
                logger.warning(f"Failed to log model to MLflow: {str(e)}. Metrics and parameters have been logged successfully.")
