import sys

from heart_stroke.exception import HeartStrokeException
from heart_stroke.logger import logging
from heart_stroke.entity.artifact_entity import ModelPusherArtifact, ModelTrainerArtifact 
from heart_stroke.entity.config_entity import ModelPusherConfig
from heart_stroke.entity.s3_estimator import StrokeEstimator


class ModelPusher:
    def __init__(self,model_trainer_artifact: ModelTrainerArtifact,
                 model_pusher_config: ModelPusherConfig,):

        self.model_trainer_artifact = model_trainer_artifact
        self.model_pusher_config = model_pusher_config
        self.stroke_estimator = StrokeEstimator(
            bucket_name=model_pusher_config.bucket_name,
            model_path=model_pusher_config.s3_model_key_path,
        )

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model pusher

        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_model_pusher method of ModelPusher class")
        try:
            logging.info("Uploading artifacts folder to s3 bucket")
            self.stroke_estimator.save_model(
                from_file=self.model_trainer_artifact.trained_model_file_path
            )
            model_pusher_artifact = ModelPusherArtifact(
                bucket_name=self.model_pusher_config.bucket_name,
                s3_model_path=self.model_pusher_config.s3_model_key_path,
            )
            logging.info("Uploaded artifacts folder to s3 bucket")
            logging.info(f"Model pusher artifact: [{model_pusher_artifact}]")
            logging.info("Exited initiate_model_pusher method of ModelPusher class")
            return model_pusher_artifact
        
        except Exception as e:
            raise HeartStrokeException(e, sys) from e
