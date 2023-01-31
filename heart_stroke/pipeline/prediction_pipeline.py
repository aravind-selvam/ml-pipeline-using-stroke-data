import logging
import os
import sys

from heart_stroke.constant.training_pipeline import SCHEMA_FILE_PATH
from heart_stroke.entity.config_entity import StrokePredictorConfig
from heart_stroke.entity.s3_estimator import StrokeEstimator
from heart_stroke.exception import HeartStrokeException
from heart_stroke.logger import logging
from heart_stroke.utils.main_utils import read_yaml_file
from pandas import DataFrame


class HeartData:
    def __init__(self, gender: str,
                age : int,
                hypertension : int,
                heart_disease: int,
                ever_married: str,
                work_type : str,
                Residence_type : str,
                avg_glucose_level : float,
                bmi : float,
                smoking_status : str,
                stroke : int = None 
                ):
        """
        Heart Data constructor
        Input: all features of the trained model for prediction
        """
        try:
            
            self.gender = gender
            self.age = age
            self.hypertension = hypertension
            self.heart_disease = heart_disease
            self.ever_married = ever_married
            self.work_type = work_type
            self.Residence_type = Residence_type
            self.avg_glucose_level = avg_glucose_level
            self.bmi = bmi
            self.smoking_status = smoking_status

        except Exception as e:
            raise HeartStrokeException(e, sys) from e

    def get_heart_stroke_input_data_frame(self)-> DataFrame:
        """
        This function returns a DataFrame from HeartData class input
        """
        try:
            
            heart_stroke_input_dict = self.get_heart_stroke_data_as_dict()
            return DataFrame(heart_stroke_input_dict)
        
        except Exception as e:
            raise HeartStrokeException(e, sys) from e

    def get_heart_stroke_data_as_dict(self)-> dict:
        """
        This function returns a dictionary from HeartData class input 
        """
        try:
            input_data = {
                "gender": [self.gender],
                "age": [self.age],
                "hypertension": [self.hypertension],
                "heart_disease": [self.heart_disease],
                "ever_married": [self.ever_married],
                "work_type": [self.work_type],
                "Residence_type": [self.Residence_type],
                "avg_glucose_level": [self.avg_glucose_level],
                "bmi": [self.bmi],
                "smoking_status": [self.smoking_status]
                }
            return input_data
        
        except Exception as e:
            raise HeartStrokeException(e, sys)

class HeartStrokeClassifier:
    def __init__(self,prediction_pipeline_config: StrokePredictorConfig = StrokePredictorConfig(),) -> None:
        """
        :param prediction_pipeline_config:
        """
        try:
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise HeartStrokeException(e, sys)

    def predict(self, dataframe) -> str:
        """
        This is the method of HeartStrokeClassifier
        Returns: Prediction in string format
        """
        try:
            logging.info("Entered predict method of HeartStrokeClassifier class")
            model = StrokeEstimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            result =  model.predict(dataframe)
            if result == 1:
                return "High chance of Heart stroke"
            
            else:
                return "Low chance of Heart stroke"
        
        except Exception as e:
            raise HeartStrokeException(e, sys)
