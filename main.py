from src.text_summarizer.logging import logger
from src.text_summarizer.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.text_summarizer.pipeline.data_transformation_pipeline import DataTransformationTrainingPipeline
from src.text_summarizer.pipeline.model_trainer_pipeline import ModelTrainingPipeline
from src.text_summarizer.pipeline.model_evaluation_pipeline import ModelEvaluationPipeline

STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f"stage {STAGE_NAME} initiated")
    data_ingestion_pipeline = DataIngestionTrainingPipeline()
    data_ingestion_pipeline.initiate_data_ingestion()
    logger.info(f"stage {STAGE_NAME} completed")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Transformation Stage"

try:
    logger.info(f"stage {STAGE_NAME} initiated")
    data_transformation_pipeline = DataTransformationTrainingPipeline()
    data_transformation_pipeline.initiate_data_transformation()
    logger.info(f"stage {STAGE_NAME} completed")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Training Stage"
try:
    logger.info(f"stage {STAGE_NAME} initiated")
    model_trainer_pipeline = ModelTrainingPipeline()
    model_trainer_pipeline.initiate_model_trainer()
    logger.info(f"stage {STAGE_NAME} completed")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Evaluation Stage"
try:
    logger.info(f"***********************************")
    logger.info(f"stage {STAGE_NAME} initiated")
    model_evaluation = ModelEvaluationPipeline()
    model_evaluation.initiate_model_evaluation()
    logger.info(f"stage {STAGE_NAME} completed")
except Exception as e:
    logger.exception(e)
    raise e
