from src.pipeline.training_pipeline import run_training_pipeline
from src.exception import SrcException 
from src.logger import logging
import os,sys

if __name__=="__main__":

    try:
        training_pipeline=run_training_pipeline() #runing training pipline

    except Exception as e:
        raise SrcException(e,sys)