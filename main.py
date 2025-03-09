from src.pipeline.training_pipeline import run_training_pipeline
from src.pipeline.clustered_pipeline import start_Cluster_prediction
from src.exception import SrcException 
from src.logger import logging
import os,sys

file_path=os.path.join(os.getcwd() , "dataset/customer.csv")

if __name__=="__main__":

    try:
        
        
        print(f"{'>'*20} Running Training Pipeline {'<'*20}")
        run_training_pipeline() #runing training pipline
        
        print(f"{'>'*20} Running Cluster Prediction Pipeline {'<'*20}")
        start_Cluster_prediction(input_file_path=file_path)
        

    except Exception as e:
        raise SrcException(e,sys)