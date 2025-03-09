from dotenv import load_dotenv
from dataclasses import dataclass
from src.logger import logging
import pymongo as pm
import os ,sys

logging.info(f"loading .env file")
print(f"\nloading .env file")
load_dotenv()


#Creating class for environments variables , i can directly load all sensitive info without showing anyone
dataclass 
class EnvironmentVariables:
    mongo_url:str = os.getenv("MONGO_URL")


#creating instance/object for class
env=EnvironmentVariables()

mongo_client=pm.MongoClient(env.mongo_url)
print(f"Connected to MongoDB database")
logging.info(f"Connected to MongoDB database")

features_to_drop=["ID" , "Year_Birth" ,'Dt_Customer',
                  'year_enroll','AcceptedCmp5','AcceptedCmp3',
                  'AcceptedCmp4','AcceptedCmp1' ,'AcceptedCmp2' ,
                  'Z_CostContact','Complain' , 'Z_Revenue' ,'Response' ]

outliers_handling_features= ['Age' , 'Income','Income_to_Spending_Ratio' , "customer_spending"] #Important Numericals Features to be handled