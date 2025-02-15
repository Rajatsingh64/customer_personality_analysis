from dotenv import load_dotenv
from dataclasses import dataclass
import pymongo as pm
import os ,sys

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