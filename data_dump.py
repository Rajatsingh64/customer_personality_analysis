# Import necessary libraries
from src.config import mongo_client
import pandas as pd

# Constants
FILE_URL = "https://raw.githubusercontent.com/amankharwal/Website-data/master/marketing_campaign.csv"
DATABASE_NAME = "customer"
COLLECTION_NAME = "records"

def data_dump_mongo_db(mongo_client, data_path, dataset_sep=None):
    """
    Function: data_dump_mongo_db

    Purpose:
    - Reads a CSV file from a given path.
    - Cleans the data by removing missing values.
    - Converts the data into a dictionary format.
    - Inserts the data into a MongoDB collection.

    Parameters:
    - mongo_client (object): MongoDB client instance.
    - data_path (str): URL or file path of the dataset.
    - dataset_sep (str, optional): Delimiter used in the CSV file (default is None).

    Returns:
    - None (prints success or error messages).

    Usage Example:
    data_dump_mongo_db(mongo_client, "data.csv")
    """
    try:
        # Load the dataset
        df = pd.read_csv(data_path, sep=dataset_sep)
        
        # Drop missing values
        df.dropna(inplace=True)
        
        # Convert DataFrame to dictionary format
        dict_data = df.to_dict(orient="records")
        
        # Insert data into MongoDB
        mongo_client[DATABASE_NAME][COLLECTION_NAME].insert_many(dict_data)
        
        print("✅ Successfully dumped data into the database")
    
    except Exception as e:
        print(f"Error: {e}")

# Main execution
if __name__ == "__main__":
    try:
        data_dump_mongo_db(mongo_client=mongo_client, data_path=FILE_URL ,dataset_sep=";")
    except Exception as e:
        print(f"❌ Error: {e}")
