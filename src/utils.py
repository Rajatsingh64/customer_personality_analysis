from src.config import mongo_client
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from src.logger import logging
from src.exception import SrcException
import yaml
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score
import os ,sys


def get_collection_as_dataframe(mongo_client, database_name, collection_name):
    """
    This function extracts data from a MongoDB collection and returns it as a pandas DataFrame.

    Parameters:
    - mongo_client (MongoClient): Instance of a MongoDB client connected to MongoDB Atlas.
    - database_name (str): The name of the MongoDB database.
    - collection_name (str): The name of the MongoDB collection.

    Returns:
    - pd.DataFrame: DataFrame containing the data from the collection.
    """
    try:
        # Extract data from the specified collection
        collection = mongo_client[database_name][collection_name]
        
        # Find all documents in the collection and convert to DataFrame
        cursor = collection.find()  # using find() instead of find_all()
        df = pd.DataFrame(list(cursor))  # Convert the cursor to a list and then to a DataFrame
        
        # Removing irrelevant features (e.g., MongoDB's _id)
        if "_id" in df.columns:
            df.drop("_id", axis=1, inplace=True)
        
        return df  # Return the dataframe

    except Exception as e:
        print(f"Error: {e}")  # Print the exception if anything goes wrong
        return pd.DataFrame()  # Return an empty DataFrame in case of an error
     
def write_yaml_file(file_path,data:dict):
    try:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir,exist_ok=True)
        with open(file_path,"w") as file_writer:
            yaml.dump(data,file_writer)
    except Exception as e:
        raise SrcException(e, sys)
    

def convert_columns_float(df: pd.DataFrame) -> pd.DataFrame:
    try:
        for column in df.columns:
            if df[column].dtype == 'object':  # Check if it's a string
                logging.info(f"Skipping conversion for non-numeric column: {column}")
                continue  # Skip non-numeric columns

            try:
                df[column] = df[column].astype('float')
            except ValueError as e:
                    logging.warning(f"Column '{column}' could not be converted to float: {e}")
        return df
    except Exception as e:
        raise e


def silhouette_analysis(df, model="kmeans", cluster_range=[2, 3, 4, 5, 6], random_state=10):
    """
    Perform silhouette analysis for different clustering models on the given dataframe.

    Parameters:
    - df: DataFrame -> The encoded dataset to cluster.
    - model: str -> The clustering model to use ('kmeans', 'agg', 'dbscan').
    - cluster_range: list -> List of cluster numbers to analyze (ignored for DBSCAN).
    - random_state: int -> Random state for reproducibility.

    Returns:
    - dict: A dictionary where keys are the cluster numbers and values are the cluster labels.
    """
    cluster_results = {}

    for n_clusters in cluster_range:
        # Initialize the clustering model
        if model == "kmeans":
            clusterer = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
        elif model == "agg":
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        elif model == "dbscan":
            clusterer = DBSCAN()  # DBSCAN does not use n_clusters
        else:
            raise ValueError("Invalid model. Choose from 'kmeans', 'agg', or 'dbscan'.")

        # Fit and predict clusters
        cluster_labels = clusterer.fit_predict(df)
        cluster_results[n_clusters] = cluster_labels  # Store cluster labels

        # Calculate silhouette score (not valid for DBSCAN if noise points exist)
        if model != "dbscan":
            silhouette_avg = silhouette_score(df, cluster_labels)
            print(f"For n_clusters = {n_clusters}, The average silhouette_score is: {silhouette_avg:.4f}")

        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

        # Silhouette plot settings
        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(df) + (n_clusters + 1) * 10])

        # Compute silhouette scores for each sample (if applicable)
        if model != "dbscan":
            sample_silhouette_values = silhouette_samples(df, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
                y_lower = y_upper + 10  

            ax1.set_title("Silhouette plot for various clusters")
            ax1.set_xlabel("Silhouette coefficient values")
            ax1.set_ylabel("Cluster label")
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
            ax1.set_yticks([])
            ax1.set_xticks(np.arange(-0.1, 1.1, 0.2))

        # Scatter plot for clusters
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(df.iloc[:, 0], df.iloc[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k")

        ax2.set_title("Visualization of clustered data")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            f"Silhouette analysis for {model.upper()} clustering with n_clusters = {n_clusters}",
            fontsize=14,
            fontweight="bold",
        )

        plt.show()
    
    return cluster_results  # Return dictionary of cluster labels


