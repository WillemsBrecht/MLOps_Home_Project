import argparse
import json
import os
from re import T
import sys
import traceback
from glob import glob
import math

import joblib
import numpy as np
import matplotlib.pyplot as plt

from azureml.core import Dataset, Datastore, Experiment, Run, Workspace
from azureml.core.authentication import AzureCliAuthentication, ServicePrincipalAuthentication
from azureml.data.datapath import DataPath

from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# For local development, set values in this section
load_dotenv()


def uploadData(env, data_folder, ws, datastore): # download our data
    # get environment variables
    dataset_name = env.get('DATASET_NAME') #name of dataset
    dataset_description = env.get('DATASET_DESCRIPTION')  #desc of dataset
    dataset_new_version = env.get('DATASET_NEW_VERSION') == 'true'
    show_progress_dataprep = env.get('SHOW_PROGRESS_DATAPREP') == 'true'

    # upload dataset to workspace
    ds_target = DataPath(datastore, dataset_name)
    dataset_obj = Dataset.File.upload_directory(src_dir=data_folder, target=ds_target, show_progress=show_progress_dataprep, overwrite=False)
    dataset_obj = dataset_obj.register(workspace=ws, name=dataset_name, description=dataset_description, create_new_version=dataset_new_version)

    #return dataset variables
    return {'name' : dataset_name, 'description' : dataset_description}


def main():
    print('Executing main - 01_DataPreparing')


    # get environment variables
    ENV_AZURE = json.loads(os.environ.get("ENV_AZURE"))
    ENV_GENERAL = json.loads(os.environ.get("ENV_GENERAL"))
    ENV_DATA = json.loads(os.environ.get("ENV_DATA"))
    print(ENV_AZURE)

    resource_group = ENV_AZURE.get("RESOURCE_GROUP") # Azure Resource grouo
    subscription_id = ENV_AZURE.get("SUBSCRIPTION_ID") # Azure Subscription ID
    workspace_name = ENV_AZURE.get("WORKSPACE_NAME") # ML Service Workspace of resource group
    temp_state_directory = ENV_GENERAL.get('TEMP_STATE_DIRECTORY')

    # azure authentication
    cli_auth = AzureCliAuthentication()

    # setup workspace + datastore
    print(f'Connect to workspace {workspace_name}')
    ws = Workspace.get(name=workspace_name, 
                        subscription_id=subscription_id, 
                        resource_group=resource_group, 
                        auth=cli_auth)
    datastore = Datastore(ws)

    # download workspace
    data_folder = os.path.join(os.getcwd(), ENV_DATA.get('DATA_FOLDER'))
    data_result = uploadData(ENV_DATA, data_folder, ws, datastore)

    # create temporary directory
    os.makedirs(temp_state_directory, exist_ok=True)

    # dump dictionnary into file dataset.json
    path_json = os.path.join(temp_state_directory, 'dataset.json')
    with open(path_json, 'w') as dataset_json:
        json.dump(data_result, dataset_json)


    # -----------------------------------------------------------------
    print('Executing main - 01_DataPreparing - SUCCES')


if __name__ == '__main__':
    main()
