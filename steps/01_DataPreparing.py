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
from azureml.core.authentication import AzureCliAuthentication
from azureml.data.datapath import DataPath

from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# For local development, set values in this section
load_dotenv()


def uploadData(data_folder, ws, datastore): # download our data
    # get environment variables
    dataset_name = os.environ.get('DATASET_NAME') #name of dataset
    dataset_description = os.environ.get('DATASET_DESCRIPTION')  #desc of dataset
    dataset_new_version = os.environ.get('DATASET_NEW_VERSION') == 'true'

    # upload dataset to workspace
    ds_target = DataPath(datastore, dataset_name)
    dataset_obj = Dataset.File.upload_directory(src_dir=data_folder, target=ds_target, show_progress=True, overwrite=True)
    dataset_obj = dataset_obj.register(workspace=ws, name=dataset_name, description=dataset_description, create_new_version=dataset_new_version)

    #return dataset variables
    return {'name' : dataset_name, 'description' : dataset_description}


def main():
    print('Executing main - 01_DataPreparing')
    # authentication Microsoft Azure (login on VM)
    cli_auth = AzureCliAuthentication()

    # get environment variables
    resource_group = os.environ.get("RESOURCE_GROUP") # Azure Resource grouo
    subscription_id = os.environ.get("SUBSCRIPTION_ID") # Azure Subscription ID
    workspace_name = os.environ.get("WORKSPACE_NAME") # ML Service Workspace of resource group
    temp_state_directory = os.environ.get('TEMP_STATE_DIRECTORY')

    # setup workspace + datastore
    ws = Workspace.get(name=workspace_name, 
                        subscription_id=subscription_id, 
                        resource_group=resource_group, 
                        auth=cli_auth)
    datastore = Datastore(ws)

    # download workspace
    data_folder = os.path.join(os.getcwd(), os.environ.get('DATA_FOLDER'))
    data_result = uploadData(data_folder, ws, datastore)

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
