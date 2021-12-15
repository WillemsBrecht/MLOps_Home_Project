import os
import sys

import json
from dotenv import load_dotenv

from azureml.data.datapath import DataPath
from azureml.core import Dataset, Datastore, Workspace
from azureml.core.authentication import AzureCliAuthentication


# For local development, set values in this section
load_dotenv()


def uploadData(data_folder, ws, datastore): # download our data
    # get environment variables
    ENV_DATA = json.loads(os.environ.get("ENV_DATA"))
    # ---
    dataset_name = ENV_DATA.get('DATASET_NAME') #name of dataset
    dataset_description = ENV_DATA.get('DATASET_DESCRIPTION')  #desc of dataset
    dataset_new_version = ENV_DATA.get('DATASET_NEW_VERSION') == 'true'
    show_progress_dataprep = ENV_DATA.get('SHOW_PROGRESS_DATAPREP') == 'true'

    # upload dataset to workspace
    ds_target = DataPath(datastore, dataset_name)
    dataset_obj = Dataset.File.upload_directory(src_dir=data_folder, 
                                                target=ds_target, 
                                                show_progress=show_progress_dataprep, 
                                                overwrite=False)
    dataset_obj = dataset_obj.register(workspace=ws, 
                                        name=dataset_name, 
                                        description=dataset_description, 
                                        create_new_version=dataset_new_version)

    #return dataset variables
    return {'name' : dataset_name, 'description' : dataset_description}


def main():
    print('Executing main - 01_DataPreparing')
    # -----------------------------------------------------------------
    # get environment variables
    ENV_AZURE = json.loads(os.environ.get("ENV_AZURE"))
    ENV_GENERAL = json.loads(os.environ.get("ENV_GENERAL"))
    ENV_DATA = json.loads(os.environ.get("ENV_DATA"))
    # ---
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
    data_result = uploadData(data_folder, ws, datastore)

    # create temporary directory & dump step results into file json
    os.makedirs(temp_state_directory, exist_ok=True)
    # ---
    path_json = os.path.join(temp_state_directory, 'dataset.json')
    with open(path_json, 'w') as dataset_json:
        json.dump(data_result, dataset_json)

    # -----------------------------------------------------------------
    print('Executing main - 01_DataPreparing - SUCCES')


if __name__ == '__main__':
    main()
