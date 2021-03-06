import os
import sys

import json
import shutil
from datetime import date
from dotenv import load_dotenv

from azureml.core import Run, Experiment, Workspace
from azureml.core.authentication import AzureCliAuthentication


# For local development, set values in this section
load_dotenv()


def getConfiguration(details_file): # get configuration from json
    try:
        with open(details_file) as f:
            config = json.load(f)
    except Exception as e:
        print(e)
        sys.exit(0)

    return config

def downloadModel(run, version): # download model from azure for build
    # get environment variables
    ENV_REGISTER = json.loads(os.environ.get("ENV_REGISTER"))
    ENV_MODEL = json.loads(os.environ.get("ENV_MODEL"))
    # ---
    azure_path = ENV_REGISTER.get("AZURE_OUTPUT")
    download_path = ENV_REGISTER.get("MODEL_FOLDER")
    download_path_abs = ENV_REGISTER.get("MODEL_FOLDER_ABS")
    model_name = ENV_MODEL.get("MODEL_NAME")
    model_extension = ENV_MODEL.get("MODEL_EXTENSION")

    #create model folder
    os.makedirs(f'./{download_path}', exist_ok=True) 

    #variables
    d = date.today().strftime('%d_%m_%Y')
    m = f'{model_name}{model_extension}'
    m_absolute = f'{d}_{model_name}_{version}{model_extension}'
    path_azure = f'{azure_path}/{m}'
    path_local = f'./{download_path}/{m}'
    path_absolute = f'{download_path_abs}/{m_absolute}'

    #download model from run history/outputs
    run.download_file(name=path_azure, output_file_path=path_local)
    path_absolute = shutil.copy(path_local, path_absolute)
    print(f'Downloaded model {m} from {path_azure} on Azure to local {path_local} and absolute {path_absolute}')
    return {'path_azure':path_azure, 'path_local':path_local, 'path_absolute':path_absolute}


def main():
    print('Executing main - 04_DeployModel')
    # -----------------------------------------------------------------
    # get environment variables
    ENV_AZURE = json.loads(os.environ.get("ENV_AZURE"))
    ENV_GENERAL = json.loads(os.environ.get("ENV_GENERAL"))
    # ---
    resource_group = ENV_AZURE.get("RESOURCE_GROUP") # Azure Resource grouo
    subscription_id = ENV_AZURE.get("SUBSCRIPTION_ID") # Azure Subscription ID
    workspace_name = ENV_AZURE.get("WORKSPACE_NAME") # ML Service Workspace of resource group

    temp_state_directory = ENV_GENERAL.get('TEMP_STATE_DIRECTORY')
    experiment_name = ENV_GENERAL.get("EXPERIMENT_NAME")

    # azure authentication
    cli_auth = AzureCliAuthentication()

    # setup workspace + datastore
    ws = Workspace.get(name=workspace_name,
                        subscription_id=subscription_id,
                        resource_group=resource_group,
                        auth=cli_auth)

    # get training details and connect to run
    path_json = os.path.join(temp_state_directory, 'training_run.json')
    config = getConfiguration(path_json)
    exp = Experiment(workspace=ws, name=experiment_name)
    run = Run(experiment=exp, run_id=config['runId'])

    # download model for image build
    path_json = os.path.join(temp_state_directory, 'model_details.json')
    model_details = getConfiguration(path_json)
    download_json = downloadModel(run, version=model_details.get('model').get('version'))
    
    # dump download details into file json
    path_json = os.path.join(temp_state_directory, 'download_details.json')
    with open(path_json, "w") as download_details:
        json.dump(download_json, download_details)

    # -----------------------------------------------------------------
    print('Executing main - 04_DeployModel - SUCCES')
    
    
if __name__ == '__main__':
    main()