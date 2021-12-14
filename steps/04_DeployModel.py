import os
import sys
import json
import uuid
import joblib
import argparse
import traceback
from dotenv import load_dotenv

from azureml.core.model import InferenceConfig
from azureml.core.environment import Environment
from azureml.core import Run, Experiment, Workspace, Model
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.webservice import Webservice, AciWebservice
from azureml.core.authentication import AzureCliAuthentication, ServicePrincipalAuthentication

load_dotenv()

def getConfiguration(details_file):
    try:
        with open(details_file) as f:
            config = json.load(f)
    except Exception as e:
        print(e)
        sys.exit(0)

    return config

def downloadModel(run, name_model='model.pt', model_extension='.pt', azure_path='outputs', download_path='modelDownloads'):
    #create model folder
    os.makedirs(f'./{download_path}', exist_ok=True) 
    #download model from run history/outputs
    m = f'{name_model}{model_extension}'
    model_path_azure = f'{azure_path}/{m}'
    model_path_local = f'./{download_path}/{m}'
    run.download_file(name=model_path_azure, output_file_path=model_path_local)
    print(f'Downloaded model {m} from {model_path_azure} on Azure to local {model_path_local}')

def main():
    print('Executing main - 04_DeployModel')

    # get environment variables
    env = os.environ.get("SECRETS_CONTEXT") # Azure Resource grouo
    env = json.loads(env)

    #cli_auth = ServicePrincipalAuthentication(tenant_id=env.get("TENANT_ID"), service_principal_id=env.get("CLIENT_ID"), service_principal_password=env.get("CLIENT_SECRET"))
    cli_auth = AzureCliAuthentication()
    # get environment variables 
    workspace_name = env.get("WORKSPACE_NAME")
    resource_group = env.get("RESOURCE_GROUP")
    subscription_id = env.get("SUBSCRIPTION_ID")

    experiment_name = env.get("EXPERIMENT_NAME")

    model_name = env.get("MODEL_NAME")
    model_extension = env.get("MODEL_EXTENSION")
    azure_path = env.get("AZURE_OUTPUT")
    download_path = env.get("MODEL_FOLDER")

    #environment = env.get("AML_ENV_NAME")
    #score_script_path = os.path.join(env.get("ROOT_DIR"), 'scripts', 'score.py')
    temp_state_directory = env.get('TEMP_STATE_DIRECTORY')

    # setup workspace + datastore
    ws = Workspace.get(
        name=workspace_name,
        subscription_id=subscription_id,
        resource_group=resource_group,
        auth=cli_auth
    )

    # get run
    path_json = os.path.join(temp_state_directory, 'training_run.json')
    config = getConfiguration(path_json)
    exp = Experiment(workspace=ws, name=experiment_name)
    run = Run(experiment=exp, run_id=config['runId'])

    # download model to
    downloadModel(run, model_name, model_extension, azure_path, download_path)
    
    # Finish
    print('Executing main - 04_DeployModel - SUCCES')
    
    

if __name__ == '__main__':
    main()
