import os
import sys
import json
import joblib
import argparse
import traceback
from dotenv import load_dotenv
from azureml.core import Run, Experiment, Workspace, Model
from azureml.core.authentication import AzureCliAuthentication, ServicePrincipalAuthentication
# For local development, set values in this section
load_dotenv()

def getConfiguration(details_file):
    try:
        with open(details_file) as f:
            config = json.load(f)
    except Exception as e:
        sys.exit(0)
    return config

def registerModel(model_name, model_extension, description, run):
    # register model for outputs
    model = run.register_model(model_name=model_name, model_path=f'outputs/{model_name}{model_extension}', tags={"runId": run.id}, description=description)
    print("Model registered: {} \nModel Description: {} \nModel Version: {}".format(model.name, model.description, model.version))
    return model

def downloadModel(run, name_model='model.pt', model_extension='.pt', azure_path='outputs', download_path='modelDownloads'):
    #create model folder
    os.makedirs(f'./{download_path}', exist_ok=True) 

    #download model from run history/outputs
    m = f'{name_model}{model_extension}'
    model_path_azure = f'{azure_path}/{m}'
    model_path_local = f'./{download_path}/{m}'
    run.download_file(name=model_path_azure, output_file_path=model_path_local)
    print(f'Downloaded model {m} from {model_path_azure} on Azure to local {model_path_local}')
    return model_path_azure, model_path_local


def main():
    print('Executing main - 03_RegisterModel')

    # get environment variables
    env = os.environ.get("SECRETS_CONTEXT") # Azure Resource grouo
    env = json.loads(env)

    # azure authentication
    cli_auth = AzureCliAuthentication()

    # get environment variables 
    workspace_name = env.get("WORKSPACE_NAME")
    resource_group = env.get("RESOURCE_GROUP")
    subscription_id = env.get("SUBSCRIPTION_ID")
    
    model_name = env.get("MODEL_NAME")
    model_extension = env.get("MODEL_EXTENSION")
    model_description = env.get("MODEL_DESCRIPTION")
    experiment_name = env.get("EXPERIMENT_NAME")

    azure_path = env.get("AZURE_OUTPUT")
    download_path = env.get("MODEL_FOLDER")
    temp_state_directory = env.get('TEMP_STATE_DIRECTORY')

    # setup workspace
    ws = Workspace.get(
        name=workspace_name,
        subscription_id=subscription_id,
        resource_group=resource_group,
        auth=cli_auth
    )

    # connect to experiment
    path_json = os.path.join(temp_state_directory, 'training_run.json')
    config = getConfiguration(path_json)
    exp = Experiment(workspace=ws, name=experiment_name)
    run = Run(experiment=exp, run_id=config['runId'])

    # register model
    model = registerModel(model_name, model_extension, model_description, run)
    model_json = {}
    model_json["model"] = model.serialize()
    model_json["run"] = config
    print(model_json)

    # save model details
    path_json = os.path.join(temp_state_directory, 'model_details.json')
    with open(path_json, "w") as model_details:
        json.dump(model_json, model_details)

    # download model for build
    path_azure, path_local = downloadModel(run, model_name, model_extension, azure_path, download_path)
    download_json = {'path_azure':path_local, 'path_local':path_local}

    # save download details
    path_json = os.path.join(temp_state_directory, 'download_details.json')
    with open(path_json, "w") as download_details:
        json.dump(download_json, download_details)

    # Finish
    print('Executing main - 03_RegisterModel - SUCCES')

if __name__ == '__main__':
    main()
