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
    model = run.register_model(model_name=model_name, model_path=f'outputs/{model_name}{model_extension}', tags={"runId": run.id}, description=description)
    print("Model registered: {} \nModel Description: {} \nModel Version: {}".format(model.name, model.description, model.version))
    return model

def main():
    print('Executing main - 03_RegisterModel')

    # get environment variables
    env = os.environ.get("SECRETS_CONTEXT") # Azure Resource grouo
    env = json.loads(env)

    #cli_auth = ServicePrincipalAuthentication(tenant_id=env.get("TENANT_ID"), service_principal_id=env.get("CLIENT_ID"), service_principal_password=env.get("CLIENT_SECRET"))
    cli_auth = AzureCliAuthentication()
    # get environment variables 
    workspace_name = env.get("WORKSPACE_NAME")
    resource_group = env.get("RESOURCE_GROUP")
    subscription_id = env.get("SUBSCRIPTION_ID")
    
    model_name = env.get("MODEL_NAME")
    model_extension = env.get("MODEL_EXTENSION")
    model_description = env.get("MODEL_DESCRIPTION")
    experiment_name = env.get("EXPERIMENT_NAME")

    temp_state_directory = env.get('TEMP_STATE_DIRECTORY')

    # setup workspace + datastore
    ws = Workspace.get(
        name=workspace_name,
        subscription_id=subscription_id,
        resource_group=resource_group,
        auth=cli_auth
    )

    # create experiment
    path_json = os.path.join(temp_state_directory, 'training_run.json')
    config = getConfiguration(path_json)
    exp = Experiment(workspace=ws, name=experiment_name)
    run = Run(experiment=exp, run_id=config['runId'])

    #-TODO ------------------------------
    def checkModel():
        # DUMMY CODE
        # Get Model in production
        model_in_production = None
        new_model = None
        old_acc = model_in_production.metrics.get('accuracy')
        new_acc = new_model.metrics.get('accuracy')
        return new_acc > old_acc
    #-------------------------------

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

    # Finish
    print('Executing main - 03_RegisterModel - SUCCES')

if __name__ == '__main__':
    main()
