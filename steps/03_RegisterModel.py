import os
import sys
import json
import joblib
import argparse
import traceback
from dotenv import load_dotenv
from azureml.core import Run, Experiment, Workspace, Model
from azureml.core.authentication import AzureCliAuthentication
# For local development, set values in this section
load_dotenv()

def getConfiguration(details_file):
    try:
        with open(details_file) as f:
            config = json.load(f)
    except Exception as e:
        sys.exit(0)
    return config

def registerModel(model_name, description, run):
    model = run.register_model(model_name=model_name, model_path='outputs/keras_example.onnx', tags={"runId": run.id}, description=description)
    print("Model registered: {} \nModel Description: {} \nModel Version: {}".format(model.name, model.description, model.version))
    return model

def main():
    print('Executing main - 03_RegisterModel')

    # authentication
    cli_auth = AzureCliAuthentication()

    # get environment variables 
    workspace_name = os.environ.get("WORKSPACE_NAME")
    resource_group = os.environ.get("RESOURCE_GROUP")
    subscription_id = os.environ.get("SUBSCRIPTION_ID")
    
    model_name = os.environ.get("MODEL_NAME")
    model_description = os.environ.get("MODEL_DESCRIPTION")
    experiment_name = os.environ.get("EXPERIMENT_NAME")

    temp_state_directory = os.environ.get('TEMP_STATE_DIRECTORY')

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
    model = registerModel(model_name, model_description, run)
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