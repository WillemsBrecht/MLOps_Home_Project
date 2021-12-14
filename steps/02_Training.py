import os
import sys
import json
import joblib
import argparse
import traceback
from dotenv import load_dotenv

from azureml.core import ScriptRunConfig
from azureml.core.environment import Environment
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.authentication import AzureCliAuthentication, ServicePrincipalAuthentication
from azureml.core import Run, Experiment, Workspace, Dataset, Datastore


# For local development, set values in this section
load_dotenv()


def prepareMachines(ws):
    # get environment variables
    ENV_MODEL = json.loads(os.environ.get("ENV_MODEL"))
    ENV_CLUSTER = json.loads(os.environ.get("ENV_CLUSTER"))

    compute_name = ENV_CLUSTER.get("AML_COMPUTE_CLUSTER_NAME")
    compute_min_nodes = int(ENV_CLUSTER.get("AML_COMPUTE_CLUSTER_MIN_NODES"))
    compute_max_nodes = int(ENV_CLUSTER.get("AML_COMPUTE_CLUSTER_MAX_NODES"))
    vm_size = ENV_CLUSTER.get("AML_COMPUTE_CLUSTER_SKU")
    show_progress_training = ENV_MODEL.get('SHOW_PROGRESS_TRAINING') == 'true'

    print(f'show_progress_training={show_progress_training}')
    if compute_name in ws.compute_targets:
        compute_target = ws.compute_targets[compute_name]
        if compute_target and type(compute_target) is AmlCompute:
            print("Found compute target, will use this one: " + compute_name)
    else:
        print("creating new compute target...")
        provisioning_config = AmlCompute.provisioning_configuration(vm_size = vm_size, min_nodes = compute_min_nodes, max_nodes = compute_max_nodes)
        compute_target = ComputeTarget.create(ws, compute_name, provisioning_config)
        compute_target.wait_for_completion(show_output=show_progress_training, min_node_count=None, timeout_in_minutes=20)
    return compute_target


def prepareEnv(ws, env_name):
    # Create environment
    env = Environment(env_name)
    env.docker.enabled = True
    dockerfile = r"""
    FROM mcr.microsoft.com/azureml/base:intelmpi2018.3-ubuntu16.04
    RUN apt-get update && apt-get install -y libgl1-mesa-glx
    RUN echo "Hello from custom container!"
    """
    env.docker.base_image = None
    env.docker.base_dockerfile = dockerfile


    cd = CondaDependencies.create(  pip_packages=[  'azureml-dataset-runtime[pandas,fuse]', 'azureml-defaults', 'azure-ml-api-sdk', 
                                                    'opencv-python', 'onnxmltools', 'onnxruntime', 'tf2onnx'], 
                                    conda_packages = ['scikit-learn==0.22.1', 'tensorflow']
                                    )
    env.python.conda_dependencies = cd

    # Register environment to re-use later
    env.register(workspace = ws)
    return env


def prepareTraining(script_folder, compute_target, environment):
    # get environment variables
    ENV_MODEL = json.loads(os.environ.get("ENV_MODEL"))
    ENV_DATA = json.loads(os.environ.get("ENV_DATA"))

    train_script_name = ENV_MODEL.get('TRAIN_SCRIPT_NAME')
    parameter_name = ENV_MODEL.get('MODEL_NAME')
    parameter_version = float(ENV_MODEL.get('MODEL_VERSION'))
    parameter_epochs = int(ENV_MODEL.get('MODEL_EPOCHS'))
    parameter_batchsize = int(ENV_MODEL.get('MODEL_BATCH_SIZE'))
    parameter_dataset_name = ENV_DATA.get('DATASET_NAME')


    # define arguments and training script
    args = ['--modelname', parameter_name,
            '--modelversion', parameter_version,
            '--epochs', parameter_epochs,
            '--batchsize', parameter_batchsize,
            '--dataset_name', parameter_dataset_name
            ]
    src = ScriptRunConfig(  source_directory=script_folder, 
                            script=train_script_name, 
                            arguments=args,  
                            compute_target=compute_target, 
                            environment=environment)

    return src


def main():
    print('Executing main - 02_Training')

    # get environment variables
    ENV_AZURE = json.loads(os.environ.get("ENV_AZURE"))
    ENV_GENERAL = json.loads(os.environ.get("ENV_GENERAL"))
    ENV_DATA = json.loads(os.environ.get("ENV_DATA"))
    
    resource_group = ENV_AZURE.get("RESOURCE_GROUP") # Azure Resource grouo
    subscription_id = ENV_AZURE.get("SUBSCRIPTION_ID") # Azure Subscription ID
    workspace_name = ENV_AZURE.get("WORKSPACE_NAME") # ML Service Workspace of resource group

    temp_state_directory = ENV_GENERAL.get('TEMP_STATE_DIRECTORY')
    experiment_name = ENV_GENERAL.get("EXPERIMENT_NAME")
    env_name = ENV_GENERAL.get("AML_ENV_NAME")
    root_dir = ENV_GENERAL.get('ROOT_DIR')

    dataset_name = ENV_DATA.get('DATASET_NAME')
    script_folder = os.path.join(root_dir, 'scripts')

    # azure authentication
    cli_auth = AzureCliAuthentication()

    # setup workspace + datastore
    print(f'Connect to workspace {workspace_name} for experiment {experiment_name}')
    ws = Workspace.get(
        name=workspace_name,
        subscription_id=subscription_id,
        resource_group=resource_group,
        auth=cli_auth
    )

    # Prepare!
    print(dataset_name)
    dataset = Dataset.get_by_name(workspace=ws, name=dataset_name)
    compute_target = prepareMachines(ws)
    environment = prepareEnv(ws, env_name)
    src = prepareTraining(script_folder, compute_target, environment)

    ## Start training
    exp = Experiment(workspace=ws, name=experiment_name)
    run = exp.submit(config=src)

    run.wait_for_completion()
    run_details = {k:v for k,v in run.get_details().items() if k not in ['inputDatasets', 'outputDatasets']}
    print(run.get_metrics())
    print(run.get_file_names())
    path_json = os.path.join(temp_state_directory, 'training_run.json')
    with open(path_json, 'w') as training_run_json:
        json.dump(run_details, training_run_json)

    # Finish
    print('Executing main - 02_Training - SUCCES')
    

if __name__ == '__main__':
    main()
