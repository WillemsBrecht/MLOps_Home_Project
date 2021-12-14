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


def prepareMachines(env, ws):
    ## If machine not yet ready, create !
    # choose a name for your cluster
    compute_name = env.get("AML_COMPUTE_CLUSTER_NAME")
    compute_min_nodes = int(env.get("AML_COMPUTE_CLUSTER_MIN_NODES"))
    compute_max_nodes = int(env.get("AML_COMPUTE_CLUSTER_MAX_NODES"))
    vm_size = env.get("AML_COMPUTE_CLUSTER_SKU")
    show_progress_training = env.get('SHOW_PROGRESS_TRAINING') == 'true'

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


def prepareTraining(env, dataset, script_folder, compute_target, environment):
    # get environment variables
    parameter_name = env.get('MODEL_NAME')
    parameter_version = float(env.get('MODEL_VERSION'))
    parameter_epochs = int(env.get('MODEL_EPOCHS'))
    parameter_batchsize = int(env.get('MODEL_BATCH_SIZE'))
    parameter_dataset_name = env.get('DATASET_NAME')

    train_script_name = env.get('TRAIN_SCRIPT_NAME')

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
    env = os.environ.get("SECRETS_CONTEXT") # Azure Resource grouo
    env = json.loads(env)

    #cli_auth = ServicePrincipalAuthentication(tenant_id=env.get("TENANT_ID"), service_principal_id=env.get("CLIENT_ID"), service_principal_password=env.get("CLIENT_SECRET"))
    cli_auth = AzureCliAuthentication()
    
    # get environment variables 
    workspace_name = env.get("WORKSPACE_NAME")
    experiment_name = env.get("EXPERIMENT_NAME")
    resource_group = env.get("RESOURCE_GROUP")
    subscription_id = env.get("SUBSCRIPTION_ID")
    temp_state_directory = env.get('TEMP_STATE_DIRECTORY')

    env_name = env.get("AML_ENV_NAME")
    dataset_name = env.get('DATASET_NAME')

    root_dir = env.get('ROOT_DIR')
    script_folder = os.path.join(root_dir, 'scripts')

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
    compute_target = prepareMachines(env, ws)
    environment = prepareEnv(ws, env_name)
    src = prepareTraining(env, dataset, script_folder, compute_target, environment)

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
