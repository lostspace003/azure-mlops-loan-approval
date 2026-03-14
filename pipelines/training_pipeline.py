# pipelines/training_pipeline.py
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import get_ml_client, get_azure_config
from azure.ai.ml import Input, Output, command, dsl
from azure.ai.ml.entities import Environment

ml_client = get_ml_client()
cfg = get_azure_config()

# Create custom environment
env = Environment(
    name="loan-approval-env",
    conda_file="./environments/conda.yml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
)
env = ml_client.environments.create_or_update(env)

# Define components
data_prep_component = command(
    name="data_prep",
    display_name="Data Preparation",
    inputs={
        'input_data': Input(type='uri_file'),
        'test_size': Input(type='number', default=0.2),
    },
    outputs={
        'train_data': Output(type='uri_folder'),
        'test_data': Output(type='uri_folder'),
    },
    code='./src/data_prep/',
    command='python data_prep.py '
            '--input_data ${{inputs.input_data}} '
            '--train_data ${{outputs.train_data}} '
            '--test_data ${{outputs.test_data}} '
            '--test_size ${{inputs.test_size}}',
    environment=f'{env.name}:{env.version}',
)

train_component = command(
    name="train",
    display_name="Model Training",
    inputs={
        'train_data': Input(type='uri_folder'),
        'n_estimators': Input(type='integer', default=100),
        'learning_rate': Input(type='number', default=0.1),
        'max_depth': Input(type='integer', default=5),
    },
    outputs={
        'model_output': Output(type='uri_folder'),
    },
    code='./src/train/',
    command='python train.py '
            '--train_data ${{inputs.train_data}} '
            '--model_output ${{outputs.model_output}} '
            '--n_estimators ${{inputs.n_estimators}} '
            '--learning_rate ${{inputs.learning_rate}} '
            '--max_depth ${{inputs.max_depth}}',
    environment=f'{env.name}:{env.version}',
)

evaluate_component = command(
    name="evaluate",
    display_name="Model Evaluation",
    inputs={
        'model_input': Input(type='uri_folder'),
        'test_data': Input(type='uri_folder'),
    },
    outputs={
        'evaluation_output': Output(type='uri_folder'),
    },
    code='./src/evaluate/',
    command='python evaluate.py '
            '--model_input ${{inputs.model_input}} '
            '--test_data ${{inputs.test_data}} '
            '--evaluation_output ${{outputs.evaluation_output}}',
    environment=f'{env.name}:{env.version}',
)

register_component = command(
    name="register",
    display_name="Model Registration",
    inputs={
        'model_input': Input(type='uri_folder'),
        'evaluation_output': Input(type='uri_folder'),
    },
    code='./src/register/',
    command='python register.py '
            '--model_input ${{inputs.model_input}} '
            '--evaluation_output ${{inputs.evaluation_output}}',
    environment=f'{env.name}:{env.version}',
)


# Build pipeline
@dsl.pipeline(
    name='loan-approval-training-pipeline',
    description='E2E training pipeline for loan approval model',
    compute='serverless',
)
def loan_approval_pipeline(input_data, test_size=0.2):
    prep = data_prep_component(input_data=input_data, test_size=test_size)
    train = train_component(train_data=prep.outputs.train_data)
    evaluate = evaluate_component(
        model_input=train.outputs.model_output,
        test_data=prep.outputs.test_data,
    )
    register = register_component(
        model_input=train.outputs.model_output,
        evaluation_output=evaluate.outputs.evaluation_output,
    )
    return {
        'model': train.outputs.model_output,
        'metrics': evaluate.outputs.evaluation_output,
    }


# Submit pipeline
data_asset = ml_client.data.get('loan-approval-dataset', version='1')
pipeline_job = loan_approval_pipeline(
    input_data=Input(type='uri_file', path=data_asset.path)
)
submitted = ml_client.jobs.create_or_update(
    pipeline_job, experiment_name='loan-approval-experiment'
)
print(f"Pipeline submitted: {submitted.name}")
print(f"Studio URL: {submitted.studio_url}")

# Wait for completion
ml_client.jobs.stream(submitted.name)
