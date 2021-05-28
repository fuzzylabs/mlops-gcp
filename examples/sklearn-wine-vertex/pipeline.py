import kfp
from kfp.v2 import compiler
from kfp.v2.google.client import AIPlatformClient
from google.cloud import aiplatform
from google_cloud_pipeline_components import aiplatform as gcc_aip

project_id = "fuzzylabs"
region = "europe-west4"
pipeline_root_path = "gs://wine-quality-model"
gcs_source = "gs://wine-quality-model/wine-quality.csv"

# Pipeline definition using Google's prebuilt components
@kfp.dsl.pipeline(
    name="automl-wine-quality",
    pipeline_root=pipeline_root_path)
def pipeline(project_id: str):
    ds_op = gcc_aip.TabularDatasetCreateOp(
        project=project_id,
        display_name="wine-quality",
        gcs_source=gcs_source,
    )

    training_job_run_op = gcc_aip.AutoMLTabularTrainingJobRunOp(
        project=project_id,
        display_name="wine-quality-automl",
        optimization_prediction_type="regression",
        dataset=ds_op.outputs["dataset"],
        model_display_name="wine-quality-model",
        target_column="quality",
        training_fraction_split=0.6,
        validation_fraction_split=0.2,
        test_fraction_split=0.2,
        budget_milli_node_hours=1000,
    )

    endpoint_op = gcc_aip.ModelDeployOp(
        project=project_id, model=training_job_run_op.outputs["model"]
    )

# Compile the pipeline
compiler.Compiler().compile(pipeline_func=pipeline,
        package_path='wine-quality-pipeline.json')

# Launch the compiled pipeline on Vertex AI
api_client = AIPlatformClient(project_id=project_id, region=region)

response = api_client.create_run_from_job_spec(
    'wine-quality-pipeline.json',
    pipeline_root=pipeline_root_path,
    parameter_values={
        'project_id': project_id
    })
