import json
from google.cloud.aiplatform import CustomJob, Model
from google.cloud.secretmanager import SecretManagerServiceClient
from google.cloud import storage
import dvc.api
import yaml
import uuid
import os.path
from sacred import Experiment
from sacred.observers import MongoObserver


def access_secret_version(secret_id, version_id="latest"):
    # Create the Secret Manager client.
    client = SecretManagerServiceClient()

    # Build the resource name of the secret version.
    name = f"projects/fuzzylabs/secrets/{secret_id}/versions/{version_id}"

    # Access the secret version.
    response = client.access_secret_version(name=name)

    # Return the decoded payload.
    return response.payload.data.decode('UTF-8')


def get_mongo_connection_string():
    return access_secret_version("sacred-mongodb-connection-string")


ex = Experiment("fashion-mnist-model-training")
ex.observers.append(MongoObserver(
    url=get_mongo_connection_string(),
))


@ex.config
def config():
    print("Preparing job to run on Vertex AI")
    params = yaml.safe_load(open("../params.yaml"))["train"]

    # Get dataset links
    train_uri = dvc.api.get_url("data/fashion-mnist/train.pickle")
    test_uri = dvc.api.get_url("data/fashion-mnist/test.pickle")

    # Define output bucket
    output_dir = f"gs://fashion-mnist-model/custom_jobs/{uuid.uuid4()}"
    # output_dir = "gs://fashion-mnist-model/custom_jobs/d1f3d7c0-0f02-4ad9-bd52-495edc1331ca"

    model_gs_link = os.path.join(output_dir, "model.joblib")
    metrics_gs_link = os.path.join(output_dir, "metrics.json")


@ex.automain
def main(
        _run,
        params,
        train_uri: str,
        test_uri: str,
        output_dir: str,
        metrics_gs_link: str,
):
    print("Running the job")
    # Run job
    CustomJob.from_local_script(
        display_name="Fashion MNIST Naive Bayes",
        script_path="train.py",
        container_uri="europe-docker.pkg.dev/cloud-aiplatform/training/scikit-learn-cpu.0-23:latest",
        requirements=[
            "scikit-learn==0.23.1",
            "google-cloud-storage==1.38.0",
            "dill==0.3.4",
            "scipy==1.6.3",
        ],
        args=[
            "--model-dir", output_dir,
            "--model-metrics-path", metrics_gs_link,
            "--n-neigbours", str(params["n_neighbours"]),
            train_uri,
            test_uri
        ],
        replica_count=1,
        project="fuzzylabs",
        location="europe-west4",
        staging_bucket="gs://fashion-mnist-model/"
    ).run()

    # Get results back
    print("Fetching the results")  # TODO see options for linking from gs, instead of downloading locally

    client = storage.Client()
    metrics = json.loads(storage.Blob.from_string(metrics_gs_link, client).download_as_bytes())

    for metric in metrics:
        _run.log_scalar(metric, metrics[metric])

    # Create model on Vertex
    print("Creating model")
    model = Model.upload(
        display_name="fashion-mnist-model",
        project="fuzzylabs",
        location="europe-west4",
        serving_container_image_uri="gcr.io/fuzzylabs/fashion-mnist-prediction-server",
        serving_container_predict_route="/infer",
        serving_container_health_route="/health",
        serving_container_ports=[8000],
        artifact_uri=output_dir,
    )

    with open("../model/vertex_model.json", "w") as f:
        json.dump({
            "model_name": model.resource_name,
        }, f)
