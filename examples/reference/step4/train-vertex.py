import json
from google.cloud.aiplatform import CustomJob, Model
from google.cloud import storage
import dvc.api
import yaml
import uuid
import os.path
from sacred import Experiment
from sacred.observers import GoogleCloudStorageObserver

ex = Experiment("fashion-mnist-train-vertex")
ex.observers.append(GoogleCloudStorageObserver(
    bucket="fashion-mnist-model",
    basedir="sacred"
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
            "scikit-learn",
            "google-cloud-storage==1.38.0",
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

    print("")

    # Get results back
    print("Fetching the results") # TODO see options for linking from gs, instead of downloading locally

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
