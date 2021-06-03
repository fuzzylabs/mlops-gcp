from google.cloud.aiplatform import CustomJob
from google.cloud import storage
import dvc.api
import yaml
import uuid
import os.path

print("Preparing job to run on Vertex AI")
params = yaml.safe_load(open("../params.yaml"))["train"]

# Get dataset links
train_uri = dvc.api.get_url("data/fashion-mnist/train.pickle")
test_uri = dvc.api.get_url("data/fashion-mnist/test.pickle")

# Define output bucket
output_dir = f"gs://fashion-mnist-model/custom_jobs/{uuid.uuid4()}"
model_gs_link = os.path.join(output_dir, "model.joblib")
metrics_gs_link = os.path.join(output_dir, "metrics.json")

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

# Get results back
print("Fetching the results") # TODO see options for linking from gs, instead of downloading locally

client = storage.Client()
with open("../model/model.joblib", "wb") as f:
    storage.Blob.from_string(model_gs_link, client).download_to_file(f)
with open("../metrics.json", "wb") as f:
    storage.Blob.from_string(metrics_gs_link, client).download_to_file(f)
