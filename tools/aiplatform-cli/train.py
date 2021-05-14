from project import ProjectDefinition
from google.cloud import aiplatform
from setuptools import sandbox
from google.cloud import storage


def train():
    project = ProjectDefinition.from_file("./project.yaml")
    print(project)
    build_dist()
    trainer_url = upload_dist(project)
    print(trainer_url)
    create_custom_job(project)


def build_dist():
    sandbox.run_setup("setup.py", ["sdist", "--formats=gztar"])


def upload_dist(project: ProjectDefinition, dist_path: str = "./dist/trainer-0.1.tar.gz") -> str:
    # TODO infer the distribution name/path from setup.py
    storage_client = storage.Client()
    bucket = storage_client.bucket(project.gcp_bucket_name)
    blob = bucket.blob("trainer.tar.gz")
    blob.upload_from_filename(dist_path)

    return blob.public_url

def create_custom_job(
        project: ProjectDefinition,
        api_endpoint: str = "europe-west2-aiplatform.googleapis.com",
):
    # TODO infer endpoint from the config
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.JobServiceClient(client_options=client_options)
    custom_job = {
        "display_name": project.name,
        "job_spec": {
            "worker_pool_specs": [
                {
                    "machine_spec": {
                        "machine_type": project.machine_type,
                    },
                    "replica_count": 1,
                    "python_package_spec": {
                        "package_uris": [
                            f"gs://{project.gcp_bucket_name}/trainer.tar.gz"
                        ],
                        "executor_image_uri": project.executor_image,
                        "python_module": project.python_module
                    }
                }
            ],
            "base_output_directory": {
                "output_uri_prefix":  f"gs://{project.gcp_bucket_name}/trained"
            }
        },
    }
    parent = f"projects/{project.gcp_project}/locations/{project.gcp_region}"
    response = client.create_custom_job(parent=parent, custom_job=custom_job)
    print("response:", response)
