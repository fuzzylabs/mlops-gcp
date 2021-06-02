from google.cloud.aiplatform import CustomJob

# Upload dataset


#
CustomJob.from_local_script(
    display_name="Fashion MNIST Naive Bayes",
    script_path="train.py",
    container_uri="europe-docker.pkg.dev/cloud-aiplatform/training/scikit-learn-cpu.0-23:latest",
    requirements=[
        "scikit-learn",
        "python-mnist",
        "dvc[gs]"
    ],
    replica_count=1,
    project="fuzzylabs",
    location="europe-west4",
    staging_bucket="gs://fashion-mnist-model/"
).run()
