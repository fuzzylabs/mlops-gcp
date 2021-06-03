## Step 0 -- locally train a model
## Step 1 -- locally train with DVC pipeline
* Copy `dvc-step1.yaml` to `dvc.yaml`
* `dvc repro` to run training
## Step 2 -- train on Vertex AI
* Copy `dvc-step2.yaml` to `dvc.yaml`
* `dvc repro` to run training
## Step 3 -- train and deploy on Vertex AI
* Copy `dvc-step3.yaml` to `dvc.yaml`
* `dvc repro` to run training
* `cd step3 && python3 deploy-vertex.py` to deploy