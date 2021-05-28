# Wine Quality modelling with Vertex AI 

This example project:
* Trains a wine quality model locally with sklearn and uploads experiment logging to Vertex AI hosted Tensorboard
* Trains a wine quality model with AutoML on Vertex AI pipelines

## Setup 
Prerequisites:
* Python 3 and venv
* gcloud CLI tools

Install the depoendencies 
```
python3 -m venv env/
source env/bin/activate
pip install -r requirements.txt
```

## Local training and experiment tracking

Create Vertex AI tensorboard
```
gcloud beta ai tensorboards create --display-name DISPLAY_NAME \
  --project PROJECT_NAME \
  --region REGION 
```

Run the `train.ipynb` notebook. The notebook downloads the dataset, trains the model, and logs the results in Tensorboard.

You can view runs locally with `tensorboard --logdir runs/`

To upload a run to Vertex AI:
```
!tb-gcp-uploader --tensorboard_resource_name projects/PROJECT_NAME/locations/REGION/tensorboards/TENSORBOARD_NAME \
  --experiment_name="EXPERIMENT_NAME" \
  --logdir=LOG_DIR --one_shot
```

N.B. we could not confirm that, as GCP returns 500 error, when we were trying to upload

## AutoML and pipelines
To run a training job in Vertex AI pipelines, substitute project ID, region, pipeline root directory, and dataset URI (gcs_source) in `pipeline.py` and run the script