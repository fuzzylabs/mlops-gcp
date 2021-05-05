# Introduction

This repository is our end-to-end MLOps example for [Google Cloud AI Platform](https://cloud.google.com/ai-platform). It contains everything you would need to get up-and-running with an MLOps stack, and covers a number of different use-cases and machine learning frameworks. Before diving into the details, here's an overview of the three topics we've set out to address in this project:

## Experimenting and training

Enabling teams to collaboratively develop a model with confidence. This includes:

* Data versioning.
* Data labelling.
* Provenance tracking.
* Experiment tracking.
* Collaborative development.

## Production deployment

This not only covers how to deploy a model to production, but also how to re-train a model on demand in the cloud.

## On-going monitoring

Once a model has been deployed we need to monitor it. Why?

* Data can drift over time. When it does, the model might not be producing expected results anymore, and could need re-training.
* Models need to tolerate unexpected inputs, such as values outside of the expected range for a particular feature.
* A served model may go wrong, so we need to monitor error rates, as we would with any software deployed to production.

# Contents

* [MLOps concepts explained](#concepts).
* Examples
** [Wine quality with sklearn](examples/sklearn-wine/README.md).

<a name="#concepts">
# MLOps concepts explained

...

## mlops-gcp
MLOps Examples for Google Compute Cloud


# wine-quality
## Training
### In notebook

The `wine-quality.ipynb` notebook fetches the wine quality dataset, trains a linear model.
It produces two artifacts:
* Joblib `model.joblib` file that can be used with pre-built AI Platform containers
* [BentoML](https://www.bentoml.ai/) package

#### Deployment
Find and push trained model to the Container Registry
To push the BentoML packaged container to Google Cloud Container registry:

```
GCP_PROJECT=fuzzylabs
saved_path=$(bentoml get WineQualityModelService:latest --print-location --quiet)
cd $saved_path
gcloud builds submit --tag gcr.io/$GCP_PROJECT/wine-quality-model
```
### On AI Platform
Build the trainer package and upload it Google Cloud Storage
```
PROJECT_ID=fuzzylabs
BUCKET_NAME="wine-quality-model"
REGION=europe-west2
python setup.py sdist --formats=gztar
gsutil cp dist/trainer-0.1.tar.gz gs://$BUCKET_NAME/
```

Create a custom training task
```
gcloud beta ai custom-jobs create --region=$REGION --display-name=wine-quality-trained --config=training-job.yml
```
The command above will start a training job. When finished, the resulting joblib model artifact will be written to the provided Google Cloud Storage bucket. 


## Deployment
### Create model with pre-built container
```
gcloud beta ai models upload \
  --region=europe-west2 \
  --display-name=wine-quality \
  --artifact-uri=gs://wine-quality-model/trained/model/ \
  --container-image-uri=europe-docker.pkg.dev/cloud-aiplatform/prediction/sklearn-cpu.0-23:latest
MODEL_ID=${returned model ID}
```

### Create model with BentoML container
Create model on AI Platform (Unified) with this container
```
gcloud beta ai models upload \ 
  --region=europe-west2 \
  --display-name=wine-quality-bentoml \
  --container-image-uri=gcr.io/fuzzylabs/wine-quality-model \
  --container-predict-route="/predict" \
  --container-health-route="/healthz" \
  --container-ports=5000
MODEL_ID=${returned model ID}
```

### Deploy endpoint
Create an endpoint and deploy the model 
```
gcloud ai endpoints create --display-name wine-quality --region europe-west2 # Returns endpoint ID
ENDPOINT_ID=${returned endpoint ID}
gcloud ai endpoints deploy-model $ENDPOINT_ID --region=europe-west2 --model=$MODEL_ID --display-name="wine-quality"
```

Query endpoint
```
curl \
-X POST \
-H "Authorization: Bearer $(gcloud auth print-access-token)" \
-H "Content-Type: application/json" \
https://europe-west2-aiplatform.googleapis.com/v1alpha1/projects/${GCP_PROJECT}/locations/europe-west2/endpoints/${ENDPOINT_ID}:predict \
-d '@test-input.json'
```
