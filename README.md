# mlops-gcp
MLOps Examples for Google Compute Cloud

## wine-quality
## Training
### In notebook

The `wine-quality.ipynb` notebook fetches the wine quality dataset, trains a linear model.
It produces two artifacts:
* Joblib `model.joblib` file that can be used with pre-built AI Platform containers
* [BentoML](https://www.bentoml.ai/) package

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