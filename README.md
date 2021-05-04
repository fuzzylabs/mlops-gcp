# mlops-gcp
MLOps Examples for Google Compute Cloud

## wine-quality
### Notebook

The notebook fetches the wine quality dataset, trains a linear model and packages the trained model with [BentoML](https://www.bentoml.ai/).

### Deployment
Find and push trained model to the Container Registry
```
GCP_PROJECT=fuzzylabs
saved_path=$(bentoml get WineQualityModelService:latest --print-location --quiet)
cd $saved_path
gcloud builds submit --tag gcr.io/$GCP_PROJECT/wine-quality-model
```

Create model on AI Platform (Unified) with this container
```
gcloud beta ai models upload --region=europe-west2 --display-name=wine-quality --container-image-uri=gcr.io/fuzzylabs/wine-quality-model --container-predict-route="/predict" --container-health-route="/healthz" --container-ports=5000
MODEL_ID=${returned model ID}
```

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
https://us-central1-aiplatform.googleapis.com/v1alpha1/projects/${GCP_PROJECT}/locations/europe-west2/endpoints/${ENDPOINT_ID}:predict \
-d '@test-input.json'
```