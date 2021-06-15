import os
import pickle
import joblib
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.requests import Request
from google.cloud import storage
from google.cloud.storage.blob import Blob

storage_client = storage.Client()

storage_uri = os.environ.get("AIP_STORAGE_URI")
Blob.from_string(os.path.join(storage_uri, "model.joblib"), storage_client).download_to_file("model.joblib")
model = joblib.load("model.joblib")
Blob.from_string(os.path.join(storage_uri, "monitoring.pickle"), storage_client).download_to_file("monitoring.pickle")
with open("monitoring.pickle", "rb") as f:
    monitoring = pickle.load(f)


async def health(request: Request):
    return PlainTextResponse("OK")


async def infer(request: Request):
    data = await request.json()
    instances = data["instances"]

    monitoring.add_data(instances)
    predictions = model.predict(instances)
    return JSONResponse({
        "predictions": predictions
    })


routes = [
    Route('/health', endpoint=health),
    Route('/infer', endpoint=infer, methods=["POST"])
]

app = Starlette(debug=True, routes=routes)
