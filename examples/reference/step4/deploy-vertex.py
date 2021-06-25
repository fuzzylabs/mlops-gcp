import json
from google.cloud.aiplatform import Model, Endpoint

endpoint_id = "6923967767733338112"

endpoint = Endpoint(endpoint_name=endpoint_id, project="fuzzylabs", location="europe-west4")

endpoint.undeploy_all()

with open("../model/vertex_model.json") as f:
    model_dict = json.load(f)

model = Model(model_dict["model_name"])

print("Deploying model")

endpoint.deploy(
    model=model,
    traffic_percentage=100,
    machine_type="n1-standard-2"
)
# model.deploy(
#     machine_type="n1-standard-2",
# )
