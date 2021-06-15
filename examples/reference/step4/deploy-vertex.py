import json
from google.cloud.aiplatform import Model

with open("../model/vertex_model.json") as f:
    model_dict = json.load(f)

model = Model(model_dict["model_name"])

print("Deploying model")
model.deploy(
    machine_type="n1-standard-2",
)
