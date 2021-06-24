import time
import dill
import random
from google.cloud.aiplatform import Endpoint

with open("../data/fashion-mnist/test.pickle", "rb") as f:
    X, y = dill.load(f)

endpoint = Endpoint(endpoint_name="4669916154234404864", project="fuzzylabs", location="europe-west4")

for i in range(100):
    instance = random.choice(X)
    prediction = endpoint.predict(instances=[instance])
    print(i, prediction)
    time.sleep(1)

