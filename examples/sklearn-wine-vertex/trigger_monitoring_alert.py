import random
import time
import argparse
from google.cloud.aiplatform_v1beta1.types.prediction_service import PredictRequest
from google.cloud.aiplatform_v1beta1.services.prediction_service import PredictionServiceClient
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value

def random_uid():
    digits = [str(i) for i in range(10)] + ["A", "B", "C", "D", "E", "F"]
    return "".join(random.choices(digits, k=32))

PREDICT_API_ENDPOINT = "us-central1-prediction-aiplatform.googleapis.com"

def send_predict_request(endpoint, input):
    client_options = {"api_endpoint": PREDICT_API_ENDPOINT}
    client = PredictionServiceClient(client_options=client_options)
    params = {}
    params = json_format.ParseDict(params, Value())
    request = PredictRequest(endpoint=endpoint, parameters=params)
    inputs = [json_format.ParseDict(input, Value())]
    request.instances.extend(inputs)
    response = client.predict(request)
    return response

def monitoring_test(endpoint, count, sleep):
    for i in range(0, 10000000):
        input = {
            "alcohol": str(random.randint(1000, 2000)),
            "chlorides": "0.042",
            "citric_acid": "0.31",
            "density": "0.99",
            "fixed_acidity": "6.8",
            "free_sulfur_dioxide": "33.0",
            "pH": str(random.randint(-500, -400)),
            "residual_sugar": "4.9",
            "sulphates": "0.47",
            "total_sulfur_dioxide": "132",
            "volatile_acidity": "0.26"
        }
        print(f"Sending prediction {i}")
        try:
            result = send_predict_request(endpoint, input)
            print(result)
        except Exception as e:
            print("prediction request failed")
            print(e)
        time.sleep(sleep)
    print("Test Completed.")


parser = argparse.ArgumentParser(description="Generate wrong prediction requests to trigger monitoring alert")
parser.add_argument("endpoint")

args = parser.parse_args()

test_time = 1200
tests_per_sec = 1
sleep_time = 1 / tests_per_sec
iterations = test_time * tests_per_sec
monitoring_test(args.endpoint, iterations, sleep_time)