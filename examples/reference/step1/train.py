import json

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pickle
import joblib
import os
import argparse

parser = argparse.ArgumentParser("Train model")
parser.add_argument('--model-dir', dest='model_dir', default=os.getenv("AIP_MODEL_DIR"))

args = parser.parse_args()

# Load whole dataset
with open("../data/fashion-mnist/train.pickle", "rb") as f:
    train_images, train_labels = pickle.load(f)

with open("../data/fashion-mnist/test.pickle", "rb") as f:
    test_images, test_labels = pickle.load(f)

# Define the simplest SVC model
model = GaussianNB()
print(model)

# Train model
model.fit(train_images, train_labels)

# Test model
predicted_labels = model.predict(test_images)
accuracy = accuracy_score(list(test_labels), predicted_labels)
print("Accuracy:", accuracy)
with open("metrics.json", "w") as f:
    json.dump({
        "accuracy": accuracy
    }, f, indent=2)


# Dump model
if args.model_dir.startswith("gs://"):
    with open('model.joblib', 'wb') as f:
        joblib.dump(model, f)
    from google.cloud import storage
    from google.cloud.storage.blob import Blob
    client = storage.Client()
    print(client)
    blob = Blob.from_string(os.path.join(args.model_dir, 'model.joblib'), client)
    print(blob)
    blob.upload_from_filename("model.joblib")
else:
    with open(os.path.join(args.model_dir, 'model.joblib'), 'wb') as f:
        joblib.dump(model, f)
