import json

from dvc.repo import Repo
from dvc.path_info import PathInfo
from funcy import reraise
from dvc.exceptions import OutputNotFoundError, PathMissingError

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle
import joblib
import os
import yaml
from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment("fashion-mnist-model-training")
ex.observers.append(MongoObserver(
    url="mongodb://localhost:27017"
))

def dvc_get_md5(path, repo=None, rev=None, remote=None):
    with Repo.open(repo, rev=rev, subrepos=True, uninitialized=True) as _repo:
        path_info = PathInfo(_repo.root_dir) / path
        with reraise(FileNotFoundError, PathMissingError(path, repo)):
            metadata = _repo.repo_fs.metadata(path_info)

        if not metadata.is_dvc:
            raise OutputNotFoundError(path, repo)

        md5 = metadata.repo.dvcfs.info(path_info)["md5"]
        return md5

@ex.config
def config():
    n_neighbors = yaml.safe_load(open("../params.yaml"))["train"]["n_neighbours"]
    model_dir = os.getenv("AIP_MODEL_DIR", "../model/")
    train_md5 = dvc_get_md5("data/fashion-mnist/train.pickle")
    test_md5 = dvc_get_md5("data/fashion-mnist/test.pickle")

@ex.automain
def main(n_neighbors, model_dir, _run):
    # Load whole dataset
    with open("../data/fashion-mnist/train.pickle", "rb") as f:
        train_images, train_labels = pickle.load(f)

    with open("../data/fashion-mnist/test.pickle", "rb") as f:
        test_images, test_labels = pickle.load(f)

    # Define the simplest SVC model
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    print(model)

    # Train model
    model.fit(train_images, train_labels)

    # Test model
    predicted_labels = model.predict(test_images)
    accuracy = accuracy_score(list(test_labels), predicted_labels)
    _run.log_scalar("accuracy", accuracy)
    print("Accuracy:", accuracy)
    with open("../metrics.json", "w") as f:
        json.dump({
            "accuracy": accuracy
        }, f, indent=2)

    # Dump model
    if model_dir.startswith("gs://"):
        with open('model.joblib', 'wb') as f:
            joblib.dump(model, f)
        from google.cloud import storage
        from google.cloud.storage.blob import Blob
        client = storage.Client()
        print(client)
        blob = Blob.from_string(os.path.join(model_dir, 'model.joblib'), client)
        print(blob)
        blob.upload_from_filename("model.joblib")
    else:
        with open(os.path.join(model_dir, 'model.joblib'), 'wb') as f:
            joblib.dump(model, f)

    return dict(
        model_md5=dvc_get_md5("model/model.joblib")
    )
