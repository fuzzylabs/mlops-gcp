import json
import os.path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import argparse
import pickle
import joblib
from sklearn.decomposition import PCA
from scipy.stats import kstest


class PCAMonitoring:
    def __init__(self, train_data):
        self.pca = PCA(n_components=5)
        self.train_data_pca = self.pca.fit_transform(train_data)
        self.data = []
        self.monitoring_trigger_number = 100

    def get_distances(self, data):
        """
        Calculates Kolmogorov-Smirnov distance and p-values for each PC
        :param data:
        :return: Array of tuples (KS distance, p-value)
        """
        data_pca = self.pca.transform(data)
        return [tuple(kstest(self.train_data_pca[:, i], data_pca[:, i])) for i in range(self.pca.n_components)]

    def add_data(self, data):
        """

        :param data: 2D array of input data of shape (n_samples, dimensions)
        :return:
        """
        self.data += data
        if len(self.data) >= self.monitoring_trigger_number:
            distances = self.get_distances(self.data)
            drifted_pcs = [i for i in range(self.pca.n_components) if distances[i][1] < 0.05]
            level = "INFO" if len(drifted_pcs) == 0 else "WARNING"
            print({
                "severity": level,
                "kstest": distances,
                "drifted_pcs": drifted_pcs,
            })
            self.data = []


def wrap_open(path: str, mode: str = "r"):
    if path.startswith("gs://"):
        from google.cloud import storage
        from google.cloud.storage.blob import Blob

        client = storage.Client()

        return Blob.from_string(path, client).open(mode=mode)
    else:
        return open(path, mode=mode)


def train_model(train_dataset, n_neighbors):
    (train_images, train_labels) = train_dataset

    # Define the simplest SVC model
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    print(model)

    # Train model
    model.fit(train_images, train_labels)
    return model


def test_model(model, test_dataset):
    (test_images, test_labels) = test_dataset

    predicted_labels = model.predict(test_images)
    accuracy = accuracy_score(list(test_labels), predicted_labels)
    print("Accuracy:", accuracy)
    return {
        "accuracy": accuracy
    }


def load_datasets(train_set_path, test_set_path):
    with wrap_open(train_set_path, "rb") as f:
        train_set = pickle.load(f)
    with wrap_open(test_set_path, "rb") as f:
        test_set = pickle.load(f)

    return train_set, test_set


def save_results(model, monitoring_model, metrics, model_output_dir, metrics_output_path):
    with wrap_open(os.path.join(model_output_dir, 'model.joblib'), 'wb') as f:
        joblib.dump(model, f)

    with wrap_open(metrics_output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    with wrap_open(os.path.join(model_output_dir, 'monitoring.pickle'), 'wb') as f:
        pickle.dump(monitoring_model, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a model")
    parser.add_argument("train_set_path")
    parser.add_argument("test_set_path")
    parser.add_argument("--n-neigbours", dest="n_neighbours", default=1, type=int)
    parser.add_argument('--model-dir', dest='model_dir', default=os.getenv("AIP_MODEL_DIR"))
    parser.add_argument("--model-metrics-path", dest="model_metrics_path", default="../metrics.json")

    args = parser.parse_args()

    train_set, test_set = load_datasets(args.train_set_path, args.test_set_path)
    model = train_model(train_set, args.n_neighbours)
    metrics = test_model(model, test_set)
    monitoring_model = PCAMonitoring(train_set[0])
    save_results(model, monitoring_model, metrics, args.model_dir, args.model_metrics_path)
