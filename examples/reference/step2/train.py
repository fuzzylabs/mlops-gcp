import os.path

from mnist import MNIST
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import dvc.api


def train_model(data):
    train_images, train_labels = data.load_training()
    test_images, test_labels = data.load_testing()
    print(set(train_labels))
    # Define the simplest SVC model
    model = GaussianNB()
    print(model)

    # Train model
    model.fit(train_images, train_labels)

    # Test model
    predicted_labels = model.predict(test_images)
    print(accuracy_score(list(test_labels), predicted_labels))
    return model


def load_dataset():
    # Load whole dataset
    mndata = MNIST('../data/fashion-mnist', gz=True)
    return mndata


def pull_file(path, root_dir="../", git_prefix="examples/reference/"):
    bytes = dvc.api.read(os.path.join(git_prefix, path), mode="rb", repo="https://github.com/fuzzylabs/mlops-gcp", rev="reference-example")
    print(len(bytes))
    with open(os.path.join(root_dir, path), "wb") as out_file:
        out_file.write(bytes)


def load_from_remote(root_dir="../"):
    pull_file("data/fashion-mnist/t10k-images-idx3-ubyte.gz", root_dir)
    pull_file("data/fashion-mnist/t10k-labels-idx1-ubyte.gz", root_dir)
    pull_file("data/fashion-mnist/train-images-idx3-ubyte.gz", root_dir)
    pull_file("data/fashion-mnist/train-labels-idx1-ubyte.gz", root_dir)


load_from_remote()
dataset = load_dataset()
model = train_model(dataset)
