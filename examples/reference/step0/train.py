from mnist import MNIST
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load whole dataset
mndata = MNIST('../data/fashion-mnist', gz=True)

train_images, train_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()

# Define the simplest SVC model
model = GaussianNB()
print(model)

# Train model
model.fit(train_images, train_labels)

# Test model
predicted_labels = model.predict(test_images)
print(accuracy_score(list(test_labels), predicted_labels))
