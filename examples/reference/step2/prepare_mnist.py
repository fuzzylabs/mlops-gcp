from mnist import MNIST
import pickle


mndata = MNIST('../data/fashion-mnist', gz=True)

train_images, train_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()

with open("../data/fashion-mnist/train.pickle", "wb") as f:
    pickle.dump((train_images, train_labels), f)

with open("../data/fashion-mnist/test.pickle", "wb") as f:
    pickle.dump((test_images, test_labels), f)
