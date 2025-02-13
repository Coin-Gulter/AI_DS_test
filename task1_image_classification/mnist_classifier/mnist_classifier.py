from mnist_classifier.random_forest_classifier import RandomForestClassifierModel
from mnist_classifier.neural_network_classifier import NeuralNetworkClassifier
from mnist_classifier.cnn_classifier import CNNClassifier


class MnistClassifier():
    def __init__(self, algorithm='cnn', **kwargs):
        if algorithm == 'rf':
            self.model = RandomForestClassifierModel(**kwargs)
        elif algorithm == 'nn':
            self.model = NeuralNetworkClassifier(**kwargs)
        elif algorithm == 'cnn':
            self.model = CNNClassifier(**kwargs)
        else:
            raise ValueError(f"Unknown algorithm type: {algorithm}")

    def train(self, train_loader, test_loader, **kwargs):
        return self.model.train(train_loader, test_loader, **kwargs)

    def predict(self, input):
        return self.model.predict(input)