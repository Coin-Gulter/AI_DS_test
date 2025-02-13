import abc

class MnistClassifierInterface(abc.ABC):
    @abc.abstractmethod
    def train(self, train_loader, test_loader):
        pass

    @abc.abstractmethod
    def predict(self, input):
        pass