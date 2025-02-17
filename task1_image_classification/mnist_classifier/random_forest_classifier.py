import numpy as np
from sklearn.ensemble import RandomForestClassifier
from mnist_classifier.mnist_classifier_interface import MnistClassifierInterface

class RandomForestClassifierModel(MnistClassifierInterface):
    def __init__(self, n_estimators=100):
        self.model = RandomForestClassifier(n_estimators=n_estimators)

    def _extract_data(self, dataloader):
        X, y = [], []
        for images, labels in dataloader:
            X.append(images.view(images.shape[0], -1).numpy())
            y.append(labels.numpy())
        return np.vstack(X), np.concatenate(y)

    def train(self, train_loader, test_loader, **kwargs):
        print(f"Training started")
        X_train, y_train = self._extract_data(train_loader)
        X_test, y_test = self._extract_data(test_loader)
        self.model.fit(X_train, y_train)
        print(f"Training finished")
        return self.model.score(X_test, y_test)
    
    def save_weights(self, path="./model.pkl"):
        import joblib
        joblib.dump(self.model, path)
    
    def load_weights(self, path="./model.pkl"):
        import joblib
        self.model = joblib.load(path)

    def predict(self, input):
        X = input.view(input.shape[0], -1).numpy()
        return self.model.predict(X)