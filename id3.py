# src/classifiers/id3.py
from sklearn.tree import DecisionTreeClassifier
from .base_classifier import BaseClassifier

class ID3Classifier(BaseClassifier):
    def __init__(self):
        # ID3 algoritması 'Entropy' (Bilgi Kazancı) metriğini kullanır.
        self.model = DecisionTreeClassifier(criterion='entropy', random_state=42)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
        
    def get_name(self):
        return "ID3 (Entropy)"