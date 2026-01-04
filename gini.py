# src/classifiers/gini.py
from sklearn.tree import DecisionTreeClassifier
from .base_classifier import BaseClassifier

class GiniClassifier(BaseClassifier):
    def __init__(self):
        # Standart CART algoritması 'Gini Impurity' metriğini kullanır.
        self.model = DecisionTreeClassifier(criterion='gini', random_state=42)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def get_name(self):
        return "Gini (CART)"