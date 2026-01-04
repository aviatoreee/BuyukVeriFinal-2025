# src/classifiers/factory.py
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix
from .id3 import ID3Classifier
from .gini import GiniClassifier
from .twoing import TwoingClassifier

class ClassifierFactory:
    def __init__(self):
        # Modelleri önbellekte (memory) tutmak için
        # Program kapandığında silinirler.
        self._instances = {
            'id3': ID3Classifier(),
            'gini': GiniClassifier(),
            'twoing': TwoingClassifier()
        }

    def get_classifier(self, method_name):
        """İlgili yöntemin sınıf örneğini döner."""
        if method_name not in self._instances:
            raise ValueError(f"Geçersiz yöntem adı: {method_name}")
        return self._instances[method_name]

    def evaluate_model(self, method_name, X_test, y_test):
        """
        Modeli test eder ve dashboard için gerekli metrikleri hesaplayıp döner.
        """
        model_instance = self.get_classifier(method_name)
        
        # Tahmin işlemi
        y_pred = model_instance.predict(X_test)
        
        # Metriklerin hesaplanması
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            # average='weighted': Dengesiz veri setleri için ağırlıklı ortalama alır
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        return metrics