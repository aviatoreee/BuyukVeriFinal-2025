# src/classifiers/base_classifier.py
from abc import ABC, abstractmethod

class BaseClassifier(ABC):
    """
    Tüm sınıflandırıcıların türetilmesi gereken temel soyut sınıf.
    Bu yapı, kodun geri kalanının modelin iç yapısını bilmeden
    çalışabilmesini (Polimorfizm) sağlar.
    """
    
    @abstractmethod
    def train(self, X_train, y_train):
        """Modeli eğitir."""
        pass

    @abstractmethod
    def predict(self, X_test):
        """Test verisi üzerinde tahmin yapar."""
        pass
    
    @abstractmethod
    def get_name(self):
        """Modelin ekranda görünecek adını döner."""
        pass