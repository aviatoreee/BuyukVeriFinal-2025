import numpy as np
from collections import Counter
from .base_classifier import BaseClassifier

class Node:
    """Ağaçtaki her bir düğümü temsil eder."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # Hangi özellikten bölündü (indeks)
        self.threshold = threshold  # Eşik değeri (örn: > 0.5)
        self.left = left            # Sol çocuk (Node)
        self.right = right          # Sağ çocuk (Node)
        self.value = value          # Eğer yaprak ise, sınıf değeri (0 veya 1)

class CustomTwoingTree:

    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        """Modeli eğitir (Ağacı inşa eder)."""
        self.n_classes_ = len(np.unique(y))
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Durma Kriterleri:
        # 1. Maksimum derinliğe ulaşıldı
        # 2. Sınıf ayrımı bitti (pure node)
        # 3. Örnek sayısı çok az
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # En iyi bölünmeyi bul (Twoing Kriterine göre)
        feat_idx, threshold = self._best_split(X, y, n_features)

        # Eğer bölecek mantıklı bir yer bulamadıysa yaprak yap
        if feat_idx is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Böl ve Rekürsif (Özyinelemeli) olarak devam et
        left_idxs = X[:, feat_idx] < threshold
        right_idxs = ~left_idxs
        
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        
        return Node(feature=feat_idx, threshold=threshold, left=left, right=right)

    def _best_split(self, X, y, n_features):
        best_score = -1
        split_idx, split_thresh = None, None

        # Rastgelelik yerine tüm özellikleri gezebiliriz veya belirli sayıda özellik seçebiliriz.
        # Performans için özellikleri sırayla geziyoruz.
        for feat_idx in range(n_features):
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            
            # Çok fazla threshold varsa hepsini deneme, örnekleme yap (Hız optimizasyonu)
            if len(thresholds) > 100:
                thresholds = np.percentile(thresholds, np.linspace(0, 100, 50))

            for thr in thresholds:
                # Bölünmeyi gerçekleştir
                left_mask = X_column < thr
                
                # Boş bölünme kontrolü
                if not np.any(left_mask) or np.all(left_mask):
                    continue
                
                # Twoing Skorunu Hesapla
                y_left = y[left_mask]
                y_right = y[~left_mask]
                score = self._twoing_criteria(y_left, y_right)

                if score > best_score:
                    best_score = score
                    split_idx = feat_idx
                    split_thresh = thr

        return split_idx, split_thresh

    def _twoing_criteria(self, y_left, y_right):
        """
        Twoing Rule Formülü:
        (Pl * Pr / 4) * (Sum(|P(c|L) - P(c|R)|))^2
        """
        n_left = len(y_left)
        n_right = len(y_right)
        n_total = n_left + n_right
        
        p_L = n_left / n_total
        p_R = n_right / n_total
        
        # Sınıf dağılımları
        # Not: Sınıfların 0, 1, 2... şeklinde gittiğini varsayıyoruz
        # Performans için bincount kullanıyoruz
        counts_left = np.bincount(y_left, minlength=self.n_classes_)
        counts_right = np.bincount(y_right, minlength=self.n_classes_)
        
        probs_left = counts_left / n_left
        probs_right = counts_right / n_right
        
        # Formülün uygulanması
        # Sum of absolute differences
        diff_sum = np.sum(np.abs(probs_left - probs_right))
        
        twoing_value = (p_L * p_R / 4) * (diff_sum ** 2)
        return twoing_value

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature] < node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)


# --- ADAPTER SINIFI (Proje Uyumu İçin) ---

class TwoingClassifier(BaseClassifier):
    def __init__(self):
        # Scikit-learn değil, kendi yazdığımız ağacı kullanıyoruz.
        # max_depth=10: Sonsuz döngüye girmesin ve hızlı çalışsın diye sınır koyduk.
        self.model = CustomTwoingTree(max_depth=10, min_samples_split=5)

    def train(self, X_train, y_train):
        # Y verisinin integer olduğundan emin olalım (bincount için)
        y_train = y_train.astype(int)
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def get_name(self):
        return "Twoing (Custom Impl.)"
