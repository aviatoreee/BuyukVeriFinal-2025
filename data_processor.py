import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, KBinsDiscretizer, LabelEncoder

class AdvancedDataProcessor:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.base_dir = os.path.dirname(dataset_path)

    def load_raw_data(self):
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError("SOMLAP.csv bulunamadı!")
        df = pd.read_csv(self.dataset_path)
        df.fillna(0, inplace=True)
        # Hedef değişkeni ayır (Son sütun varsayımı)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        # Label Encoding (Hedef değişken string ise)
        if y.dtype == 'object' or isinstance(y[0], str):
            le = LabelEncoder()
            y = le.fit_transform(y)
            
        return X, y

    def save_split(self, X, y, prefix):
        """Veriyi böler ve prefix ile kaydeder (örn: id3_train.csv)"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=42, stratify=y
        )
        
        # DataFrame oluştur
        train_df = pd.DataFrame(X_train)
        train_df['target'] = y_train
        
        test_df = pd.DataFrame(X_test)
        test_df['target'] = y_test
        
        train_path = os.path.join(self.base_dir, f'{prefix}_train.csv')
        test_path = os.path.join(self.base_dir, f'{prefix}_test.csv')
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        return len(X_train), len(X_test)

    def process_for_id3(self):
        """ID3 için Discretization (Sürekliden Kategorike)"""
        X, y = self.load_raw_data()
        # Veriyi 5 ayrı kategoriye bölüyoruz (Binning)
        est = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
        X_disc = est.fit_transform(X)
        return self.save_split(X_disc, y, "id3")

    def process_for_gini(self):
        """Gini için Standart Scaler"""
        X, y = self.load_raw_data()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return self.save_split(X_scaled, y, "gini")

    def process_for_twoing(self):
        """Twoing için MinMax Scaler"""
        X, y = self.load_raw_data()
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        return self.save_split(X_scaled, y, "twoing")

    def load_test_data(self, method_name):
        path = os.path.join(self.base_dir, f'{method_name}_test.csv')
        if not os.path.exists(path):
            raise FileNotFoundError(f"{method_name} için test verisi bulunamadı.")
        df = pd.read_csv(path)
        y = df['target'].values
        X = df.drop(columns=['target']).values
        return X, y
    
    def load_train_data(self, method_name):
        path = os.path.join(self.base_dir, f'{method_name}_train.csv')
        df = pd.read_csv(path)
        y = df['target'].values
        X = df.drop(columns=['target']).values
        return X, y