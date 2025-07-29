import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif

class GenomicPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.feature_selector = None
        self.label_encoder = LabelEncoder()
        
    def load_genomic_data(self, file_path):
        """Load genomic data from various formats"""
        try:
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            elif file_path.endswith('.tsv'):
                data = pd.read_csv(file_path, sep='\t')
            else:
                raise ValueError("Unsupported file format")
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
            
    def handle_missing_values(self, data):
        """Handle missing values in genomic data"""
        # Replace missing values with mean
        data_imputed = self.imputer.fit_transform(data)
        return pd.DataFrame(data_imputed, columns=data.columns)
        
    def normalize_data(self, data):
        """Normalize genomic data"""
        return pd.DataFrame(
            self.scaler.fit_transform(data),
            columns=data.columns
        )
        
    def select_features(self, X, y, k=1000):
        """Select top k features using ANOVA F-value"""
        self.feature_selector = SelectKBest(f_classif, k=k)
        X_selected = self.feature_selector.fit_transform(X, y)
        selected_features = X.columns[self.feature_selector.get_support()]
        return pd.DataFrame(X_selected, columns=selected_features)
        
    def encode_labels(self, labels):
        """Encode categorical labels"""
        return self.label_encoder.fit_transform(labels)
        
    def preprocess_pipeline(self, data_path, target_column, k_features=1000):
        """Complete preprocessing pipeline"""
        # Load data
        data = self.load_genomic_data(data_path)
        if data is None:
            return None
            
        # Separate features and target
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        
        # Handle missing values
        X = self.handle_missing_values(X)
        
        # Normalize data
        X = self.normalize_data(X)
        
        # Select features
        X = self.select_features(X, y, k=k_features)
        
        # Encode labels if needed
        if y.dtype == 'object':
            y = self.encode_labels(y)
            
        return X, y
        
    def get_feature_names(self):
        """Get names of selected features"""
        if self.feature_selector is not None:
            return self.feature_selector.get_feature_names_out()
        return None 