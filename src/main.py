import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class GenomicDataAnalyzer:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        
    def load_data(self, data_path):
        """Load genomic data from file"""
        try:
            self.data = pd.read_csv(data_path)
            print(f"Data loaded successfully. Shape: {self.data.shape}")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
            
    def preprocess_data(self):
        """Preprocess the genomic data"""
        # Handle missing values
        self.data = self.data.fillna(self.data.mean())
        
        # Separate features and target
        X = self.data.drop('target', axis=1)
        y = self.data['target']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
    def train_model(self, model_name='random_forest'):
        """Train a machine learning model"""
        if model_name == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Model {model_name} not implemented")
            
        model.fit(self.X_train, self.y_train)
        self.models[model_name] = model
        
    def evaluate_model(self, model_name='random_forest'):
        """Evaluate model performance"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
            
        model = self.models[model_name]
        y_pred = model.predict(self.X_test)
        
        # Print classification report
        print(classification_report(self.y_test, y_pred))
        
        # Plot confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
    def get_feature_importance(self, model_name='random_forest'):
        """Get feature importance from the model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
            
        model = self.models[model_name]
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_names = self.data.drop('target', axis=1).columns
            
            # Create feature importance DataFrame
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
            plt.title('Top 20 Most Important Features')
            plt.show()
            
            return feature_importance
        else:
            print("Model does not support feature importance")
            return None

if __name__ == "__main__":
    # Example usage
    analyzer = GenomicDataAnalyzer()
    
    # Load and preprocess data
    analyzer.load_data("data/genomic_data.csv")
    analyzer.preprocess_data()
    
    # Train and evaluate model
    analyzer.train_model()
    analyzer.evaluate_model()
    
    # Get feature importance
    analyzer.get_feature_importance() 