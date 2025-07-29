import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class GenomicModels:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(random_state=42),
            'xgboost': XGBClassifier(random_state=42),
            'lightgbm': LGBMClassifier(random_state=42),
            'svm': SVC(random_state=42),
            'logistic_regression': LogisticRegression(random_state=42),
            'neural_network': MLPClassifier(random_state=42)
        }
        
        # Define hyperparameter grids for each model
        self.param_grids = {
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            'xgboost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            'lightgbm': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            'svm': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            },
            'logistic_regression': {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            },
            'neural_network': {
                'hidden_layer_sizes': [(100,), (100, 50), (200, 100)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01]
            }
        }
        
    def train_model(self, X_train, y_train, model_name='random_forest', tune_hyperparameters=True):
        """Train a specific model with optional hyperparameter tuning"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not implemented")
            
        if tune_hyperparameters:
            # Perform grid search for hyperparameter tuning
            grid_search = GridSearchCV(
                self.models[model_name],
                self.param_grids[model_name],
                cv=5,
                scoring='f1_weighted',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            self.models[model_name] = grid_search.best_estimator_
            print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        else:
            # Train with default parameters
            self.models[model_name].fit(X_train, y_train)
            
    def evaluate_model(self, X_test, y_test, model_name):
        """Evaluate model performance"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
            
        model = self.models[model_name]
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
        
        return metrics
        
    def get_feature_importance(self, model_name):
        """Get feature importance from the model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
            
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        elif hasattr(model, 'coef_'):
            return np.abs(model.coef_[0])
        else:
            print(f"Model {model_name} does not support feature importance")
            return None
            
    def train_all_models(self, X_train, y_train, X_test, y_test, tune_hyperparameters=True):
        """Train and evaluate all models"""
        results = {}
        
        for model_name in self.models.keys():
            print(f"\nTraining {model_name}...")
            self.train_model(X_train, y_train, model_name, tune_hyperparameters)
            metrics = self.evaluate_model(X_test, y_test, model_name)
            results[model_name] = metrics
            
        return results 