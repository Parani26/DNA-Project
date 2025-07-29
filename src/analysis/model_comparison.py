import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

class SickleCellModelAnalyzer:
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(random_state=42),
            'XGBoost': XGBClassifier(random_state=42),
            'LightGBM': LGBMClassifier(random_state=42),
            'SVM': SVC(random_state=42, probability=True),
            'Logistic Regression': LogisticRegression(random_state=42)
        }
        self.scaler = StandardScaler()
        self.results = {}
        self.predicted_mutations = {}
        self.sequence = None
        
    def prepare_data(self, sequence_data, variants=None):
        """Prepare data for mutation prediction"""
        self.sequence = sequence_data['wild_type']
        sequence = str(sequence_data['wild_type'])
        X = []
        y = []
        
        # Extract features for each codon
        for i in range(0, len(sequence) - 2, 3):
            codon = sequence[i:i+3]
            
            # Calculate features
            features = self.extract_features(sequence, i)
            
            # Label: 1 if it's a potential mutation site
            label = 1 if self.is_potential_mutation_site(codon, features) else 0
            
            X.append(features)
            y.append(label)
            
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
        
    def extract_features(self, sequence, position):
        """Extract features for mutation prediction"""
        window = 30
        start = max(0, position - window)
        end = min(len(sequence), position + window + 3)
        local_seq = sequence[start:end]
        
        # Feature 1: GC content in local region
        gc_content = (local_seq.count('G') + local_seq.count('C')) / len(local_seq)
        
        # Feature 2: Codon usage frequency
        codon_usage = {}
        for i in range(0, len(local_seq) - 2, 3):
            codon = local_seq[i:i+3]
            codon_usage[codon] = codon_usage.get(codon, 0) + 1
        total_codons = sum(codon_usage.values())
        current_codon = sequence[position:position+3]
        codon_frequency = codon_usage.get(current_codon, 0) / total_codons
        
        # Feature 3: Position in gene (normalized)
        position_norm = position / len(sequence)
        
        # Feature 4: Nucleotide composition
        nucleotide_composition = [
            local_seq.count('A') / len(local_seq),
            local_seq.count('T') / len(local_seq),
            local_seq.count('G') / len(local_seq),
            local_seq.count('C') / len(local_seq)
        ]
        
        return [gc_content, codon_frequency, position_norm] + nucleotide_composition
        
    def is_potential_mutation_site(self, codon, features):
        """Determine if a codon is a potential mutation site"""
        gc_content, codon_frequency, position_norm = features[:3]
        
        # Higher GC content regions are more likely to have mutations
        gc_factor = gc_content > 0.4
        
        # Rare codons are more likely to mutate
        rarity_factor = codon_frequency < 0.1
        
        # Middle of the gene is more likely to have functional mutations
        position_factor = 0.3 < position_norm < 0.7
        
        return gc_factor or rarity_factor or position_factor
        
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Train and evaluate models"""
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0)
            }
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            self.results[name] = {
                'metrics': metrics,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            # Store predicted mutation sites
            self.predicted_mutations[name] = self.predict_mutation_sites(
                model, X_test, y_test
            )
            
            print(f"Metrics for {name}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
            print(f"  CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
    def predict_mutation_sites(self, model, X_test, y_test=None):
        """Predict potential mutation sites using the trained model."""
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except (AttributeError, NotImplementedError):
            try:
                # For SVM, normalize decision_function to [0,1] range
                decisions = model.decision_function(X_test)
                y_proba = (decisions - decisions.min()) / (decisions.max() - decisions.min())
            except:
                # If neither method works, use binary predictions
                y_proba = model.predict(X_test).astype(float)
        
        # Get indices of top 10 predicted mutation sites
        top_indices = np.argsort(y_proba)[-10:][::-1]
        
        predicted_sites = []
        for idx in top_indices:
            position = idx * 3  # Convert to nucleotide position
            codon = self.sequence[position:position+3]
            mutations = self.generate_possible_mutations(codon)
            
            site_info = {
                'position': position + 1,  # 1-based position
                'probability': float(y_proba[idx]),
                'wild_type_codon': codon,
                'possible_mutations': mutations
            }
            if y_test is not None:
                site_info['actual'] = bool(y_test[idx])
            predicted_sites.append(site_info)
            
        return predicted_sites
    
    def generate_possible_mutations(self, codon):
        """Generate all possible single nucleotide mutations for a codon"""
        nucleotides = ['A', 'T', 'G', 'C']
        mutations = []
        
        for i in range(3):
            for n in nucleotides:
                if n != codon[i]:
                    mutated_codon = list(codon)
                    mutated_codon[i] = n
                    mutations.append(''.join(mutated_codon))
        
        return mutations
        
    def plot_results(self):
        """Visualize model comparison results"""
        # Create metrics DataFrame
        metrics_df = pd.DataFrame({
            name: results['metrics']
            for name, results in self.results.items()
        }).T
        
        # Plot metrics
        plt.figure(figsize=(12, 6))
        metrics_df.plot(kind='bar')
        plt.title('Model Performance Comparison')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save plot
        plt.savefig('results/model_comparison.png')
        plt.close()
        
        # Plot predicted mutation sites
        plt.figure(figsize=(15, 8))
        for i, (name, sites) in enumerate(self.predicted_mutations.items()):
            plt.subplot(2, 3, i+1)
            for site in sites:
                plt.bar(site['position'], site['probability'], 
                       label=f"Site {site['position']}")
            plt.title(f'{name} Predictions')
            plt.xlabel('Position')
            plt.ylabel('Mutation Probability')
            plt.legend()
        plt.tight_layout()
        plt.savefig('results/predicted_mutations.png')
        plt.close()
        
    def get_feature_importance(self):
        """Get and plot feature importance for tree-based models"""
        importance_results = {}
        feature_names = ['GC Content', 'Codon Frequency', 'Position', 
                        'A Content', 'T Content', 'G Content', 'C Content']
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_results[name] = dict(zip(feature_names, model.feature_importances_))
                
                # Plot feature importance
                plt.figure(figsize=(8, 4))
                plt.bar(feature_names, model.feature_importances_)
                plt.title(f'Feature Importance - {name}')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f'results/feature_importance_{name.lower().replace(" ", "_")}.png')
                plt.close()
            
        return importance_results

    def train_and_evaluate_all(self):
        """Train and evaluate all models, returning comprehensive results"""
        if not hasattr(self, 'X_train_scaled'):
            raise ValueError("Must call prepare_data before training models")

        # Train and evaluate all models
        self.train_and_evaluate(self.X_train_scaled, self.X_test_scaled, 
                              self.y_train, self.y_test)
        
        # Plot results
        self.plot_results()
        
        # Get feature importance
        importance_results = self.get_feature_importance()
        
        # Prepare final results
        final_results = {
            'metrics': self.results,
            'predicted_mutations': self.predicted_mutations,
            'feature_importance': importance_results
        }
        
        return final_results 