from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqUtils import gc_fraction
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class SickleCellSequenceAnalyzer:
    def __init__(self):
        self.wild_type_seq = None
        self.potential_mutations = []
        
    def analyze_sequence(self, sequence_data: dict):
        """Analyze the HBB gene sequence and predict potential mutation sites"""
        # Parse the sequence
        sequence_str = str(sequence_data.get('sequence', ''))
        if not sequence_str:
            raise ValueError("No sequence data provided")
            
        # Clean the sequence (remove any non-nucleotide characters)
        sequence_str = ''.join(c for c in sequence_str.upper() if c in 'ATGC')
        if not sequence_str:
            raise ValueError("Invalid sequence data - no valid nucleotides found")
            
        self.wild_type_seq = Seq(sequence_str)
        
        # Predict potential mutation sites
        self.predict_mutation_sites()
        
        # Calculate sequence stats
        sequence_length = len(self.wild_type_seq)
        gc_content = gc_fraction(self.wild_type_seq) * 100
        
        return {
            'wild_type': str(self.wild_type_seq),
            'sequence_length': sequence_length,
            'gc_content': gc_content,
            'potential_mutations': self.potential_mutations
        }
        
    def predict_mutation_sites(self):
        """Predict potential mutation sites based on sequence characteristics"""
        # Look for all codons that could potentially cause disease
        for i in range(0, len(self.wild_type_seq) - 2, 3):
            codon = str(self.wild_type_seq[i:i+3])
            
            # Calculate local GC content
            window = 30
            start = max(0, i - window)
            end = min(len(self.wild_type_seq), i + window + 3)
            local_seq = self.wild_type_seq[start:end]
            gc_content = (local_seq.count('G') + local_seq.count('C')) / len(local_seq)
            
            # Calculate codon usage in the region
            codon_usage = {}
            for j in range(start, end - 2, 3):
                current_codon = str(self.wild_type_seq[j:j+3])
                codon_usage[current_codon] = codon_usage.get(current_codon, 0) + 1
            
            # Predict mutation likelihood based on features
            mutation_likelihood = self.calculate_mutation_likelihood(
                codon, gc_content, codon_usage
            )
            
            if mutation_likelihood > 0.3:  # Lower threshold to get more potential sites
                # Generate all possible single nucleotide mutations
                mutations = self.generate_possible_mutations(codon)
                
                self.potential_mutations.append({
                    'position': i + 1,  # 1-based position
                    'wild_type_codon': codon,
                    'possible_mutations': mutations,
                    'mutation_likelihood': mutation_likelihood,
                    'gc_content': gc_content,
                    'codon_usage': codon_usage
                })
        
    def calculate_mutation_likelihood(self, codon, gc_content, codon_usage):
        """Calculate likelihood of mutation based on sequence features"""
        # Higher GC content regions are more likely to have mutations
        gc_factor = gc_content
        
        # Rare codons are more likely to mutate
        total_codons = sum(codon_usage.values())
        codon_frequency = codon_usage.get(codon, 0) / total_codons
        rarity_factor = 1 - codon_frequency
        
        # Combine factors
        likelihood = (gc_factor + rarity_factor) / 2
        return likelihood
        
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
        
    def generate_sequence_report(self):
        """Generate a comprehensive sequence analysis report"""
        report = {
            'sequence_length': len(self.wild_type_seq),
            'gc_content': gc_fraction(self.wild_type_seq) * 100,
            'potential_mutations': self.potential_mutations,
            'total_potential_sites': len(self.potential_mutations)
        }
        
        return report
        
    def visualize_mutation(self, sequence_analysis):
        """Create visualization of potential mutation sites"""
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 6))
        
        # Convert sequence to numeric values for visualization
        nucleotide_map = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
        seq_numeric = [nucleotide_map[n] for n in sequence_analysis['wild_type']]
        
        # Plot sequence
        ax.imshow([seq_numeric], aspect='auto', cmap='viridis')
        ax.set_title('HBB Gene Sequence with Predicted Mutation Sites')
        ax.set_yticks([])
        
        # Highlight predicted mutation sites
        for mut in sequence_analysis['potential_mutations']:
            pos = mut['position'] - 1  # Convert to 0-based index
            ax.add_patch(plt.Rectangle((pos-1, -0.5), 3, 1, fill=False, 
                                     edgecolor='red', lw=2))
            # Add text label
            ax.text(pos, -1, f"{mut['wild_type_codon']}\n{mut['mutation_likelihood']:.2f}", 
                   ha='center', va='top', color='red')
        
        plt.tight_layout()
        return fig 