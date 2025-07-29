import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
from data.sickle_cell_data import SickleCellDataAcquisition
from analysis.sequence_analysis import SickleCellSequenceAnalyzer
from analysis.model_comparison import SickleCellModelAnalyzer

def convert_to_python_types(obj):
    """Convert numpy and pandas types to Python native types"""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {key: convert_to_python_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(item) for item in obj]
    elif isinstance(obj, (bool, str)):
        return obj
    elif obj is None:
        return None
    else:
        try:
            return str(obj)
        except:
            return None

def main():
    # Create output directory if it doesn't exist
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    # Initialize data acquisition
    data_acquisition = SickleCellDataAcquisition(email="your.email@example.com")
    
    # Fetch and save raw data
    data = data_acquisition.get_all_data()
    with open(output_dir / "raw_data.json", "w") as f:
        json.dump(convert_to_python_types(data), f, indent=2)

    # Analyze sequence
    sequence_analyzer = SickleCellSequenceAnalyzer()
    sequence_analysis = sequence_analyzer.analyze_sequence(data['sequence'])
    with open(output_dir / "sequence_analysis.json", "w") as f:
        json.dump(convert_to_python_types(sequence_analysis), f, indent=2)

    # Visualize mutations
    sequence_analyzer.visualize_mutation(sequence_analysis)
    plt.savefig(output_dir / "mutation_visualization.png")
    plt.close()

    # Run model comparisons
    model_analyzer = SickleCellModelAnalyzer()
    model_analyzer.prepare_data(sequence_analysis, data['variants'])
    model_results = model_analyzer.train_and_evaluate_all()
    
    with open(output_dir / "model_results.json", "w") as f:
        json.dump(convert_to_python_types(model_results), f, indent=2)

    # Print summary
    print("\nAnalysis Summary:")
    print("----------------")
    print(f"HBB Gene Analysis:")
    print(f"- Sequence Length: {sequence_analysis['sequence_length']} bp")
    print(f"- GC Content: {sequence_analysis['gc_content']:.2f}%")
    print("\nModel Results:")
    for model_name, results in model_results['metrics'].items():
        print(f"\n{model_name}:")
        metrics = results['metrics']
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        print(f"  CV Score: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
        
        print("\n  Top 2 Predicted Mutation Sites:")
        for i, pred in enumerate(model_results['predicted_mutations'][model_name][:2], 1):
            print(f"    {i}. Position: {pred['position']}, "
                  f"Codon: {pred['wild_type_codon']}, "
                  f"Probability: {pred['probability']:.2f}")

    print("\nResults saved in 'results' directory:")
    print("- raw_data.json")
    print("- sequence_analysis.json")
    print("- mutation_visualization.png")
    print("- model_comparison.png")
    print("- predicted_mutations.png")
    print("- feature_importance_*.png")
    print("- model_results.json")

if __name__ == "__main__":
    main() 