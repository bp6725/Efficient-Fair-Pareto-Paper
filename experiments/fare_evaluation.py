"""
FARE Datasets Evaluation

This script evaluates the FairPareto algorithm on the FARE benchmark datasets.
It computes the fairness-performance Pareto front and saves results for analysis.

Usage:
    python experiments/fare_evaluation.py --datasets all
    python experiments/fare_evaluation.py --datasets ACSIncome ACSEmployment
"""

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

# Import the fairpareto package
from fairpareto import FairParetoClassifier

# Add parent directory to path for utils
sys.path.append(str(Path(__file__).parent.parent))
from utils.data_loading import load_fare_dataset, get_available_fare_datasets
from utils.plotting import plot_pareto_front, save_pareto_front_plot

warnings.filterwarnings('ignore')


def run_single_evaluation(
    dataset_name: str,
    data_dir: Path,
    n_bins: int = 50,
    k: int = 5,
    verbose: bool = True
) -> Dict[float, float]:
    """
    Run FairPareto evaluation on a single dataset.
    
    Parameters
    ----------
    dataset_name : str
        Name of the FARE dataset
    data_dir : Path
        Directory containing the FARE datasets
    n_bins : int, default=50
        Number of bins for histogram discretization
    k : int, default=5
        Number of representation points per bin pair
    verbose : bool, default=True
        Whether to print progress messages
        
    Returns
    -------
    pareto_front : Dict[float, float]
        Dictionary mapping gamma (fairness level) to accuracy
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*60}")
    
    # Load dataset
    try:
        X_train, y_train, s_train = load_fare_dataset(
            dataset_name, 
            data_dir, 
            split='train'
        )
        
        if verbose:
            print(f"Dataset loaded: {X_train.shape[0]} samples, {X_train.shape[1]} features")
            print(f"Sensitive attribute distribution: {np.bincount(s_train)}")
            print(f"Label distribution: {np.bincount(y_train)}")
    
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        return {}
    
    # Initialize FairPareto classifier
    clf = FairParetoClassifier(
        n_bins=n_bins,
        k=k,
        loss_type='accuracy',
        verbose=verbose
    )
    
    # Fit and compute Pareto front
    try:
        # Combine sensitive attribute with features for sklearn compatibility
        X_combined = pd.DataFrame(X_train)
        X_combined['sensitive'] = s_train
        
        clf.fit(X_combined, y_train, sensitive_column='sensitive')
        
        pareto_front = clf.pareto_front_
        
        if verbose:
            print(f"\nPareto front computed with {len(pareto_front)} points")
            print(f"Fairness range: [{min(pareto_front.keys()):.4f}, {max(pareto_front.keys()):.4f}]")
            print(f"Accuracy range: [{min(pareto_front.values()):.4f}, {max(pareto_front.values()):.4f}]")
        
        return pareto_front
    
    except Exception as e:
        print(f"Error computing Pareto front for {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return {}


def run_all_evaluations(
    datasets: List[str],
    data_dir: Path,
    output_dir: Path,
    n_bins: int = 50,
    k: int = 5,
    verbose: bool = True
) -> Dict[str, Dict[float, float]]:
    """
    Run evaluations on multiple datasets.
    
    Parameters
    ----------
    datasets : List[str]
        List of dataset names to evaluate
    data_dir : Path
        Directory containing the FARE datasets
    output_dir : Path
        Directory to save results
    n_bins : int, default=50
        Number of bins for histogram discretization
    k : int, default=5
        Number of representation points per bin pair
    verbose : bool, default=True
        Whether to print progress messages
        
    Returns
    -------
    all_results : Dict[str, Dict[float, float]]
        Dictionary mapping dataset names to their Pareto fronts
    """
    all_results = {}
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'plots').mkdir(exist_ok=True)
    (output_dir / 'metrics').mkdir(exist_ok=True)
    
    # Process each dataset
    for dataset_name in tqdm(datasets, desc="Processing datasets"):
        # Check if results already exist
        results_file = output_dir / 'metrics' / f'{dataset_name}_pareto.pkl'
        
        if results_file.exists() and not args.force_recompute:
            print(f"\nLoading cached results for {dataset_name}")
            with open(results_file, 'rb') as f:
                pareto_front = pickle.load(f)
        else:
            # Run evaluation
            pareto_front = run_single_evaluation(
                dataset_name=dataset_name,
                data_dir=data_dir,
                n_bins=n_bins,
                k=k,
                verbose=verbose
            )
            
            # Save results
            if pareto_front:
                with open(results_file, 'wb') as f:
                    pickle.dump(pareto_front, f)
        
        if pareto_front:
            all_results[dataset_name] = pareto_front
            
            # Create and save plot
            plot_file = output_dir / 'plots' / f'{dataset_name}_pareto.png'
            save_pareto_front_plot(
                pareto_front,
                title=f'Pareto Front: {dataset_name}',
                output_path=plot_file
            )
    
    return all_results


def save_summary_table(
    all_results: Dict[str, Dict[float, float]],
    output_dir: Path
):
    """
    Save summary statistics table for all datasets.
    
    Parameters
    ----------
    all_results : Dict[str, Dict[float, float]]
        Dictionary mapping dataset names to their Pareto fronts
    output_dir : Path
        Directory to save the summary table
    """
    summary_data = []
    
    for dataset_name, pareto_front in all_results.items():
        if not pareto_front:
            continue
            
        gammas = list(pareto_front.keys())
        accuracies = list(pareto_front.values())
        
        summary_data.append({
            'Dataset': dataset_name,
            'Num Points': len(pareto_front),
            'Min Gamma': min(gammas),
            'Max Gamma': max(gammas),
            'Min Accuracy': min(accuracies),
            'Max Accuracy': max(accuracies),
            'Accuracy @ Gamma=0.05': pareto_front.get(0.05, np.nan),
            'Accuracy @ Gamma=0.10': pareto_front.get(0.10, np.nan),
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = output_dir / 'metrics' / 'summary_table.csv'
    summary_df.to_csv(summary_file, index=False)
    
    print(f"\nSummary table saved to {summary_file}")
    print("\n" + "="*80)
    print(summary_df.to_string(index=False))
    print("="*80)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description='Evaluate FairPareto on FARE benchmark datasets'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['all'],
        help='Dataset names to evaluate (or "all" for all available datasets)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/Fare_datasets',
        help='Directory containing FARE datasets'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Directory to save results'
    )
    parser.add_argument(
        '--n-bins',
        type=int,
        default=50,
        help='Number of bins for histogram discretization'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=5,
        help='Number of representation points per bin pair'
    )
    parser.add_argument(
        '--force-recompute',
        action='store_true',
        help='Force recomputation even if cached results exist'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress messages'
    )
    
    global args
    args = parser.parse_args()
    
    # Convert paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    # Validate data directory
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        print("Please download FARE datasets and place them in the data directory")
        sys.exit(1)
    
    # Get dataset list
    if 'all' in args.datasets:
        datasets = get_available_fare_datasets(data_dir)
        print(f"Found {len(datasets)} FARE datasets")
    else:
        datasets = args.datasets
    
    if not datasets:
        print("No datasets to process")
        sys.exit(1)
    
    print(f"\nStarting evaluation on {len(datasets)} datasets")
    print(f"Parameters: n_bins={args.n_bins}, k={args.k}")
    print(f"Results will be saved to: {output_dir}")
    
    # Run evaluations
    all_results = run_all_evaluations(
        datasets=datasets,
        data_dir=data_dir,
        output_dir=output_dir,
        n_bins=args.n_bins,
        k=args.k,
        verbose=args.verbose
    )
    
    # Save summary
    if all_results:
        save_summary_table(all_results, output_dir)
        print(f"\nEvaluation complete! Results saved to {output_dir}")
    else:
        print("\nNo results generated")


if __name__ == '__main__':
    main()
