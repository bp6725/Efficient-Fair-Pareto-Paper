"""
Baseline Method Comparison

This script compares FairPareto against baseline fairness methods:
- FairGBM (in-processing)
- Balanced Group Threshold (post-processing)

Usage:
    python experiments/baseline_comparison.py --datasets ADULT COMPAS LSAC
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Baseline methods
try:
    from aequitas.flow.methods.inprocessing import FairGBM
    from aequitas.flow.methods.postprocessing import BalancedGroupThreshold
    from lightgbm import LGBMClassifier
    BASELINES_AVAILABLE = True
except ImportError:
    BASELINES_AVAILABLE = False
    print("Warning: Baseline methods not available. Install aequitas and lightgbm.")

# Import fairpareto
from fairpareto import FairParetoClassifier

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.data_loading import load_legacy_dataset
from utils.plotting import plot_baseline_comparison, plot_grid_comparison

warnings.filterwarnings('ignore')


def run_fairgbm_baseline(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    s_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    s_test: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    s_val: np.ndarray,
    verbose: bool = False
) -> Dict[float, float]:
    """
    Run FairGBM baseline with multiple hyperparameter configurations.
    
    Returns
    -------
    results : Dict[float, float]
        Dictionary mapping gamma (fairness) to accuracy
    """
    if not BASELINES_AVAILABLE:
        return {}
    
    if verbose:
        print("\nRunning FairGBM baseline...")
    
    results = {}
    
    # Grid of hyperparameters to try
    constraint_types = ['FNR', 'FPR']
    multiplier_lrs = [0.01, 0.1, 0.5]
    
    configs = []
    for ct in constraint_types:
        for mlr in multiplier_lrs:
            configs.append({
                'constraint_type': ct,
                'multiplier_learning_rate': mlr,
                'n_estimators': 100,
                'learning_rate': 0.1
            })
    
    for config in tqdm(configs, desc="FairGBM configs", disable=not verbose):
        try:
            model = FairGBM(**config)
            model.fit(X_train, y_train, s_train)
            
            # Get predictions on test set
            scores_val = model.predict_proba(X_val, s_val)
            scores_test = model.predict_proba(X_test, s_test)
            
            # Find optimal threshold on validation
            thresholds = np.linspace(0, 1, 100)
            best_acc = 0
            best_threshold = 0.5
            
            for threshold in thresholds:
                y_pred = (scores_val >= threshold).astype(int)
                acc = accuracy_score(y_val, y_pred)
                if acc > best_acc:
                    best_acc = acc
                    best_threshold = threshold
            
            # Apply threshold on test set
            y_test_pred = (scores_test >= best_threshold).astype(int)
            
            # Calculate metrics
            acc = accuracy_score(y_test, y_test_pred)
            
            # Calculate statistical parity
            group_0_rate = np.mean(y_test_pred[s_test == 0])
            group_1_rate = np.mean(y_test_pred[s_test == 1])
            stat_parity = abs(group_1_rate - group_0_rate)
            
            results[stat_parity] = acc
            
        except Exception as e:
            if verbose:
                print(f"FairGBM config failed: {e}")
            continue
    
    if verbose:
        print(f"FairGBM: Generated {len(results)} points")
    
    return results


def run_postprocessing_baseline(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    s_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    s_test: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    s_val: np.ndarray,
    verbose: bool = False
) -> Dict[float, float]:
    """
    Run LGBM + post-processing baseline.
    
    Returns
    -------
    results : Dict[float, float]
        Dictionary mapping gamma (fairness) to accuracy
    """
    if not BASELINES_AVAILABLE:
        return {}
    
    if verbose:
        print("\nRunning Post-processing baseline...")
    
    results = {}
    
    # Train base LGBM model
    model = LGBMClassifier(random_state=42, verbose=-1)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[LGBMClassifier.LGBMCallback()]
    )
    
    # Get probability scores
    scores_val = model.predict_proba(X_val)[:, 1]
    scores_test = model.predict_proba(X_test)[:, 1]
    
    scores_val_series = pd.Series(scores_val, index=X_val.index)
    scores_test_series = pd.Series(scores_test, index=X_test.index)
    
    # Try different post-processing configurations
    metrics_acc = ['top_pct', 'fpr']
    metrics_sp = ['fpr', 'pprev']
    threshold_vals = np.linspace(0.01, 1, 20)
    
    for m_acc in metrics_acc:
        for m_sp in metrics_sp:
            for fpr_val in tqdm(threshold_vals, desc=f"Post-proc {m_acc}-{m_sp}", 
                              disable=not verbose, leave=False):
                try:
                    threshold = BalancedGroupThreshold(m_acc, fpr_val, m_sp)
                    threshold.fit(X_val, scores_val_series, y_val, s_val)
                    
                    corrected_scores = threshold.transform(
                        X_test, scores_test_series, s_test
                    )
                    
                    # Calculate metrics
                    acc = accuracy_score(y_test, corrected_scores)
                    
                    group_0_rate = np.mean(corrected_scores[s_test == 0])
                    group_1_rate = np.mean(corrected_scores[s_test == 1])
                    stat_parity = abs(group_1_rate - group_0_rate)
                    
                    results[stat_parity] = acc
                    
                except Exception as e:
                    continue
    
    if verbose:
        print(f"Post-processing: Generated {len(results)} points")
    
    return results


def run_mifpo(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    s_train: np.ndarray,
    n_bins: int = 50,
    k: int = 5,
    verbose: bool = False
) -> Dict[float, float]:
    """
    Run MIFPO algorithm.
    
    Returns
    -------
    pareto_front : Dict[float, float]
        Dictionary mapping gamma to accuracy
    """
    if verbose:
        print("\nRunning MIFPO...")
    
    clf = FairParetoClassifier(
        n_bins=n_bins,
        k=k,
        loss_type='accuracy',
        verbose=verbose
    )
    
    # Add sensitive column to features
    X_combined = X_train.copy()
    X_combined['sensitive'] = s_train
    
    clf.fit(X_combined, y_train, sensitive_column='sensitive')
    
    if verbose:
        print(f"MIFPO: Generated {len(clf.pareto_front_)} points")
    
    return clf.pareto_front_


def compare_single_dataset(
    dataset_name: str,
    data_dir: Path,
    output_dir: Path,
    n_bins: int = 50,
    k: int = 5,
    verbose: bool = True
) -> Tuple[Dict[float, float], Dict[float, float], Dict[float, float]]:
    """
    Compare all methods on a single dataset.
    
    Returns
    -------
    mifpo_results : Dict[float, float]
        MIFPO Pareto front
    fairgbm_results : Dict[float, float]
        FairGBM results
    postprocess_results : Dict[float, float]
        Post-processing results
    """
    print(f"\n{'='*80}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'='*80}")
    
    # Load data
    try:
        train, test, val = load_legacy_dataset(dataset_name, data_dir)
        
        # Extract features and labels
        X_train = train.drop(['attr', 'label'], axis=1)
        y_train = train['label'].values
        s_train = train['attr'].values
        
        X_test = test.drop(['attr', 'label'], axis=1)
        y_test = test['label'].values
        s_test = test['attr'].values
        
        X_val = val.drop(['attr', 'label'], axis=1)
        y_val = val['label'].values
        s_val = val['attr'].values
        
        print(f"Train: {X_train.shape}, Test: {X_test.shape}, Val: {X_val.shape}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return {}, {}, {}
    
    # Run MIFPO
    mifpo_results = run_mifpo(X_train, y_train, s_train, n_bins, k, verbose)
    
    # Run baselines
    fairgbm_results = run_fairgbm_baseline(
        X_train, y_train, s_train,
        X_test, y_test, s_test,
        X_val, y_val, s_val,
        verbose
    )
    
    postprocess_results = run_postprocessing_baseline(
        X_train, y_train, s_train,
        X_test, y_test, s_test,
        X_val, y_val, s_val,
        verbose
    )
    
    # Save results
    results = {
        'mifpo': mifpo_results,
        'fairgbm': fairgbm_results,
        'postprocess': postprocess_results
    }
    
    results_file = output_dir / 'metrics' / f'{dataset_name}_comparison.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    # Create comparison plot
    plot_file = output_dir / 'plots' / f'{dataset_name}_comparison.png'
    plot_baseline_comparison(
        mifpo_results,
        fairgbm_results if fairgbm_results else None,
        postprocess_results if postprocess_results else None,
        dataset_name=dataset_name,
        output_path=plot_file
    )
    
    print(f"\nResults saved to {results_file}")
    print(f"Plot saved to {plot_file}")
    
    return mifpo_results, fairgbm_results, postprocess_results


def main():
    """Main comparison function."""
    parser = argparse.ArgumentParser(
        description='Compare FairPareto against baseline methods'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['ADULT', 'COMPAS', 'LSAC'],
        help='Dataset names to evaluate'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Base data directory'
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
        help='Number of bins for MIFPO'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=5,
        help='k parameter for MIFPO'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress'
    )
    
    args = parser.parse_args()
    
    # Convert paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directories
    (output_dir / 'plots').mkdir(parents=True, exist_ok=True)
    (output_dir / 'metrics').mkdir(parents=True, exist_ok=True)
    
    if not BASELINES_AVAILABLE:
        print("\nWarning: Baseline methods not available.")
        print("Install with: pip install aequitas lightgbm")
        print("Proceeding with MIFPO only...\n")
    
    # Run comparisons
    all_results = {}
    
    for dataset_name in args.datasets:
        mifpo, fairgbm, postprocess = compare_single_dataset(
            dataset_name=dataset_name,
            data_dir=data_dir,
            output_dir=output_dir,
            n_bins=args.n_bins,
            k=args.k,
            verbose=args.verbose
        )
        
        if mifpo:
            all_results[dataset_name] = {
                'mifpo': mifpo,
                'fairgbm': fairgbm,
                'postprocess': postprocess
            }
    
    # Create grid comparison plot
    if all_results:
        grid_plot_file = output_dir / 'plots' / 'all_datasets_comparison.png'
        plot_grid_comparison(
            all_results,
            list(all_results.keys()),
            output_path=grid_plot_file
        )
        print(f"\nGrid comparison plot saved to {grid_plot_file}")
    
    print("\nComparison complete!")


if __name__ == '__main__':
    main()
