"""
Plotting Utilities

Functions for visualizing Pareto fronts and comparison results.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300


def plot_pareto_front(
    pareto_front: Dict[float, float],
    ax: Optional[plt.Axes] = None,
    label: str = 'Pareto Front',
    color: str = 'green',
    marker: str = 'o',
    linestyle: str = '-',
    markersize: int = 8,
    linewidth: int = 2,
    show_points: bool = True
) -> plt.Axes:
    """
    Plot a single Pareto front.
    
    Parameters
    ----------
    pareto_front : Dict[float, float]
        Dictionary mapping gamma (fairness) to accuracy
    ax : plt.Axes, optional
        Matplotlib axes to plot on. If None, creates new figure
    label : str, default='Pareto Front'
        Legend label
    color : str, default='green'
        Line and marker color
    marker : str, default='o'
        Marker style
    linestyle : str, default='-'
        Line style
    markersize : int, default=8
        Size of markers
    linewidth : int, default=2
        Width of line
    show_points : bool, default=True
        Whether to show individual points
        
    Returns
    -------
    ax : plt.Axes
        Matplotlib axes with plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by gamma for proper line plotting
    gammas = sorted(pareto_front.keys())
    accuracies = [pareto_front[g] for g in gammas]
    
    # Plot line
    ax.plot(
        gammas, 
        accuracies,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        label=label,
        alpha=0.8
    )
    
    # Plot points if requested
    if show_points:
        ax.scatter(
            gammas,
            accuracies,
            color=color,
            marker=marker,
            s=markersize**2,
            zorder=5,
            edgecolors='white',
            linewidths=1
        )
    
    return ax


def save_pareto_front_plot(
    pareto_front: Dict[float, float],
    output_path: Path,
    title: str = 'Fairness-Performance Pareto Front',
    xlabel: str = 'Statistical Parity (γ)',
    ylabel: str = 'Accuracy',
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Create and save a Pareto front plot.
    
    Parameters
    ----------
    pareto_front : Dict[float, float]
        Dictionary mapping gamma to accuracy
    output_path : Path
        Where to save the plot
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    figsize : Tuple[int, int]
        Figure size (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    plot_pareto_front(pareto_front, ax=ax)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def plot_multiple_pareto_fronts(
    pareto_fronts: Dict[str, Dict[float, float]],
    title: str = 'Pareto Front Comparison',
    xlabel: str = 'Statistical Parity (γ)',
    ylabel: str = 'Accuracy',
    figsize: Tuple[int, int] = (12, 7),
    colors: Optional[List[str]] = None,
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot multiple Pareto fronts for comparison.
    
    Parameters
    ----------
    pareto_fronts : Dict[str, Dict[float, float]]
        Dictionary mapping names to Pareto fronts
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    figsize : Tuple[int, int]
        Figure size
    colors : List[str], optional
        Colors for each curve. If None, uses default palette
    output_path : Path, optional
        If provided, saves plot to this path
        
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if colors is None:
        colors = sns.color_palette("husl", len(pareto_fronts))
    
    for (name, pf), color in zip(pareto_fronts.items(), colors):
        plot_pareto_front(
            pf,
            ax=ax,
            label=name,
            color=color,
            markersize=6
        )
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    
    return fig


def plot_baseline_comparison(
    mifpo_front: Dict[float, float],
    fairgbm_results: Optional[Dict[float, float]] = None,
    postprocess_results: Optional[Dict[float, float]] = None,
    dataset_name: str = '',
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot MIFPO Pareto front against baseline methods.
    
    Parameters
    ----------
    mifpo_front : Dict[float, float]
        MIFPO Pareto front
    fairgbm_results : Dict[float, float], optional
        FairGBM results (gamma -> accuracy)
    postprocess_results : Dict[float, float], optional
        Post-processing method results
    dataset_name : str
        Name of dataset for title
    output_path : Path, optional
        Where to save plot
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot MIFPO front (the optimal curve)
    plot_pareto_front(
        mifpo_front,
        ax=ax,
        label='MIFPO (Optimal)',
        color='green',
        marker='o',
        markersize=8,
        linewidth=2.5
    )
    
    # Plot baseline methods as scatter points
    if fairgbm_results:
        gammas = list(fairgbm_results.keys())
        accs = list(fairgbm_results.values())
        ax.scatter(
            gammas, accs,
            marker='+',
            s=100,
            color='blue',
            label='FairGBM',
            zorder=3,
            linewidths=2
        )
    
    if postprocess_results:
        gammas = list(postprocess_results.keys())
        accs = list(postprocess_results.values())
        ax.scatter(
            gammas, accs,
            marker='x',
            s=100,
            color='red',
            label='LGBM + Post-process',
            zorder=3,
            linewidths=2
        )
    
    ax.set_xlabel('Statistical Parity (γ)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    
    title = f'Fairness-Performance Comparison'
    if dataset_name:
        title += f': {dataset_name}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    
    return fig


def plot_grid_comparison(
    all_results: Dict[str, Dict[str, Dict[float, float]]],
    datasets: List[str],
    figsize: Tuple[int, int] = (20, 12),
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create a grid of subplots comparing results across datasets.
    
    Parameters
    ----------
    all_results : Dict[str, Dict[str, Dict[float, float]]]
        Nested dict: dataset -> method -> pareto_front
    datasets : List[str]
        List of dataset names to plot
    figsize : Tuple[int, int]
        Figure size
    output_path : Path, optional
        Where to save plot
        
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    n_datasets = len(datasets)
    n_cols = 3
    n_rows = (n_datasets + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_datasets > 1 else [axes]
    
    for idx, dataset_name in enumerate(datasets):
        ax = axes[idx]
        
        if dataset_name not in all_results:
            ax.set_visible(False)
            continue
        
        methods = all_results[dataset_name]
        
        # Plot each method
        colors = {'mifpo': 'green', 'fairgbm': 'blue', 'postprocess': 'red'}
        markers = {'mifpo': 'o', 'fairgbm': '+', 'postprocess': 'x'}
        
        for method_name, pf in methods.items():
            if method_name.lower() == 'mifpo':
                plot_pareto_front(
                    pf, ax=ax,
                    label=method_name,
                    color=colors.get(method_name.lower(), 'gray'),
                    marker=markers.get(method_name.lower(), 'o')
                )
            else:
                # Baseline methods as scatter
                gammas = list(pf.keys())
                accs = list(pf.values())
                ax.scatter(
                    gammas, accs,
                    marker=markers.get(method_name.lower(), 'o'),
                    s=80,
                    label=method_name,
                    color=colors.get(method_name.lower(), 'gray')
                )
        
        ax.set_title(dataset_name, fontsize=11, fontweight='bold')
        ax.set_xlabel('Statistical Parity', fontsize=9)
        ax.set_ylabel('Accuracy', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_datasets, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle('Fairness-Performance Comparison Across Datasets', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    
    return fig
