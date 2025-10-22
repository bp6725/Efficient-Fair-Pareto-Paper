"""Utilities for FairPareto evaluations."""

from .data_loading import (
    load_fare_dataset,
    get_available_fare_datasets,
    load_legacy_dataset,
    format_fare_input,
    print_dataset_info
)

from .plotting import (
    plot_pareto_front,
    save_pareto_front_plot,
    plot_multiple_pareto_fronts,
    plot_baseline_comparison,
    plot_grid_comparison
)

__all__ = [
    # Data loading
    'load_fare_dataset',
    'get_available_fare_datasets',
    'load_legacy_dataset',
    'format_fare_input',
    'print_dataset_info',
    # Plotting
    'plot_pareto_front',
    'save_pareto_front_plot',
    'plot_multiple_pareto_fronts',
    'plot_baseline_comparison',
    'plot_grid_comparison',
]
