# Quick Start Guide

This guide will help you get started with the FairPareto evaluations.

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/fairpareto-evaluations.git
cd fairpareto-evaluations
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Downloading Datasets

### FARE Datasets

The FARE (Fairness Across REpresentations) benchmark datasets should be downloaded and placed in `data/Fare_datasets/`. 

Expected structure:
```
data/
└── Fare_datasets/
    ├── ACSIncome-ALL-2014.pkl
    ├── ACSEmployment-ALL-2014.pkl
    ├── ACSPublicCoverage-ALL-2014.pkl
    └── ... (other FARE datasets)
```

### Legacy Datasets (ADULT, COMPAS, LSAC)

For baseline comparisons, legacy datasets should be placed in:
```
data/
├── ADULT/
│   └── data_file.pkl
├── COMPAS/
│   └── data_file.pkl
└── LSAC/
    └── data_file.pkl
```

## Running Experiments

### 1. FARE Dataset Evaluation

Evaluate FairPareto on all FARE datasets:

```bash
python experiments/fare_evaluation.py --datasets all --output-dir results/ --verbose
```

Evaluate on specific datasets:

```bash
python experiments/fare_evaluation.py \
    --datasets ACSIncome-ALL-2014.pkl ACSEmployment-ALL-2014.pkl \
    --n-bins 50 \
    --k 5 \
    --verbose
```

### 2. Baseline Comparison

Compare against FairGBM and post-processing baselines:

```bash
python experiments/baseline_comparison.py \
    --datasets ADULT COMPAS LSAC \
    --output-dir results/ \
    --verbose
```

**Note**: This requires `aequitas` and `lightgbm` to be installed:
```bash
pip install aequitas lightgbm
```

## Viewing Results

### Results Directory Structure

After running experiments, you'll find:

```
results/
├── plots/                          # Visualization plots
│   ├── ACSIncome_pareto.png
│   ├── ADULT_comparison.png
│   └── all_datasets_comparison.png
└── metrics/                        # Numerical results
    ├── ACSIncome_pareto.pkl
    ├── ADULT_comparison.pkl
    └── summary_table.csv
```

### Using the Analysis Notebook

1. **Start Jupyter**:
```bash
jupyter notebook
```

2. **Open the analysis notebook**:
   - Navigate to `notebooks/results_analysis.ipynb`
   - Run all cells to generate analysis

### Command-Line Summary

View summary statistics:

```bash
# Summary table is automatically generated
cat results/metrics/summary_table.csv
```

## Example Usage

### Quick Test Run

Test on a single dataset:

```bash
python experiments/fare_evaluation.py \
    --datasets ACSIncome-ALL-2014.pkl \
    --n-bins 20 \
    --k 5 \
    --output-dir test_results/ \
    --verbose
```

### Batch Processing

Process multiple datasets with specific parameters:

```bash
for dataset in ACSIncome ACSEmployment ACSPublicCoverage; do
    python experiments/fare_evaluation.py \
        --datasets ${dataset}-ALL-2014.pkl \
        --n-bins 50 \
        --k 5 \
        --output-dir results/${dataset}/ \
        --verbose
done
```

## Parameters

### FairPareto Parameters

- `--n-bins`: Number of histogram bins for discretization (default: 50)
  - Smaller values: Faster computation, coarser approximation
  - Larger values: Better approximation, slower computation
  
- `--k`: Number of representation points per bin pair (default: 5)
  - Affects the granularity of the Pareto front approximation

### Experiment Parameters

- `--datasets`: List of datasets to evaluate
  - Use `all` for all available FARE datasets
  - Or specify individual dataset names

- `--force-recompute`: Recompute even if cached results exist

- `--verbose`: Print detailed progress information

## Troubleshooting

### Missing Dependencies

If you see import errors:
```bash
pip install -r requirements.txt
```

### Dataset Not Found

Ensure datasets are in the correct directories:
- FARE datasets: `data/Fare_datasets/*.pkl`
- Legacy datasets: `data/{ADULT,COMPAS,LSAC}/data_file.pkl`

### Memory Issues

For large datasets, reduce `n-bins` or `k`:
```bash
python experiments/fare_evaluation.py \
    --datasets large_dataset.pkl \
    --n-bins 20 \
    --k 3
```

### Baseline Methods Failing

If baseline comparisons fail:
1. Ensure `aequitas` and `lightgbm` are installed
2. Check that validation data is available
3. Try with `--verbose` flag for detailed error messages

## Next Steps

1. **Explore Results**: Use the Jupyter notebook for interactive analysis
2. **Generate Paper Figures**: Modify plotting utilities for publication-quality figures
3. **Add New Datasets**: Add new datasets following the FARE format
4. **Customize Experiments**: Modify experiment scripts for specific research questions

## Getting Help

For issues or questions:
1. Check the main README.md
2. Review the code documentation
3. Open an issue on GitHub

## Citation

If you use this code in your research, please cite the paper:

```bibtex
@inproceedings{fairpareto2025,
  title={Efficient Fairness-Performance Pareto Front Computation},
  author={Anonymous},
  booktitle={ICLR 2025 (Under Review)},
  year={2025}
}
```
