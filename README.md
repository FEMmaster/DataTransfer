# DataTransfer: Neural Network Based Interpolation Across Non-Nested Meshes

This repository contains the Python implementation and numerical experiments associated with the preprint:

**Jiaxiong Hao, Yunqing Huang, and Nianyu Yi**  
**[DataTransfer: Neural network based interpolation across non-nested meshes](https://arxiv.org/abs/2511.12402)**  
*arXiv:2511.12402* [math.NA], first submitted November 16, 2025 (v2: April 14, 2026).  
DOI: [10.48550/arXiv.2511.12402](https://doi.org/10.48550/arXiv.2511.12402)

## Overview

The proposed approach reformulates interpolation between non-nested meshes as a data-driven regression problem. Function values at source-mesh nodes are used to construct a global approximator, which can then be evaluated at points on a different target mesh without requiring matching mesh connectivity.

The repository includes experiments with multilayer perceptrons (MLPs), extreme learning machines (ELMs), and radial basis function (RBF) networks, as well as one- and two-dimensional comparisons with conventional interpolation methods. Supporting scripts prepare mesh datasets, explore sample-point enrichment, and visualize interpolation results.

The scripts are organized as individual numerical experiments rather than a single end-to-end application.

## Requirements

The code uses Python and the following packages:

- NumPy
- SciPy
- Matplotlib
- PyTorch
- VTK
- PyVista (for 3D visualization of numerical results)

Specific tested package versions are not documented in the repository.

## Usage

The repository provides a complete workflow for generating mesh-based datasets, training neural network approximators, and transferring function values between non-nested meshes. Users can reproduce the experiments using the provided datasets or generate new data by running the corresponding scripts.

### 1. Data Generation

Two scripts are provided for preparing mesh-based datasets:

- `Train_mesh.py`: Generates multiple meshes and evaluates a prescribed function at their nodes to construct the dataset.
- `Train_document.py`: Loads existing VTU mesh files and prepares the corresponding dataset for interpolation experiments.

Both scripts can generate `mesh_data.npz`, which contains five meshes (`T1`–`T5`) and their associated nodal function values.

To generate a new dataset, run:

```bash
python Train_mesh.py
```

Alternatively, to prepare data from existing VTU meshes:

```bash
python Train_document.py
```

The generated dataset is saved in `train_data/`. An example dataset is also included in the repository, allowing the interpolation experiments to be performed without regenerating the data.

The prescribed functions and mesh configurations can be modified in the data-generation scripts to investigate different interpolation problems.

### 2. Neural Network-Based Data Transfer

The repository implements three neural network architectures for interpolation across non-nested meshes:

| Script | Method |
| --- | --- |
| `FIT_NN.py` | Multilayer perceptron (MLP) |
| `FIT_ELM.py` | Extreme learning machine (ELM) |
| `FIT_RBF.py` | Radial basis function (RBF) network |

Run the corresponding experiments using:

```bash
python FIT_NN.py
python FIT_ELM.py
python FIT_RBF.py
```

The models are fitted using the nodal function values on the source mesh `T1`. Once constructed, the approximators are evaluated on the target meshes `T2`–`T5` to perform cross-mesh interpolation.

The numerical results and solution-comparison figures are saved in `test_result/`.

### 3. Conventional Interpolation Comparisons

Additional experiments compare RBF interpolation with conventional piecewise-linear interpolation in one and two dimensions.

- `1D_RBFvsLINE_smooth.py` and `1D_RBFvsLINE_peak.py`: One-dimensional interpolation experiments.
- `2D_RBFvsLINE_smooth.py` and `2D_RBFvsLINE_peak.py`: Two-dimensional interpolation experiments.

These scripts investigate interpolation accuracy for smooth and sharply varying functions on different meshes.

**Note:** Some scripts use absolute file paths from the original development environment. Update the input and output directory paths before running the experiments.

## Code Structure

### Interpolation Comparisons

| File | Description |
| --- | --- |
| [`1D_RBFvsLINE_smooth.py`](1D_RBFvsLINE_smooth.py) | One-dimensional comparison of RBF and piecewise-linear interpolation for a smooth test function. |
| [`1D_RBFvsLINE_peak.py`](1D_RBFvsLINE_peak.py) | Corresponding one-dimensional comparison for a sharply varying test function. |
| [`2D_RBFvsLINE_smooth.py`](2D_RBFvsLINE_smooth.py) | Two-dimensional RBF and triangular linear-interpolation comparison for a smooth test function. |
| [`2D_RBFvsLINE_peak.py`](2D_RBFvsLINE_peak.py) | Corresponding two-dimensional comparison for a sharply varying test function. |

These scripts examine interpolation between different meshes and generate plots for visual comparison of the numerical results.

### Learning-Based Approximation Examples

| File | Description |
| --- | --- |
| [`FIT_NN.py`](FIT_NN.py) | Neural network fitting of solution data on meshes. |
| [`FIT_ELM.py`](FIT_ELM.py) | Extreme learning machine (ELM) fitting experiment. |
| [`FIT_RBF.py`](FIT_RBF.py) | Radial basis function network fitting experiment. |
| [`ELM_add_points_adapt.py`](ELM_add_points_adapt.py) | ELM-based experiment with adaptive enrichment of sample points. |
| [`ELM_add_points_maun.py`](ELM_add_points_maun.py) | ELM-based experiment with manually configured point enrichment and mesh-transfer tests. |

### Dataset Generation and Utilities

| File or directory | Description |
| --- | --- |
| [`Train_mesh.py`](Train_mesh.py) | Prepares mesh-based numerical data and saves a dataset for the fitting examples. |
| [`Train_document.py`](Train_document.py) | Prepares a dataset from VTU mesh files and produces mesh visualizations. |
| [`Train_dataset.py`](Train_dataset.py) | Generates and visualizes training points in a square domain. |
| [`utils_functions.py`](utils_functions.py) | Shared mesh-processing, numerical, dataset, and visualization utilities. |
| [`train_data/`](train_data/) | Example mesh files, prepared datasets, and related figures. |
| [`test_result/`](test_result/) | Saved plots from the numerical experiments. |

## Using the Examples

The fitting scripts `FIT_NN.py`, `FIT_ELM.py`, and `FIT_RBF.py` read the prepared dataset `mesh_data.npz` from `train_data/`. The repository includes this dataset and example VTU files.

Some scripts contain absolute paths from the original development environment (for example, `/home/paper2/` and `/home/fit_elm/`). Update these paths to match your local checkout before running the corresponding experiments. Output directories may likewise need adjustment.

The interpolation-comparison scripts are separate experiments and generate their own comparison figures.

## Citation

If you use this code in academic work, please cite:

```bibtex
@misc{Hao2025DataTransfer,
  title         = {DataTransfer: Neural network based interpolation across non-nested meshes},
  author        = {Hao, Jiaxiong and Huang, Yunqing and Yi, Nianyu},
  year          = {2025},
  eprint        = {2511.12402},
  archivePrefix = {arXiv},
  primaryClass  = {math.NA},
  doi           = {10.48550/arXiv.2511.12402},
  url           = {https://arxiv.org/abs/2511.12402}
}
```
