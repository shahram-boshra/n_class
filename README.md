# Node-Level Classification PyTorch Geometric Pipeline for Molecular Graphs

This repository contains a PyTorch Geometric pipeline for node-level classification on molecular graphs. It leverages RDKit for molecule processing and graph feature extraction, and implements a configurable Graph Neural Network (GNN) model for classification tasks.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Dataset](#dataset)
- [Model](#model)
- [Training](#training)
- [Testing](#testing)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
- [Future Enhancements](#future-enhancements)
- [Acknowledgements](#acknowledgements)

## Overview

This pipeline is designed for predicting node-level properties (e.g., atom properties) in molecular graphs. It provides a modular and configurable framework for:

- Loading and processing molecular data using RDKit.
- Extracting atomic and bond features.
- Building and training GNN models.
- Evaluating model performance.

## Features

- **Configurable RDKit Processing:** Define a sequence of RDKit processing steps via a YAML configuration file.
- **Feature Extraction:** Extract atomic and bond features using one-hot encoding.
- **Flexible GNN Model:** Build GNN models with various layer types (GCN, GAT, SAGE, GIN, etc.).
- **Dataset Splitting:** Stratified dataset splitting based on target values.
- **Training and Validation:** Training loop with learning rate scheduling and early stopping.
- **Evaluation Metrics:** Calculate MAE, MSE, R2, and Explained Variance.
- **Visualization:** Plot training/validation losses and evaluation metrics.
- **Logging:** Comprehensive logging for debugging and monitoring.
- **L1 Regularization:** Implemented to prevent overfitting.
- **Custom Message Passing Layer:** Provides the option for a user defined message passing layer.

## Installation

1.  **Clone the repository:**

    git clone https://github.com/shahram-boshra/n_class.git (or git clone git@github.com:shahram-boshra/n_class.git)
    cd n_class
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Prepare your dataset:** Place your molecular data (e.g., `.mol` files) in the directory specified in `config.yaml`.
2.  **Configure `config.yaml`:** Adjust the configuration parameters to match your dataset and desired settings.
3.  **Run the main script:**

    ```bash
    python main.py
    ```

## Configuration

The pipeline is configured using `config.yaml`. Key parameters include:

-   **`data`:** Dataset paths, split ratios, and caching options.
-   **`rdkit_processing`:** RDKit processing steps (hydrogenation, sanitization, etc.).
-   **`model`:** GNN model architecture, hyperparameters, and training settings.
-   **`logging`:** Logging configuration.

Example `config.yaml`:

```yaml
data:
  root_dir: "C:/Chem_Data"
  node_target_csv: "targets_n_class.csv"
  train_split: 0.8
  valid_split: 0.1
  use_cache: true

rdkit_processing:
  steps: ["HYDROGENATE", "SANITIZE", "KEKULIZE", "EMBED", "OPTIMIZE"]

model:
  first_layer_type: "gcn"
  second_layer_type: "gcn"
  hidden_channels: 64
  dropout_rate: 0.5
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.0001
  step_size: 50
  gamma: 0.5
  reduce_lr_factor: 0.5
  reduce_lr_patience: 10
  early_stopping_patience: 20
  early_stopping_delta: 0.0001
  l1_regularization_lambda: 0.0001

Dataset
Molecular data should be stored in the directory specified by data.root_dir.
Node-level target values should be provided in a CSV file specified by data.node_target_csv.

Model
The models.py module defines the MGModel class, which implements a configurable GNN model. You can choose from various GNN layer types and adjust hyperparameters.

Training
The training.py module handles the training and validation process. It includes:

Dataset splitting.
Training loop with learning rate scheduling.
Validation loop with evaluation metrics.
Early stopping.
Visualization of training progress.
Testing
The trained model is evaluated on the test set, and the results are logged and saved.

Dependencies
PyTorch
PyTorch Geometric
RDKit
NumPy
Scikit-learn
Matplotlib
PyYAML
Pandas

Contributing
Contributions are welcome! Please feel free to submit pull requests or open issues.

License
This project is licensed under the MIT License.   


Advanced Usage
Customizing GNN Layers: Modify the MGModel class in models.py to implement custom GNN layers or architectures.
Feature Engineering: Extend the MoleculeFeatureExtractor class in feature_extractor.py to add more features.
Hyperparameter Tuning: Use tools like Optuna or Ray Tune to optimize hyperparameters.
Data Augmentation: Implement data augmentation techniques to improve model generalization.

Troubleshooting
RDKit Errors: Ensure that RDKit is correctly installed and that the input molecules are valid.
CUDA Issues: Verify that CUDA is installed and that PyTorch is using the GPU.
Memory Errors: Reduce the batch size or use a more memory-efficient model.
Dataset Issues: Verify the path and content of the dataset files.

Future Enhancements
Support for more advanced graph representations.
Implementation of more sophisticated GNN architectures.
Integration with other molecular modeling tools.
Adding support for transfer learning.
Improving the speed of data loading.

Acknowledgements
The PyTorch Geometric team for their excellent library.
The RDKit community for their powerful cheminformatics toolkit.
The Scikit-learn developers for their machine learning library.