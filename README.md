# DumplingGNN: Hybrid GNN for ADC Payload Activity Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![GitHub stars](https://img.shields.io/github/stars/Iayce/DumplingGnn.svg)](https://github.com/Iayce/DumplingGnn/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/Iayce/DumplingGnn.svg)](https://github.com/Iayce/DumplingGnn/issues)

This repository contains the implementation of DumplingGNN, a hybrid Graph Neural Network designed to predict ADC (Antibody-Drug Conjugate) payload activity based on chemical structure.

## Overview

DumplingGNN is a novel hybrid Graph Neural Network architecture that combines multiple GNN components:
- Message Passing Neural Networks (MPNN) for capturing local chemical interactions
- Graph Attention Networks (GAT) for identifying important substructures
- GraphSAGE for aggregating information across different scales

The model was developed to effectively capture multi-scale molecular features using both 2D topological and 3D structural information, providing accurate predictions and interpretable results.

## Repository Contents

- `model.py`: Implementation of the DumplingGNN model and baseline models (MPNN, GCN, GAT, GraphSAGE)
- `explainmodel.py`: Extended implementation with explainability features
- `train_explainable.py`: Training script for the explainable version of DumplingGNN
- `baselines.ipynb`: Jupyter notebook for running baseline model comparisons

### Model vs ExplainModel

The repository contains two versions of DumplingGNN:

1. **Standard Version (model.py)**:
   - Basic implementation focused on prediction performance
   - More efficient for training and inference
   - Includes baseline model implementations
   - Suitable for production deployment

2. **Explainable Version (explainmodel.py)**:
   - Enhanced with interpretability features
   - Provides attention visualization capabilities
   - Enables substructure identification
   - Includes tools for analyzing attention patterns
   - Better for research and analysis purposes

### Baseline Models

The repository includes several baseline models for comparison:

1. **FIVEMPNN**:
   - Five-layer Message Passing Neural Network
   - Focuses on local chemical interactions
   - Serves as baseline for pure message-passing approach

2. **GCN (Graph Convolutional Network)**:
   - Five-layer implementation
   - Standard graph convolution operations
   - Baseline for basic graph learning

3. **FiveLayerGAT**:
   - Five-layer Graph Attention Network
   - Tests pure attention-based approach
   - Baseline for attention mechanism effectiveness

4. **FiveLayerSAGE**:
   - Five-layer GraphSAGE implementation
   - Tests neighborhood aggregation strategy
   - Baseline for sampling-based approaches

These baselines help evaluate the advantages of DumplingGNN's hybrid architecture.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/Iayce/DumplingGnn.git
cd DumplingGnn
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Note: This requires Python 3.10 or higher.

## Usage

### Preparing Data

The model expects molecular data in SDF format, with each molecule having a "label" property indicating its activity. The training script processes this data into graph representations suitable for the model.

#### Dataset Description: train_chembl.sdf

The `train_chembl.sdf` dataset contains a curated collection of molecules with DNA Topoisomerase I inhibitory activity:

- **Source**: Molecules collected from ChEMBL database (CHEMBL1781)
- **Size**: 190 unique molecular structures after processing
- **Activity Labels**: Binary classification
  - Positive: IC50 < 100 µM
  - Negative: IC50 ≥ 100 µM
- **Structure Information**:
  - 2D topological information (SMILES)
  - 3D conformations generated using OpenBabel and MMFF94 force field
  - Top 3 docking poses per molecule from NLDock
- **Quality Control**:
  - Removed redundant conformers (RMSD cutoff 1.5 Å)
  - Verified structural integrity
  - Standardized activity measurements

### Training a Model

To train the explainable version of DumplingGNN:

```bash
python train_explainable.py
```

By default, this will:
1. Load and process molecules from `train_chembl.sdf`
2. Split them into training, validation, and test sets
3. Train the model with early stopping
4. Save the best model to `saved_models/best_model.pth`
5. Display training metrics and performance on the test set

### Using the Trained Model

```python
import torch
from explainmodel import ExplainableDumplingGNN

# Initialize the model
model = ExplainableDumplingGNN(hidden_channels=32)

# Load trained weights
model.load_state_dict(torch.load('saved_models/best_model.pth'))

# Set to evaluation mode
model.eval()

# Pass your data through the model
output = model(data)
```

### Interpreting Predictions

The explainable version of DumplingGNN allows you to analyze the attention weights to understand which parts of a molecule contribute most to its predicted activity:

```python
from explainmodel import ExplainableDumplingGNNExplainer
from rdkit import Chem

# Initialize the explainer
explainer = ExplainableDumplingGNNExplainer(model)

# Load a molecule
mol = Chem.MolFromSmiles('SMILES_STRING')

# Analyze the molecule
results = explainer.analyze_case(data, mol)

# The results contain attention weights, important substructures, etc.
```

### Running Baseline Comparisons

To compare DumplingGNN with baseline models:

```python
from model import FIVEMPNN, GCN, FiveLayerGAT, FiveLayerSAGE

# Initialize models
models = {
    'MPNN': FIVEMPNN(in_channels=8, hidden_channels=32),
    'GCN': GCN(hidden_channels=32, num_classes=2),
    'GAT': FiveLayerGAT(in_channels=8, hidden_channels=32, out_channels=2),
    'SAGE': FiveLayerSAGE(in_channels=8, hidden_channels=32, out_channels=2)
}

# Train and evaluate each model
for name, model in models.items():
    train_model(model, train_loader)
    results = evaluate_model(model, test_loader)
    print(f"{name} Results:", results)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This work was supported by the National Key Research and Development Program of China under Grant 2023YFC3304501. 