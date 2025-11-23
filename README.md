# Spatial Transcriptomics Prediction with Graph Neural Networks

A Graph Attention Network (GAT) that predicts spatial cell coordinates from gene and protein expression profiles, using Xenium spatial transcriptomics data.

## Overview

This project implements a GNN-based approach to predict cell locations based on their molecular profiles (RNA + proteins), demonstrating that spatial organization correlates with gene expression patterns. The model uses **local subgraph architecture**: each training point is a subgraph of 30 cells (1 target + 29 neighbors).

## Key Features

- **Joint Encoder Architecture**: Separate encoders for RNA and protein data with optional cross-modal attention
- **Local Subgraph Approach**: Each cell is predicted using only its k-nearest neighbors (based on expression similarity)
- **K-NN Graph Construction**: Graph edges based on gene expression similarity (cosine distance), not spatial proximity
- **Efficient Caching**: Pre-computed subgraphs and data splits for faster training iterations

## Architecture

```
Input: RNA (genes) + Protein features per cell
   ↓
[RNA Encoder] + [Protein Encoder]
   ↓
[Cross-Modal Attention] (optional)
   ↓
[Joint Representation]
   ↓
[GAT Layers] - Graph convolution over k-NN graph
   ↓
Output: Predicted (x, y) spatial coordinates
```

## Installation

```bash
pip install -r requirements.txt
```

**Requirements**: PyTorch, PyTorch Geometric, SpatialData, scikit-learn, pandas, matplotlib

## Usage

```bash
python main.py
```

The pipeline automatically:
1. Loads Xenium kidney data from `./data/Xenium_V1_Human_Kidney_FFPE_Protein_updated_outs/`
2. Preprocesses and normalizes gene/protein expression
3. Builds local subgraphs (k=49 neighbors, cosine similarity)
4. Trains the GAT model with early stopping
5. Evaluates on test set and saves results to `./results/`

## Project Structure

- **`main.py`**: Main training pipeline
- **`model_joint_encoder.py`**: Joint encoder architecture with cross-modal attention
- **`model.py`**: Standard GAT model (all features concatenated)
- **`data_preprocessing.py`**: Data filtering and normalization
- **`data_preprocessing_subgraph.py`**: Local subgraph construction
- **`train_subgraph.py`**: Training loop for subgraph-based models
- **`evaluate.py`**: Evaluation metrics and visualization

## Model Variants

### 1. Joint Encoder (Recommended)
```python
use_joint_encoder = True  # Separate RNA/protein encoders
```
- Better captures modality-specific patterns
- Prevents proteins (fewer features) from being dominated by genes
- Optional cross-modal attention mechanism

### 2. Standard Model
```python
use_joint_encoder = False  # Concatenated features
```
- Simpler baseline approach
- All features processed together

## Hyperparameters

```python
k_value = 49              # Number of neighbors per subgraph
metric_value = 'cosine'   # Distance metric for K-NN
batch_size = 300          # Subgraphs per batch
epochs = 200              # Maximum training epochs
early_stopping = 20       # Patience for early stopping
lr = 0.001                # Learning rate
dropout = 0.4             # Dropout rate
```

## Output

Results saved to `./results/`:
- `training_history.png`: Loss curves
- `predictions_vs_true.png`: Scatter plots of predicted vs true coordinates
- `spatial_predictions.png`: Spatial visualization with error magnitude
- `error_distribution.png`: Euclidean distance distribution
- `test_metrics.csv`: MAE, RMSE, R² scores
- `spatial_gat_model.pt`: Trained model weights

## Performance Metrics

The model is evaluated using:
- **MAE** (Mean Absolute Error): Average coordinate error
- **RMSE** (Root Mean Squared Error): Penalizes large errors
- **R²**: Goodness of fit (per dimension)
- **Euclidean Distance**: Physical distance between predicted and true positions

## Data

Expected data structure:
```
data/
  Xenium_V1_Human_Kidney_FFPE_Protein_updated_outs/
    (Xenium output files)
```

The code uses Xenium spatial transcriptomics data with:
- Gene expression matrix
- Protein expression matrix
- Spatial coordinates (ground truth)

## Caching

Subgraph construction is cached in `cache_Xenium_V1_Human_Kidney_FFPE_Protein_updated_outs/`:
- `subgraphs_k{k}_metric_{metric}.pt`: Pre-computed subgraphs
- `subgraphs_k{k}_metric_{metric}_scaler.pkl`: Coordinate scaler
- `subgraphs_k{k}_metric_{metric}_splits.json`: Train/val/test splits

Delete cache folder to rebuild from scratch.