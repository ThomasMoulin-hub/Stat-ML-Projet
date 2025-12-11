# Spatial GNN for Xenium data — Local Subgraph Training with Joint Encoder

This repository implements a spatial Graph Neural Network (GAT) to predict cell spatial coordinates using only molecular profiles (RNA + protein) from 10x Xenium outputs. The graph is constructed from expression similarity (K-NN) rather than physical proximity.

- Local subgraphs: one subgraph per cell (central + k neighbors). The model predicts the central cell’s coordinates while learning over its local neighborhood. Batching across many subgraphs enables scalable training and DDP.


## Key features
- Xenium data ingestion via `spatialdata_io` and AnnData processing.
- Modality-aware Joint Encoder that splits RNA and protein features, then fuses representations before GAT.
- GAT stack with residual connections, dropout, to produce 2D spatial coordinates.
- Local subgraph construction per cell: k neighbors by expression similarity, bidirectional star edges to the central node.
- Distributed Data Parallel (DDP) training using PyTorch with `torch.multiprocessing.spawn` and NCCL backend.
- Comprehensive evaluation: MSE/MAE, R² per axis, Euclidean error distribution, prediction vs truth plots, and spatial error heatmaps.
- Caching of preprocessed subgraphs, scalers, and splits to accelerate reproducible runs.


## Repository structure
- `main.py`: Entry point for local subgraph pipeline with DDP training and evaluation.
- `data_preprocessing.py`: AnnData filtering and normalization; optional global K-NN graph creation.
- `data_preprocessing_subgraph.py`: Local subgraph builder per cell and train/val/test split creation.
- `model_joint_encoder.py`: Joint Encoder architectures (base/large) with optional cross-modal attention; GAT stacks with residuals and pooling.
- `model.py`: Simpler GAT baselines (base/large) without joint modality separation.
- `train_subgraph.py`: Trainer for batched local subgraphs; supports single-GPU and DDP; smoothing regularizer.
- `evaluate.py`: Metrics and plotting utilities for training curves, prediction scatter, spatial visualization, and error analysis.
- `requirements.txt`: Python dependencies.
- `data/`: Datasets (Xenium and other sources as laid out below).
- `cache_*`: Cached subgraphs, scalers, splits, metadata.
- `results_*` and `results/`: Saved checkpoints, metrics, and plots.


## Data requirements and layout
The default dataset in `main.py` expects Xenium outputs:
- `data/Xenium_V1_Human_Kidney_FFPE_Protein_updated_outs/`

Other folders present (not required for the default run):
- `data/Xenium_Prime_Human_Ovary_FF_outs/`
- `data/LiverDataReleaseTileDB/` (TileDB layout)
- `data/breast.zarr/` (Zarr layout)
- `data/GSE158013_RAW/` (scATAC-related inputs)

The pipeline uses `spatialdata_io.xenium(...)` with options to include gene and protein modalities and to represent cells as circles. The main AnnData table is available at `sdata.tables["table"]`. Features are filtered to keep only `var["feature_types"] in {"Gene Expression", "Protein Expression"}`.


## Preprocessing details
Implemented in `data_preprocessing.py::preprocess_adata`:
- Gene features: Scanpy `normalize_total(target_sum=1e4)` followed by `log1p`.
- Protein features: StandardScaler z-score per feature.
- Sparse matrices: converted to dense up front for simplicity.

Local subgraph construction in `data_preprocessing_subgraph.py::build_local_subgraphs`:
- Similarity: scikit-learn `NearestNeighbors(n_neighbors=k+1, metric=cosine|euclidean, n_jobs=-1)` fits on normalized features.
- For each cell i, build nodes: [central=i, neighbors=indices[i, 1:k+1]].
- Edges: bidirectional star from central (0) to each neighbor (1..k).
- Targets: normalized spatial coordinates for all nodes in the subgraph (supervision over all nodes), with `y_central` kept for compatibility.
- Normalization: StandardScaler fit on global spatial coordinates; inverse transform used at evaluation if requested.
- Returns: `List[torch_geometric.data.Data]`, each with `x`, `edge_index`, `y`, `y_central`, `pos_original`, and `central_idx`.

Train/val/test split creation in `create_subgraph_splits`: random permutation with ratios (default 70/15/15) and fixed seed.


## Model architectures
Joint Encoder models (`model_joint_encoder.py`):
- Split input features into RNA (n_genes) and protein (n_proteins).
- Independent encoders with LayerNorm, ReLU, Dropout; configurable dimensions.
- Fusion MLP to produce a joint representation fed to GAT layers.
- GAT stack with residual projections to stabilize deep attention layers.


Baseline GAT models (`model.py`): simpler two- or three-layer GAT with a final MLP; no modality separation.

Losses and regularizers (`train_subgraph.py`):
- Main loss: MSE on predicted vs normalized coordinates for all nodes in the batch.
- MAE tracked for reporting; ReduceLROnPlateau scheduler on validation loss.


## Training pipeline (DDP local subgraphs)
The main entry `main.py` orchestrates:
1. Resource setup: increase file descriptor limits and set PyTorch’s sharing strategy to `file_system`.
2. Load Xenium dataset via `spatialdata_io.xenium(...)`; get AnnData table.
3. Preprocess features with gene/protein normalization.
4. Cache handling: if cached `subgraphs.pt`, `scaler.pkl`, and `metadata.json` exist, reuse them; else build and cache.
5. Derive `n_genes` and `n_proteins` from `adata.var["feature_types"]` counts.
6. Create or load train/val/test splits; save to JSON for reproducibility.
7. Determine `in_channels` and select architecture; default uses Joint Encoder with parameters in `create_joint_encoder_model(...)`.
8. Spawn DDP processes via `torch.multiprocessing.spawn` with `world_size=4`.
9. In each rank:
   - Initialize `torch.distributed` with NCCL backend and bind rank to `cuda:rank`.
   - Load cached subgraphs and scaler locally.
   - Build `SubgraphTrainer` with `DistributedSampler`s and DDP-wrapped model.
   - Train with early stopping and scheduler.
10. On rank 0 only:
    - Save checkpoint to `results/spatial_gat_model.pt`.
    - Plot training history.
    - Predict on test set, denormalize if requested.
    - Compute metrics; save `results/test_metrics.csv`.
    - Save plots: predictions vs truth, spatial predictions, error distribution; print extreme errors summary.



## Outputs
On successful training (rank 0):
- `results/spatial_gat_model.pt`: checkpoint with model state, optimizer state, and training history.
- `results/training_history.png`: loss and MAE curves for train/val.
- `results/predictions_vs_true.png`: scatter plots for X and Y predictions.
- `results/spatial_predictions.png`: spatial scatter of true vs predicted and error heatmap.
- `results/error_distribution.png`: histogram and boxplot of Euclidean errors.
- `results/test_metrics.csv`: MSE, MAE, R² per axis, and Euclidean distance statistics.


## Practical notes
- Normalization: coordinate scaler is fit globally; predictions are trained in normalized space and denormalized for reporting.
- Reproducibility: splits and metadata are persisted to cache; seeds applied in split creation.