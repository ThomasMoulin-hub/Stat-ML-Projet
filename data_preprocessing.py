"""
Module de préparation des données pour le GNN spatial.
Gère la normalisation, le filtrage et la construction du graphe K-NN.
"""

import numpy as np
import scanpy as sc
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from scipy.sparse import issparse


def preprocess_adata(adata, normalize_genes=True,
                     min_genes_per_cell=200, min_cells_per_gene=3):
    """
    Filtre et normalise les données d'expression.

    Args:
        adata: AnnData object contenant les données Xenium
        normalize_genes: Si True, applique normalize_total + log1p aux gènes
        normalize_proteins: Ignoré (pour compatibilité)
        min_genes_per_cell: Nombre minimum de gènes détectés par cellule
        min_cells_per_gene: Nombre minimum de cellules exprimant un gène

    Returns:
        adata_processed: AnnData avec seulement Gene Expression (RNA)
    """
    print(f"📊 Données initiales: {adata.n_obs} cellules, {adata.n_vars} features")

    # Créer le masque pour ne garder que les gènes
    gene_mask = adata.var["feature_types"] == "Gene Expression"

    # Filtrer pour ne garder QUE les gènes (pas de protéines)
    adata_filtered = adata[:, gene_mask].copy()

    print(f"Après filtrage des protéines: {adata_filtered.n_obs} cellules, {adata_filtered.n_vars} gènes")

    # Convertir en dense dès le début pour simplifier
    if issparse(adata_filtered.X):
        adata_filtered.X = adata_filtered.X.toarray()

    # 🔍 FILTRAGE DE QUALITÉ
    print(f"\n🔍 Filtrage de qualité...")
    print(f"   • Seuil cellules: min {min_genes_per_cell} gènes détectés")
    print(f"   • Seuil gènes: min {min_cells_per_gene} cellules")

    # Filtrer les cellules avec peu de gènes détectés
    n_genes_per_cell = (adata_filtered.X > 0).sum(axis=1)
    cell_mask = n_genes_per_cell >= min_genes_per_cell
    adata_filtered = adata_filtered[cell_mask, :].copy()
    print(f"   ✓ Cellules après filtrage: {adata_filtered.n_obs} ({cell_mask.sum()}/{len(cell_mask)})")

    # Filtrer les gènes exprimés dans peu de cellules
    n_cells_per_gene = (adata_filtered.X > 0).sum(axis=0)
    gene_filter_mask = n_cells_per_gene >= min_cells_per_gene
    adata_filtered = adata_filtered[:, gene_filter_mask].copy()
    print(f"   ✓ Gènes après filtrage: {adata_filtered.n_vars} ({gene_filter_mask.sum()}/{len(gene_filter_mask)})")

    print(f"\n📊 Données après filtrage de qualité:")
    print(f"   • Cellules: {adata_filtered.n_obs}")
    print(f"   • Gènes (RNA): {adata_filtered.n_vars}")
    print(f"   • Total features: {adata_filtered.n_vars}")

    # Normalisation des gènes
    if normalize_genes:
        # Normalisation scanpy (total counts + log1p)
        sc.pp.normalize_total(adata_filtered, target_sum=1e4)
        sc.pp.log1p(adata_filtered)
        print("✓ Gènes normalisés (normalize_total + log1p)")

    # Pas de normalisation de protéines puisqu'elles sont exclues

    return adata_filtered


def build_knn_graph(features, spatial_coords, k=29, metric='cosine'):
    """
    Construit un graphe K-NN basé sur la similarité d'expression.

    Args:
        features: Matrice de features normalisées (n_cells, n_features)
        spatial_coords: Coordonnées spatiales réelles (n_cells, 2) pour les labels
        k: Nombre de voisins
        metric: 'cosine' ou 'euclidean'

    Returns:
        data: Objet torch_geometric.data.Data avec le graphe et les labels
        coords_scaler: Scaler pour dénormaliser les coordonnées
    """
    print(f"\nConstruction du graphe K-NN (k={k}, metric={metric})...")

    # Convertir features en numpy si nécessaire
    if issparse(features):
        features_array = features.toarray()
    else:
        features_array = np.array(features)

    # K-NN basé sur similarité d'expression (pas les coordonnées!)
    nbrs = NearestNeighbors(n_neighbors=k+1, metric=metric, n_jobs=-1)
    nbrs.fit(features_array)

    # Trouver les voisins (k+1 car le premier voisin est la cellule elle-même)
    distances, indices = nbrs.kneighbors(features_array)

    # Construire la liste d'arêtes (exclure self-loops)
    edge_list = []
    for i in range(len(indices)):
        for j in range(1, k+1):  # Commencer à 1 pour éviter self-loop
            edge_list.append([i, indices[i, j]])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    print(f"✓ Graphe créé: {features_array.shape[0]} nœuds, {edge_index.shape[1]} arêtes")

    # Normaliser les coordonnées spatiales (z-score)
    coords_scaler = StandardScaler()
    spatial_coords_normalized = coords_scaler.fit_transform(spatial_coords)

    # Créer l'objet Data PyTorch Geometric
    data = Data(
        x=torch.tensor(features_array, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(spatial_coords_normalized, dtype=torch.float32),
        pos_original=torch.tensor(spatial_coords, dtype=torch.float32)
    )

    print(f"✓ Features shape: {data.x.shape}")
    print(f"✓ Coordonnées normalisées: mean={spatial_coords_normalized.mean():.3f}, std={spatial_coords_normalized.std():.3f}")

    return data, coords_scaler


def create_train_val_test_masks(n_nodes, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Crée des masques pour split train/val/test.

    Args:
        n_nodes: Nombre total de nœuds (cellules)
        train_ratio: Proportion de données d'entraînement
        val_ratio: Proportion de données de validation
        test_ratio: Proportion de données de test
        seed: Seed pour reproductibilité

    Returns:
        train_mask, val_mask, test_mask: Tenseurs booléens
    """
    np.random.seed(seed)
    indices = np.random.permutation(n_nodes)

    train_size = int(train_ratio * n_nodes)
    val_size = int(val_ratio * n_nodes)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    print(f"\n✓ Split: Train={train_mask.sum()}, Val={val_mask.sum()}, Test={test_mask.sum()}")

    return train_mask, val_mask, test_mask