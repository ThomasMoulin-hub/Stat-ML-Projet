"""
Version alternative du préprocessing pour créer des sous-graphes locaux.
Chaque point d'entraînement est un graphe de 30 cellules (1 centrale + 29 voisins).
"""

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data, Batch
from scipy.sparse import issparse
from tqdm import trange


def build_local_subgraphs(features, spatial_coords, k=29, metric='cosine', seed=42):
    """
    Construit des sous-graphes locaux pour chaque cellule.
    Chaque sous-graphe contient la cellule cible + ses k voisins les plus proches.

    Args:
        features: Matrice de features (n_cells, n_features)
        spatial_coords: Coordonnées spatiales (n_cells, 2)
        k: Nombre de voisins
        metric: 'cosine' ou 'euclidean'
        seed: Pour reproductibilité

    Returns:
        subgraphs_list: Liste de Data objects (un par cellule)
        coords_scaler: Scaler pour dénormaliser
    """
    print(f"\nConstruction des sous-graphes locaux (k={k}, metric={metric})...")

    # Convertir features en numpy si nécessaire
    if issparse(features):
        features_array = features.toarray()
    else:
        features_array = np.array(features)

    n_cells = features_array.shape[0]

    # K-NN basé sur similarité d'expression
    nbrs = NearestNeighbors(n_neighbors=k+1, metric=metric, n_jobs=-1)
    nbrs.fit(features_array)

    # Trouver les voisins
    distances, indices = nbrs.kneighbors(features_array)

    # Normaliser les coordonnées
    coords_scaler = StandardScaler()
    spatial_coords_normalized = coords_scaler.fit_transform(spatial_coords)

    # Créer un sous-graphe pour chaque cellule
    subgraphs_list = []

    for i in trange(n_cells):
        # Cellule centrale = nœud 0
        # Voisins = nœuds 1 à k
        neighbors = indices[i, 1:k+1]  # Exclure la cellule elle-même

        # Indices des cellules dans ce sous-graphe (centrale + voisins)
        subgraph_cells = np.concatenate([[i], neighbors])

        # Features du sous-graphe (k+1 cellules)
        subgraph_features = features_array[subgraph_cells]

        # Coordonnées du sous-graphe
        subgraph_coords = spatial_coords_normalized[subgraph_cells]
        subgraph_coords_original = spatial_coords[subgraph_cells]

        # Créer les arêtes: cellule centrale (0) connectée à tous ses voisins (1 à k)
        edge_list = []
        for j in range(1, k+1):
            # Arête bidirectionnelle
            edge_list.append([0, j])  # centrale -> voisin
            edge_list.append([j, 0])  # voisin -> centrale

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        # Créer l'objet Data
        # La cible est UNIQUEMENT la position de la cellule centrale (nœud 0)
        data = Data(
            x=torch.tensor(subgraph_features, dtype=torch.float32),
            edge_index=edge_index,
            y=torch.tensor(subgraph_coords[0:1], dtype=torch.float32),  # Seulement cellule centrale
            pos_original=torch.tensor(subgraph_coords_original[0:1], dtype=torch.float32),
            central_idx=torch.tensor([i], dtype=torch.long)  # Index global de la cellule centrale
        )

        subgraphs_list.append(data)

    print(f"✓ {len(subgraphs_list)} sous-graphes créés")
    print(f"✓ Chaque sous-graphe: {k+1} nœuds, {edge_index.shape[1]} arêtes")
    print(f"✓ Chaque cible: position de la cellule centrale uniquement")

    return subgraphs_list, coords_scaler


def create_subgraph_splits(n_subgraphs, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Crée des indices de split pour les sous-graphes.

    Args:
        n_subgraphs: Nombre total de sous-graphes
        train_ratio, val_ratio, test_ratio: Proportions
        seed: Pour reproductibilité

    Returns:
        train_indices, val_indices, test_indices: Listes d'indices
    """
    np.random.seed(seed)
    indices = np.random.permutation(n_subgraphs)

    train_size = int(train_ratio * n_subgraphs)
    val_size = int(val_ratio * n_subgraphs)

    train_indices = indices[:train_size].tolist()
    val_indices = indices[train_size:train_size + val_size].tolist()
    test_indices = indices[train_size + val_size:].tolist()

    print(f"\n✓ Split: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")

    return train_indices, val_indices, test_indices

