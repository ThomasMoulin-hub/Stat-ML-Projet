"""
Architecture du modèle GNN avec GAT pour prédiction de coordonnées spatiales.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class SpatialGAT(nn.Module):
    """
    Graph Attention Network pour prédire les coordonnées spatiales des cellules.

    Architecture:
    - 2 couches GATConv avec attention multi-têtes
    - Dropout pour régularisation
    - MLP final pour prédiction des coordonnées (x, y)
    """

    def __init__(self, in_channels, hidden_channels=128, out_channels=2,
                 heads=4, dropout=0.4):
        """
        Args:
            in_channels: Nombre de features d'entrée (432 pour 405 gènes + 27 protéines)
            hidden_channels: Dimension des couches cachées
            out_channels: Dimension de sortie (2 pour x, y)
            heads: Nombre de têtes d'attention pour la première couche GAT
            dropout: Taux de dropout
        """
        super(SpatialGAT, self).__init__()

        self.dropout = dropout

        # Première couche GAT avec multi-head attention
        # heads * hidden_channels en sortie car chaque tête produit hidden_channels
        self.conv1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            dropout=dropout,
            concat=True  # Concatène les sorties de toutes les têtes
        )

        # Deuxième couche GAT (1 tête, dimension réduite)
        self.conv2 = GATConv(
            hidden_channels * heads,  # Entrée = sortie de conv1
            64,
            heads=1,
            dropout=dropout,
            concat=False
        )

        # MLP final pour prédire les coordonnées
        self.mlp = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, out_channels)
        )

    def forward(self, x, edge_index):
        """
        Forward pass du modèle.

        Args:
            x: Features des nœuds (n_nodes, in_channels)
            edge_index: Liste d'arêtes (2, n_edges)

        Returns:
            out: Coordonnées prédites (n_nodes, 2)
        """
        # Première couche GAT + activation + dropout
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Deuxième couche GAT + activation + dropout
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # MLP final pour prédire (x, y)
        out = self.mlp(x)

        return out

    def count_parameters(self):
        """Compte le nombre de paramètres entraînables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SpatialGATLarge(nn.Module):
    """
    Version plus profonde du modèle avec 3 couches GAT.
    À utiliser si le modèle de base sous-performe.
    """

    def __init__(self, in_channels, hidden_channels=128, out_channels=2,
                 heads=4, dropout=0.4):
        super(SpatialGATLarge, self).__init__()

        self.dropout = dropout

        # Trois couches GAT
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads,
                            dropout=dropout, concat=True)
        self.conv2 = GATConv(hidden_channels * heads, 96, heads=2,
                            dropout=dropout, concat=True)
        self.conv3 = GATConv(96 * 2, 64, heads=1,
                            dropout=dropout, concat=False)

        # MLP final
        self.mlp = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, out_channels)
        )

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.elu(self.conv3(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)

        out = self.mlp(x)
        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(in_channels, model_type='base', **kwargs):
    """
    Factory function pour créer le modèle.

    Args:
        in_channels: Nombre de features d'entrée
        model_type: 'base' ou 'large'
        **kwargs: Arguments supplémentaires (hidden_channels, heads, dropout, etc.)

    Returns:
        model: Instance du modèle
    """
    if model_type == 'base':
        model = SpatialGAT(in_channels, **kwargs)
    elif model_type == 'large':
        model = SpatialGATLarge(in_channels, **kwargs)
    else:
        raise ValueError(f"Type de modèle inconnu: {model_type}")

    print(f"\n✓ Modèle créé: {model.__class__.__name__}")
    print(f"  Paramètres entraînables: {model.count_parameters():,}")

    return model

