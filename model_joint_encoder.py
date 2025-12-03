"""
Architecture simplifiée avec encodeur ARN uniquement (sans protéines).

Cette architecture traite uniquement les données transcriptomiques (gènes/ARN).
Les protéines et la cross-attention ont été retirées pour se concentrer sur l'ARN.

Avantages de cette version simplifiée:
- Moins de paramètres à entraîner
- Plus rapide à l'entraînement
- Focus sur les patterns ARN uniquement
- Évite la complexité de la fusion multi-modale
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class JointEncoder(nn.Module):
    """
    Encodeur pour l'ARN uniquement (simplifié, sans protéines).

    Cette architecture traite uniquement les données transcriptomiques (gènes/ARN).
    Les protéines et la cross-attention ont été retirées.
    """

    def __init__(self, n_genes, rna_hidden=128, dropout=0.3):
        """
        Args:
            n_genes: Nombre de gènes (405 par exemple)
            rna_hidden: Dimension cachée pour l'encodeur ARN
            dropout: Taux de dropout
        """
        super(JointEncoder, self).__init__()

        self.n_genes = n_genes

        # Encodeur ARN (architecture profonde pour bien capturer les patterns)
        self.rna_encoder = nn.Sequential(
            nn.Linear(n_genes, rna_hidden),
            nn.LayerNorm(rna_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(rna_hidden, rna_hidden),
            nn.LayerNorm(rna_hidden),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Args:
            x: Features ARN uniquement (n_nodes, n_genes)

        Returns:
            rna_encoded: Représentation encodée (n_nodes, rna_hidden)
        """
        # x contient maintenant uniquement l'ARN (pas de protéines)
        rna_encoded = self.rna_encoder(x)

        return rna_encoded


class SpatialGATWithJointEncoder(nn.Module):
    """
    GAT avec encodeur pour ARN uniquement (sans protéines).

    Architecture:
    1. Encoder ARN: Traite les données transcriptomiques
    2. GAT layers: Message passing sur le graphe (avec connexions résiduelles)
    3. MLP final: Prédiction des coordonnées
    """

    def __init__(self, n_genes,
                 rna_hidden=128,
                 gat_hidden=128, out_channels=2, heads=4, dropout=0.4):
        """
        Args:
            n_genes: Nombre de gènes
            rna_hidden: Dimension cachée encodeur ARN
            gat_hidden: Dimension cachée des couches GAT
            out_channels: Dimension sortie (2 pour x,y)
            heads: Nombre de têtes d'attention
            dropout: Taux de dropout
        """
        super(SpatialGATWithJointEncoder, self).__init__()

        self.dropout = dropout

        # Encoder ARN uniquement (pas de protéines)
        self.joint_encoder = JointEncoder(
            n_genes=n_genes,
            rna_hidden=rna_hidden,
            dropout=dropout,
        )

        # Couches GAT sur la représentation jointe
        self.conv1 = GATConv(
            rna_hidden,
            gat_hidden,
            heads=heads,
            dropout=dropout,
            concat=True
        )

        # Projection pour la connexion résiduelle (ajuster les dimensions)
        self.residual1 = nn.Linear(rna_hidden, gat_hidden * heads)

        self.conv2 = GATConv(
            gat_hidden * heads,
            64,
            heads=1,
            dropout=dropout,
            concat=False
        )

        # Projection pour la connexion résiduelle
        self.residual2 = nn.Linear(gat_hidden * heads, 64)

        # MLP final
        mlp_input_dim = 64
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, out_channels)
        )

    def forward(self, data):
        """
        Forward pass avec encodage de l'ARN uniquement.

        Args:
            data: PyG Data object contenant:
                - x: Features ARN uniquement (n_nodes, n_genes)
                - edge_index: Liste d'arêtes (2, n_edges)
                - batch: Indices de batch (n_nodes,)

        Returns:
            out: Coordonnées prédites (n_nodes, 2)
        """
        # 1. Encoder l'ARN
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.joint_encoder(x)

        # 2. Message passing avec GAT + connexions résiduelles
        identity = x
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # Connexion résiduelle: ajouter l'identité projetée
        x = x + self.residual1(identity)

        identity = x
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # Connexion résiduelle
        x = x + self.residual2(identity)

        # 3. Prédiction finale
        out = self.mlp(x)

        return out

    def count_parameters(self):
        """Compte le nombre de paramètres entraînables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SpatialGATLargeWithJointEncoder(nn.Module):
    """
    Version plus profonde avec encodeur ARN uniquement + 3 couches GAT.

    Architecture avec connexions résiduelles (sans protéines).
    """

    def __init__(self, n_genes,
                 rna_hidden=256,
                 gat_hidden=256, out_channels=2, heads=4, dropout=0.4):
        super(SpatialGATLargeWithJointEncoder, self).__init__()

        self.dropout = dropout

        # Encoder ARN uniquement (pas de protéines)
        self.joint_encoder = JointEncoder(
            n_genes=n_genes,
            rna_hidden=rna_hidden,
            dropout=dropout,
        )

        # Trois couches GAT avec connexions résiduelles
        self.conv1 = GATConv(rna_hidden, gat_hidden, heads=heads,
                            dropout=dropout, concat=True)
        self.residual1 = nn.Linear(rna_hidden, gat_hidden * heads)

        self.conv2 = GATConv(gat_hidden * heads, 96, heads=2,
                            dropout=dropout, concat=True)
        self.residual2 = nn.Linear(gat_hidden * heads, 96 * 2)

        self.conv3 = GATConv(96 * 2, 128, heads=1,
                            dropout=dropout, concat=False)
        self.residual3 = nn.Linear(96 * 2, 128)

        # MLP final
        mlp_input_dim = 128
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_channels)
        )

    def forward(self, data):
        """
        Forward pass avec connexions résiduelles.

        Args:
            data: PyG Data object contenant:
                - x: Features ARN uniquement (n_nodes, n_genes)
                - edge_index: Liste d'arêtes (2, n_edges)
                - batch: Indices de batch (n_nodes,)

        Returns:
            out: Coordonnées prédites (n_nodes, 2)
        """

        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Encoder l'ARN uniquement
        x = self.joint_encoder(x)

        # GAT layers avec connexions résiduelles
        identity = x
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + self.residual1(identity)  # Skip connection

        identity = x
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + self.residual2(identity)  # Skip connection

        identity = x
        x = F.elu(self.conv3(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + self.residual3(identity)  # Skip connection

        # Prédiction finale
        out = self.mlp(x)
        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_joint_encoder_model(n_genes, model_type='base', **kwargs):
    """
    Factory function pour créer un modèle avec encodeur ARN uniquement.

    Args:
        n_genes: Nombre de gènes
        model_type: 'base' ou 'large'
        **kwargs: Arguments supplémentaires (rna_hidden, gat_hidden, heads, dropout, etc.)

    Returns:
        model: Instance du modèle
    """
    if model_type == 'base':
        model = SpatialGATWithJointEncoder(
            n_genes=n_genes,
            **kwargs
        )
    elif model_type == 'large':
        model = SpatialGATLargeWithJointEncoder(
            n_genes=n_genes,
            **kwargs
        )
    else:
        raise ValueError(f"Type de modèle inconnu: {model_type}")

    print(f"\n✓ Modèle avec encodeur ARN créé: {model.__class__.__name__}")
    print(f"  • {n_genes} gènes → encodeur ARN")
    print(f"  • Protéines: Aucune (désactivées)")
    print(f"  • Cross-modal attention: Désactivée (pas de protéines)")
    print(f"  • Connexions résiduelles: Oui (dans GAT layers)")
    print(f"  • Paramètres entraînables: {model.count_parameters():,}")

    return model

