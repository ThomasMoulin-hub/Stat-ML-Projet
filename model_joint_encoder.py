"""
Architecture améliorée avec Joint Encoder pour ARN et Protéines.

Cette architecture traite séparément les deux modalités biologiques (transcriptomique
et protéomique) avant de les fusionner et de les passer dans le GNN.

Avantages:
- Capture mieux les patterns spécifiques à chaque modalité
- Évite que les protéines (moins nombreuses) soient noyées par les gènes
- Permet d'appliquer des normalisations/transformations spécifiques
- Meilleure interprétabilité des représentations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class JointEncoder(nn.Module):
    """
    Encodeur séparé pour ARN et protéines avant fusion.

    Cette architecture traite séparément les modalités biologiques différentes
    (transcriptomique vs protéomique) avant de les combiner.
    """

    def __init__(self, n_genes, n_proteins, rna_hidden=128, protein_hidden=64,
                 joint_hidden=128, dropout=0.3):
        """
        Args:
            n_genes: Nombre de gènes (405 par exemple)
            n_proteins: Nombre de protéines (27 par exemple)
            rna_hidden: Dimension cachée pour l'encodeur ARN
            protein_hidden: Dimension cachée pour l'encodeur protéines
            joint_hidden: Dimension de la représentation jointe
            dropout: Taux de dropout
        """
        super(JointEncoder, self).__init__()

        self.n_genes = n_genes
        self.n_proteins = n_proteins

        # Encodeur ARN (plus de capacité car plus de features)
        self.rna_encoder = nn.Sequential(
            nn.Linear(n_genes, rna_hidden),
            nn.BatchNorm1d(rna_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(rna_hidden, rna_hidden),
            nn.BatchNorm1d(rna_hidden),
            nn.ReLU()
        )

        # Encodeur protéines (plus petit car moins de features)
        self.protein_encoder = nn.Sequential(
            nn.Linear(n_proteins, protein_hidden),
            nn.BatchNorm1d(protein_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(protein_hidden, protein_hidden),
            nn.BatchNorm1d(protein_hidden),
            nn.ReLU()
        )

        # Fusion des deux modalités
        self.fusion = nn.Sequential(
            nn.Linear(rna_hidden + protein_hidden, joint_hidden),
            nn.BatchNorm1d(joint_hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Args:
            x: Features concaténées [ARN | Protéines] (n_nodes, n_genes + n_proteins)

        Returns:
            joint_repr: Représentation jointe (n_nodes, joint_hidden)
        """
        # Séparer ARN et protéines
        rna = x[:, :self.n_genes]
        proteins = x[:, self.n_genes:]

        # Encoder séparément
        rna_encoded = self.rna_encoder(rna)
        protein_encoded = self.protein_encoder(proteins)

        # Concaténer et fusionner
        combined = torch.cat([rna_encoded, protein_encoded], dim=1)
        joint_repr = self.fusion(combined)

        return joint_repr


class SpatialGATWithJointEncoder(nn.Module):
    """
    GAT avec encodeur séparé pour ARN et protéines.

    Architecture améliorée:
    1. Joint Encoder: Traite séparément ARN et protéines puis fusionne
    2. GAT layers: Message passing sur le graphe
    3. MLP final: Prédiction des coordonnées
    """

    def __init__(self, n_genes, n_proteins,
                 rna_hidden=128, protein_hidden=64, joint_hidden=128,
                 gat_hidden=128, out_channels=2, heads=4, dropout=0.4):
        """
        Args:
            n_genes: Nombre de gènes
            n_proteins: Nombre de protéines
            rna_hidden: Dimension cachée encodeur ARN
            protein_hidden: Dimension cachée encodeur protéines
            joint_hidden: Dimension représentation jointe (input du GAT)
            gat_hidden: Dimension cachée des couches GAT
            out_channels: Dimension sortie (2 pour x,y)
            heads: Nombre de têtes d'attention
            dropout: Taux de dropout
        """
        super(SpatialGATWithJointEncoder, self).__init__()

        self.dropout = dropout

        # Joint Encoder pour séparer les modalités
        self.joint_encoder = JointEncoder(
            n_genes=n_genes,
            n_proteins=n_proteins,
            rna_hidden=rna_hidden,
            protein_hidden=protein_hidden,
            joint_hidden=joint_hidden,
            dropout=dropout
        )

        # Couches GAT sur la représentation jointe
        self.conv1 = GATConv(
            joint_hidden,
            gat_hidden,
            heads=heads,
            dropout=dropout,
            concat=True
        )

        self.conv2 = GATConv(
            gat_hidden * heads,
            64,
            heads=1,
            dropout=dropout,
            concat=False
        )

        # MLP final
        self.mlp = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, out_channels)
        )

    def forward(self, x, edge_index):
        """
        Forward pass avec encodage séparé des modalités.

        Args:
            x: Features [ARN | Protéines] (n_nodes, n_genes + n_proteins)
            edge_index: Liste d'arêtes (2, n_edges)

        Returns:
            out: Coordonnées prédites (n_nodes, 2)
        """
        # 1. Encoder séparément ARN et protéines puis fusionner
        x = self.joint_encoder(x)

        # 2. Message passing avec GAT
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 3. Prédiction finale
        out = self.mlp(x)

        return out

    def count_parameters(self):
        """Compte le nombre de paramètres entraînables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SpatialGATLargeWithJointEncoder(nn.Module):
    """
    Version plus profonde avec Joint Encoder + 3 couches GAT.
    """

    def __init__(self, n_genes, n_proteins,
                 rna_hidden=256, protein_hidden=128, joint_hidden=256,
                 gat_hidden=256, out_channels=2, heads=4, dropout=0.4):
        super(SpatialGATLargeWithJointEncoder, self).__init__()

        self.dropout = dropout

        # Joint Encoder plus large
        self.joint_encoder = JointEncoder(
            n_genes=n_genes,
            n_proteins=n_proteins,
            rna_hidden=rna_hidden,
            protein_hidden=protein_hidden,
            joint_hidden=joint_hidden,
            dropout=dropout
        )

        # Trois couches GAT
        self.conv1 = GATConv(joint_hidden, gat_hidden, heads=heads,
                            dropout=dropout, concat=True)
        self.conv2 = GATConv(gat_hidden * heads, 96, heads=2,
                            dropout=dropout, concat=True)
        self.conv3 = GATConv(96 * 2, 128, heads=1,
                            dropout=dropout, concat=False)

        # MLP final
        self.mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_channels)
        )

    def forward(self, x, edge_index):
        # Joint encoding
        x = self.joint_encoder(x)

        # GAT layers
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


def create_joint_encoder_model(n_genes, n_proteins, model_type='base', **kwargs):
    """
    Factory function pour créer un modèle avec Joint Encoder.

    Args:
        n_genes: Nombre de gènes
        n_proteins: Nombre de protéines
        model_type: 'base' ou 'large'
        **kwargs: Arguments supplémentaires (rna_hidden, protein_hidden, etc.)

    Returns:
        model: Instance du modèle
    """
    if model_type == 'base':
        model = SpatialGATWithJointEncoder(
            n_genes=n_genes,
            n_proteins=n_proteins,
            **kwargs
        )
    elif model_type == 'large':
        model = SpatialGATLargeWithJointEncoder(
            n_genes=n_genes,
            n_proteins=n_proteins,
            **kwargs
        )
    else:
        raise ValueError(f"Type de modèle inconnu: {model_type}")

    print(f"\n✓ Modèle avec Joint Encoder créé: {model.__class__.__name__}")
    print(f"  • {n_genes} gènes → encodeur ARN")
    print(f"  • {n_proteins} protéines → encodeur protéines")
    print(f"  • Paramètres entraînables: {model.count_parameters():,}")

    return model

