"""
Modèle GAT adapté pour l'approche spectrale.

Au lieu de prédire directement (x, y), le modèle prédit des coefficients
sur une base spectrale (Laplacian Eigenmaps).

Avantages:
- Prédictions naturellement "lisses" sur le graphe
- Respecte mieux la topologie
- Dimension flexible (8-16 dimensions spectrales)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from model_joint_encoder import JointEncoder


class SpectralGATWithJointEncoder(nn.Module):
    """
    GAT qui prédit des coefficients spectraux au lieu de coordonnées directes.

    Architecture:
    1. Joint Encoder: Traite séparément ARN et protéines
    2. GAT layers: Message passing sur le graphe
    3. MLP final: Prédiction des coefficients spectraux (n_spectral_dims au lieu de 2)

    La reconstruction des coordonnées (x, y) se fait ensuite via:
    coords = eigenvectors @ spectral_coeffs
    """

    def __init__(self, n_genes, n_proteins, n_spectral_dims=8,
                 rna_hidden=128, protein_hidden=64, joint_hidden=128,
                 gat_hidden=128, heads=4, dropout=0.4,
                 use_cross_attention=True):
        """
        Args:
            n_genes: Nombre de gènes
            n_proteins: Nombre de protéines
            n_spectral_dims: Nombre de dimensions spectrales (sortie du modèle)
            rna_hidden: Dimension cachée encodeur ARN
            protein_hidden: Dimension cachée encodeur protéines
            joint_hidden: Dimension représentation jointe
            gat_hidden: Dimension cachée des couches GAT
            heads: Nombre de têtes d'attention
            dropout: Taux de dropout
            use_cross_attention: Utiliser l'attention croisée entre modalités
        """
        super(SpectralGATWithJointEncoder, self).__init__()

        self.n_spectral_dims = n_spectral_dims
        self.dropout = dropout

        # Joint Encoder pour séparer les modalités
        self.joint_encoder = JointEncoder(
            n_genes=n_genes,
            n_proteins=n_proteins,
            rna_hidden=rna_hidden,
            protein_hidden=protein_hidden,
            joint_hidden=joint_hidden,
            dropout=dropout,
            use_cross_attention=use_cross_attention
        )

        # Couches GAT
        self.conv1 = GATConv(
            joint_hidden,
            gat_hidden,
            heads=heads,
            dropout=dropout,
            concat=True
        )

        self.residual1 = nn.Linear(joint_hidden, gat_hidden * heads)

        self.conv2 = GATConv(
            gat_hidden * heads,
            64,
            heads=1,
            dropout=dropout,
            concat=False
        )

        self.residual2 = nn.Linear(gat_hidden * heads, 64)

        # MLP final pour prédire les coefficients spectraux
        # CHANGEMENT: sortie = 2*n_spectral_dims (coefficients pour x et y séparément)
        self.mlp = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2 * n_spectral_dims)  # Coefficients pour x et y
        )

    def forward(self, data):
        """
        Args:
            data: Batch PyG avec attributs:
                - x: Features (n_nodes, n_genes + n_proteins)
                - edge_index: Arêtes
                - batch: Indices de batch

        Returns:
            spectral_coeffs: Coefficients spectraux prédits (n_nodes, n_spectral_dims)
        """
        x, edge_index = data.x, data.edge_index

        # 1. Joint Encoder
        x = self.joint_encoder(x)

        # 2. Première couche GAT avec résiduelle
        identity = self.residual1(x)
        x = F.elu(self.conv1(x, edge_index))
        x = x + identity
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 3. Deuxième couche GAT avec résiduelle
        identity = self.residual2(x)
        x = F.elu(self.conv2(x, edge_index))
        x = x + identity
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 4. MLP final pour coefficients spectraux
        spectral_coeffs = self.mlp(x)

        return spectral_coeffs


class SpectralGAT(nn.Module):
    """
    Version standard du GAT spectral (sans Joint Encoder).

    Pour compatibilité si on n'utilise pas les modalités séparées.
    """

    def __init__(self, in_channels, n_spectral_dims=8, hidden_channels=128,
                 heads=4, dropout=0.4):
        """
        Args:
            in_channels: Nombre de features d'entrée
            n_spectral_dims: Nombre de dimensions spectrales (sortie)
            hidden_channels: Dimension cachée des couches GAT
            heads: Nombre de têtes d'attention
            dropout: Taux de dropout
        """
        super(SpectralGAT, self).__init__()

        self.n_spectral_dims = n_spectral_dims
        self.dropout = dropout

        # Couches GAT
        self.conv1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            dropout=dropout,
            concat=True
        )

        self.conv2 = GATConv(
            hidden_channels * heads,
            64,
            heads=1,
            dropout=dropout,
            concat=False
        )

        # MLP final
        self.mlp = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2 * n_spectral_dims)  # Coefficients pour x et y
        )

    def forward(self, data):
        """
        Args:
            data: Batch PyG avec x et edge_index

        Returns:
            spectral_coeffs: Coefficients spectraux prédits
        """
        x, edge_index = data.x, data.edge_index

        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)

        spectral_coeffs = self.mlp(x)

        return spectral_coeffs


def create_spectral_model(n_genes=None, n_proteins=None, in_channels=None,
                         n_spectral_dims=8, model_type='large',
                         use_joint_encoder=True, **kwargs):
    """
    Factory function pour créer un modèle spectral.

    Args:
        n_genes: Nombre de gènes (requis si use_joint_encoder=True)
        n_proteins: Nombre de protéines (requis si use_joint_encoder=True)
        in_channels: Nombre de features (requis si use_joint_encoder=False)
        n_spectral_dims: Nombre de dimensions spectrales à prédire
        model_type: 'small', 'medium', 'large'
        use_joint_encoder: Utiliser l'encodeur séparé pour ARN/protéines
        **kwargs: Autres paramètres (rna_hidden, protein_hidden, etc.)

    Returns:
        model: Instance du modèle spectral
    """
    # Configurations prédéfinies
    configs = {
        'small': {
            'rna_hidden': 128,
            'protein_hidden': 64,
            'joint_hidden': 128,
            'gat_hidden': 128,
            'heads': 2,
            'dropout': 0.3
        },
        'medium': {
            'rna_hidden': 192,
            'protein_hidden': 96,
            'joint_hidden': 256,
            'gat_hidden': 256,
            'heads': 4,
            'dropout': 0.4
        },
        'large': {
            'rna_hidden': 256,
            'protein_hidden': 128,
            'joint_hidden': 400,
            'gat_hidden': 400,
            'heads': 4,
            'dropout': 0.4
        }
    }

    config = configs.get(model_type, configs['medium'])
    config.update(kwargs)  # Overrides avec kwargs

    if use_joint_encoder:
        if n_genes is None or n_proteins is None:
            raise ValueError("n_genes et n_proteins requis si use_joint_encoder=True")

        model = SpectralGATWithJointEncoder(
            n_genes=n_genes,
            n_proteins=n_proteins,
            n_spectral_dims=n_spectral_dims,
            rna_hidden=config['rna_hidden'],
            protein_hidden=config['protein_hidden'],
            joint_hidden=config['joint_hidden'],
            gat_hidden=config['gat_hidden'],
            heads=config['heads'],
            dropout=config['dropout'],
            use_cross_attention=config.get('use_cross_attention', True)
        )

        print(f"\n✓ Modèle Spectral avec Joint Encoder créé:")
        print(f"  • Type: {model_type}")
        print(f"  • Gènes: {n_genes}, Protéines: {n_proteins}")
        print(f"  • Dimensions spectrales: {n_spectral_dims}")
        print(f"  • GAT hidden: {config['gat_hidden']}, Heads: {config['heads']}")
    else:
        if in_channels is None:
            raise ValueError("in_channels requis si use_joint_encoder=False")

        model = SpectralGAT(
            in_channels=in_channels,
            n_spectral_dims=n_spectral_dims,
            hidden_channels=config.get('hidden_channels', 128),
            heads=config['heads'],
            dropout=config['dropout']
        )

        print(f"\n✓ Modèle Spectral standard créé:")
        print(f"  • Type: {model_type}")
        print(f"  • Input channels: {in_channels}")
        print(f"  • Dimensions spectrales: {n_spectral_dims}")

    # Compter les paramètres
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  • Paramètres: {n_params:,}")

    return model

