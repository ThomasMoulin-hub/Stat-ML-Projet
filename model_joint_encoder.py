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
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool


class CrossModalAttention(nn.Module):
    """
    Couche d'attention croisée entre ARN et protéines.

    Permet aux deux modalités de s'informer mutuellement avant la fusion :
    - L'ARN peut interroger les protéines pour enrichir sa représentation
    - Les protéines peuvent interroger l'ARN pour enrichir leur représentation

    Utilise un mécanisme d'attention multi-têtes similaire aux Transformers.
    """

    def __init__(self, rna_dim, protein_dim, num_heads=4, dropout=0.3):
        """
        Args:
            rna_dim: Dimension de la représentation ARN (doit être divisible par num_heads)
            protein_dim: Dimension de la représentation protéines (doit être divisible par num_heads)
            num_heads: Nombre de têtes d'attention
            dropout: Taux de dropout
        """
        super(CrossModalAttention, self).__init__()

        # Vérifier que les dimensions sont divisibles par num_heads
        assert rna_dim % num_heads == 0, f"rna_dim ({rna_dim}) doit être divisible par num_heads ({num_heads})"
        assert protein_dim % num_heads == 0, f"protein_dim ({protein_dim}) doit être divisible par num_heads ({num_heads})"

        self.num_heads = num_heads
        self.rna_dim = rna_dim
        self.protein_dim = protein_dim
        self.head_dim_rna = rna_dim // num_heads
        self.head_dim_protein = protein_dim // num_heads

        # Projections pour ARN interroge Protéines (ARN = Query, Protéines = Key/Value)
        self.rna_to_q = nn.Linear(rna_dim, rna_dim)
        self.protein_to_k = nn.Linear(protein_dim, rna_dim)
        self.protein_to_v = nn.Linear(protein_dim, rna_dim)

        # Projections pour Protéines interroge ARN (Protéines = Query, ARN = Key/Value)
        self.protein_to_q = nn.Linear(protein_dim, protein_dim)
        self.rna_to_k = nn.Linear(rna_dim, protein_dim)
        self.rna_to_v = nn.Linear(rna_dim, protein_dim)

        self.dropout = nn.Dropout(dropout)
        # Scale pour stabiliser l'attention (basé sur la dimension par tête)
        self.scale_rna = self.head_dim_rna ** 0.5
        self.scale_protein = self.head_dim_protein ** 0.5

    def forward(self, rna_encoded, protein_encoded):
        """
        Args:
            rna_encoded: (n_nodes, rna_dim) - Représentation ARN
            protein_encoded: (n_nodes, protein_dim) - Représentation protéines

        Returns:
            rna_attended: ARN enrichi par les protéines (n_nodes, rna_dim)
            protein_attended: Protéines enrichies par l'ARN (n_nodes, protein_dim)
        """
        n_nodes = rna_encoded.size(0)

        # === ARN interroge Protéines (RNA->Protein Cross-Attention) ===
        q_rna = self.rna_to_q(rna_encoded)  # (n_nodes, rna_dim)
        k_protein = self.protein_to_k(protein_encoded)  # (n_nodes, rna_dim)
        v_protein = self.protein_to_v(protein_encoded)  # (n_nodes, rna_dim)

        # Reshape pour multi-head attention: (n_nodes, num_heads, head_dim)
        head_dim = self.rna_dim // self.num_heads
        q_rna = q_rna.view(n_nodes, self.num_heads, head_dim)
        k_protein = k_protein.view(n_nodes, self.num_heads, head_dim)
        v_protein = v_protein.view(n_nodes, self.num_heads, head_dim)

        # Calcul de l'attention: Q @ K^T / sqrt(d_k)
        # (n_nodes, num_heads, head_dim) @ (n_nodes, num_heads, head_dim)^T
        attn_scores_rna = torch.einsum('nhd,nhd->nh', q_rna, k_protein) / self.scale_rna  # (n_nodes, num_heads)
        attn_weights_rna = F.softmax(attn_scores_rna, dim=-1)  # (n_nodes, num_heads)
        attn_weights_rna = self.dropout(attn_weights_rna)

        # Appliquer l'attention: attn_weights @ V
        # (n_nodes, num_heads) * (n_nodes, num_heads, head_dim) -> (n_nodes, num_heads, head_dim)
        rna_context = torch.einsum('nh,nhd->nhd', attn_weights_rna, v_protein)
        rna_context = rna_context.reshape(n_nodes, self.rna_dim)  # (n_nodes, rna_dim)

        # Connexion résiduelle + LayerNorm (optionnel mais recommandé)
        rna_attended = rna_encoded + rna_context

        # === Protéines interrogent ARN (Protein->RNA Cross-Attention) ===
        q_protein = self.protein_to_q(protein_encoded)  # (n_nodes, protein_dim)
        k_rna = self.rna_to_k(rna_encoded)  # (n_nodes, protein_dim)
        v_rna = self.rna_to_v(rna_encoded)  # (n_nodes, protein_dim)

        # Reshape pour multi-head attention
        head_dim_protein = self.protein_dim // self.num_heads
        q_protein = q_protein.view(n_nodes, self.num_heads, head_dim_protein)
        k_rna = k_rna.view(n_nodes, self.num_heads, head_dim_protein)
        v_rna = v_rna.view(n_nodes, self.num_heads, head_dim_protein)

        # Calcul de l'attention
        attn_scores_protein = torch.einsum('nhd,nhd->nh', q_protein, k_rna) / self.scale_protein
        attn_weights_protein = F.softmax(attn_scores_protein, dim=-1)
        attn_weights_protein = self.dropout(attn_weights_protein)

        # Appliquer l'attention
        protein_context = torch.einsum('nh,nhd->nhd', attn_weights_protein, v_rna)
        protein_context = protein_context.reshape(n_nodes, self.protein_dim)

        # Connexion résiduelle
        protein_attended = protein_encoded + protein_context

        return rna_attended, protein_attended


class JointEncoder(nn.Module):
    """
    Encodeur séparé pour ARN et protéines avant fusion.

    Cette architecture traite séparément les modalités biologiques différentes
    (transcriptomique vs protéomique) avant de les combiner.

    NOUVEAU: Intègre une couche CrossModalAttention pour permettre aux modalités
    de s'informer mutuellement avant la fusion.
    """

    def __init__(self, n_genes, n_proteins, rna_hidden=128, protein_hidden=64,
                 joint_hidden=128, dropout=0.3, use_cross_attention=True):
        """
        Args:
            n_genes: Nombre de gènes (405 par exemple)
            n_proteins: Nombre de protéines (27 par exemple)
            rna_hidden: Dimension cachée pour l'encodeur ARN
            protein_hidden: Dimension cachée pour l'encodeur protéines
            joint_hidden: Dimension de la représentation jointe
            dropout: Taux de dropout
            use_cross_attention: Utiliser l'attention croisée entre modalités
        """
        super(JointEncoder, self).__init__()

        self.n_genes = n_genes
        self.n_proteins = n_proteins
        self.use_cross_attention = use_cross_attention

        # Encodeur ARN (plus de capacité car plus de features)
        self.rna_encoder = nn.Sequential(
            nn.Linear(n_genes, rna_hidden),
            nn.LayerNorm(rna_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(rna_hidden, rna_hidden),
            nn.LayerNorm(rna_hidden),
            nn.ReLU()
        )

        # Encodeur protéines (plus petit car moins de features)
        self.protein_encoder = nn.Sequential(
            nn.Linear(n_proteins, protein_hidden),
            nn.LayerNorm(protein_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(protein_hidden, protein_hidden),
            nn.LayerNorm(protein_hidden),
            nn.ReLU()
        )

        # Attention croisée entre modalités
        if use_cross_attention:
            self.cross_attention = CrossModalAttention(
                rna_dim=rna_hidden,
                protein_dim=protein_hidden,
                num_heads=4,
                dropout=dropout
            )

        # Fusion des deux modalités
        self.fusion = nn.Sequential(
            nn.Linear(rna_hidden + protein_hidden, joint_hidden),
            nn.LayerNorm(joint_hidden),
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

        # Appliquer l'attention croisée si activée
        if self.use_cross_attention:
            rna_encoded, protein_encoded = self.cross_attention(rna_encoded, protein_encoded)

        # Concaténer et fusionner
        combined = torch.cat([rna_encoded, protein_encoded], dim=1)
        joint_repr = self.fusion(combined)

        return joint_repr


class SpatialGATWithJointEncoder(nn.Module):
    """
    GAT avec encodeur séparé pour ARN et protéines.

    Architecture améliorée:
    1. Joint Encoder: Traite séparément ARN et protéines puis fusionne (avec attention croisée)
    2. GAT layers: Message passing sur le graphe (avec connexions résiduelles)
    3. Global Pooling: Combine mean et max pooling pour capturer patterns globaux
    4. MLP final: Prédiction des coordonnées
    """

    def __init__(self, n_genes, n_proteins,
                 rna_hidden=128, protein_hidden=64, joint_hidden=128,
                 gat_hidden=128, out_channels=2, heads=4, dropout=0.4,
                 use_cross_attention=True, use_global_pooling=True):
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
            use_cross_attention: Utiliser l'attention croisée entre modalités
            use_global_pooling: Utiliser le pooling global (mean + max)
        """
        super(SpatialGATWithJointEncoder, self).__init__()

        self.dropout = dropout
        self.use_global_pooling = use_global_pooling

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

        # Couches GAT sur la représentation jointe
        self.conv1 = GATConv(
            joint_hidden,
            gat_hidden,
            heads=heads,
            dropout=dropout,
            concat=True
        )

        # Projection pour la connexion résiduelle (ajuster les dimensions)
        self.residual1 = nn.Linear(joint_hidden, gat_hidden * heads)

        self.conv2 = GATConv(
            gat_hidden * heads,
            64,
            heads=1,
            dropout=dropout,
            concat=False
        )

        # Projection pour la connexion résiduelle
        self.residual2 = nn.Linear(gat_hidden * heads, 64)

        # Déterminer la taille d'entrée du MLP
        if use_global_pooling:
            # Si pooling: on concatène mean_pool + max_pool (2x la dimension)
            mlp_input_dim = 64 * 2
        else:
            mlp_input_dim = 64

        # MLP final
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, out_channels)
        )

    def forward(self, x, edge_index, batch=None):
        """
        Forward pass avec encodage séparé des modalités.

        Args:
            x: Features [ARN | Protéines] (n_nodes, n_genes + n_proteins)
            edge_index: Liste d'arêtes (2, n_edges)
            batch: Indices de batch pour le pooling (n_nodes,)

        Returns:
            out: Coordonnées prédites (n_nodes ou batch_size, 2)
        """
        # 1. Encoder séparément ARN et protéines puis fusionner
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

        # 3. Global pooling (optionnel)
        if self.use_global_pooling and batch is not None:
            # Combiner mean et max pooling
            x_mean = global_mean_pool(x, batch)  # (batch_size, 64)
            x_max = global_max_pool(x, batch)    # (batch_size, 64)
            x = torch.cat([x_mean, x_max], dim=1)  # (batch_size, 128)

        # 4. Prédiction finale
        out = self.mlp(x)

        return out

    def count_parameters(self):
        """Compte le nombre de paramètres entraînables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SpatialGATLargeWithJointEncoder(nn.Module):
    """
    Version plus profonde avec Joint Encoder + 3 couches GAT.

    NOUVEAU: Intègre connexions résiduelles et global pooling.
    """

    def __init__(self, n_genes, n_proteins,
                 rna_hidden=256, protein_hidden=128, joint_hidden=256,
                 gat_hidden=256, out_channels=2, heads=4, dropout=0.4,
                 use_cross_attention=True, use_global_pooling=True):
        super(SpatialGATLargeWithJointEncoder, self).__init__()

        self.dropout = dropout
        self.use_global_pooling = use_global_pooling

        # Joint Encoder plus large avec attention croisée
        self.joint_encoder = JointEncoder(
            n_genes=n_genes,
            n_proteins=n_proteins,
            rna_hidden=rna_hidden,
            protein_hidden=protein_hidden,
            joint_hidden=joint_hidden,
            dropout=dropout,
            use_cross_attention=use_cross_attention
        )

        # Trois couches GAT avec connexions résiduelles
        self.conv1 = GATConv(joint_hidden, gat_hidden, heads=heads,
                            dropout=dropout, concat=True)
        self.residual1 = nn.Linear(joint_hidden, gat_hidden * heads)

        self.conv2 = GATConv(gat_hidden * heads, 96, heads=2,
                            dropout=dropout, concat=True)
        self.residual2 = nn.Linear(gat_hidden * heads, 96 * 2)

        self.conv3 = GATConv(96 * 2, 128, heads=1,
                            dropout=dropout, concat=False)
        self.residual3 = nn.Linear(96 * 2, 128)

        # Déterminer la taille d'entrée du MLP
        if use_global_pooling:
            mlp_input_dim = 128 * 2  # mean + max pooling
        else:
            mlp_input_dim = 128

        # MLP final
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_channels)
        )

    def forward(self, data):
        """
        Forward pass avec connexions résiduelles et pooling global.

        Args:
            x: Features [ARN | Protéines] (n_nodes, n_genes + n_proteins)
            edge_index: Liste d'arêtes (2, n_edges)
            batch: Indices de batch pour le pooling (n_nodes,)

        Returns:
            out: Coordonnées prédites (n_nodes ou batch_size, 2)
        """

        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Joint encoding avec attention croisée
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

        # Global pooling (combine mean + max)
        if self.use_global_pooling and batch is not None:
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x = torch.cat([x_mean, x_max], dim=1)

        # Prédiction finale
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
        **kwargs: Arguments supplémentaires (rna_hidden, protein_hidden,
                  use_cross_attention, use_global_pooling, etc.)

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
    print(f"  • Cross-modal attention: {kwargs.get('use_cross_attention', True)}")
    print(f"  • Global pooling (mean+max): {kwargs.get('use_global_pooling', True)}")
    print(f"  • Connexions résiduelles: Oui (dans GAT layers)")
    print(f"  • Paramètres entraînables: {model.count_parameters():,}")

    return model

