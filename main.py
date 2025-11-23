#!/user/tmm2219/.conda/envs/statml/bin/python
#import squidpy as sq
import spatialdata as sd
import spatialdata_io as sio
from spatialdata import read_zarr
import pandas as pd
#import spatialdata_plot
import torch
from scipy.sparse import issparse

from model_joint_encoder import create_joint_encoder_model
from data_preprocessing import preprocess_adata
from data_preprocessing_subgraph import build_local_subgraphs, create_subgraph_splits
from train_subgraph import SubgraphTrainer as Trainer


from model import create_model
from evaluate import (evaluate_predictions, plot_training_history,
                      plot_predictions_vs_true, plot_spatial_predictions,
                      plot_error_distribution, analyze_extreme_errors)
import os

import pickle
import json


dataset_name="Xenium_V1_Human_Kidney_FFPE_Protein_updated_outs/"
xenium_path = "./data/" + dataset_name

sdata = sio.xenium(xenium_path, gex_only=False, morphology_focus=False, cells_boundaries=False, nucleus_boundaries=False, cells_labels=False, nucleus_labels=False, cells_as_circles=True)

# Récupère l'AnnData
adata = sdata.tables["table"]



# # Pipeline GNN pour prédiction de coordonnées spatiales
#
# Ce notebook implémente un Graph Neural Network (GAT) qui prédit les coordonnées
# spatiales des cellules basé uniquement sur leurs profils d'expression d'ARN et de protéines.
# Le graphe K-NN est construit sur la similarité d'expression (pas les coordonnées).
#
# ## ⚙️ Deux approches disponibles :
#
# ### Approche 1 : Graphe Global (défaut)
# - **Un seul grand graphe** avec toutes les cellules
# - Chaque cellule connectée à ses k voisins les plus proches
# - Le GNN traite tout le graphe en une fois
# - **Plus rapide** mais moins flexible
# - Chaque cellule voit indirectement toutes les autres via les couches GNN
#
# ### Approche 2 : Sous-graphes Locaux (recommandé pour votre question)
# - **Un sous-graphe par cellule** : 1 cellule centrale + 29 voisins = 30 nœuds
# - Le GNN prédit UNIQUEMENT la position de la cellule centrale
# - Traitement par batches de sous-graphes
# - **Plus conforme à votre description** : chaque point d'entraînement = 1 sous-graphe
# - Isolement complet : chaque prédiction utilise uniquement son voisinage local




print(f"{'='*60}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device utilisé: {device}")
print("🎯 Approche: SOUS-GRAPHES LOCAUX")
print("   • Chaque point = 1 cellule centrale + 29 voisins")
print("   • Prédiction uniquement de la cellule centrale")
print("   • Traitement par batches")
print(f"{'='*60}\n")


# Imports pour le pipeline GNN



# ## 1. Préparation et normalisation des données


# Prétraiter les données (filtrer et normaliser)
adata_processed = preprocess_adata(adata, normalize_genes=True, normalize_proteins=True)


# Extraire les features et coordonnées spatiales
if issparse(adata_processed.X):
    features = adata_processed.X.toarray()
else:
    features = adata_processed.X

spatial_coords = adata_processed.obsm["spatial"]

print(f"Shape des features: {features.shape}")
print(f"Shape des coordonnées: {spatial_coords.shape}")


# ## 2. Construction du graphe K-NN basé sur similarité d'expression



# Approche sous-graphes locaux
print("Construction des sous-graphes locaux...")
# Paramètres de construction
k_value = 49
metric_value = 'cosine'
cache_dir = 'cache_' + dataset_name
os.makedirs(cache_dir, exist_ok=True)
cache_key = f"subgraphs_k{k_value}_metric_{metric_value}"
subgraphs_path = os.path.join(cache_dir, cache_key + '.pt')
scaler_path = os.path.join(cache_dir, cache_key + '_scaler.pkl')
splits_path = os.path.join(cache_dir, cache_key + '_splits.json')

use_cache = os.path.exists(subgraphs_path) and os.path.exists(scaler_path)
if use_cache:
    print(f"🔁 Cache détecté: chargement depuis {cache_dir}/")
    subgraphs_list = torch.load(subgraphs_path, weights_only=False)
    with open(scaler_path, 'rb') as f:
        coords_scaler = pickle.load(f)
else:
    print("🚀 Pas de cache ou incomplet: construction des sous-graphes")
    subgraphs_list, coords_scaler = build_local_subgraphs(
        features=features,
        spatial_coords=spatial_coords,
        k=k_value,
        metric=metric_value
    )
    # Sauvegarder
    torch.save(subgraphs_list, subgraphs_path)
    with open(scaler_path, 'wb') as f:
        pickle.dump(coords_scaler, f)
    print(f"💾 Sous-graphes et scaler sauvegardés dans {cache_dir}/")

# Créer / charger les splits d'indices
if os.path.exists(splits_path):
    print("🔁 Chargement des splits depuis le cache")
    with open(splits_path, 'r') as f:
        splits_data = json.load(f)
    train_indices = splits_data['train_indices']
    val_indices = splits_data['val_indices']
    test_indices = splits_data['test_indices']
else:
    print("⚙️ Création des splits d'entraînement/validation/test")
    train_indices, val_indices, test_indices = create_subgraph_splits(
        n_subgraphs=len(subgraphs_list),
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )
    with open(splits_path, 'w') as f:
        json.dump({
            'train_indices': train_indices,
            'val_indices': val_indices,
            'test_indices': test_indices
        }, f)
    print(f"💾 Splits sauvegardés dans {splits_path}")

# Pour compatibilité avec le reste du code
data = subgraphs_list[0]  # Juste pour afficher les infos
print(f"\nExemple de sous-graphe:")
print(f"  • Nœuds: {data.x.shape[0]} (1 centrale + {data.x.shape[0]-1} voisins)")
print(f"  • Features par nœud: {data.x.shape[1]}")
print(f"  • Arêtes: {data.edge_index.shape[1]}")
print(f"  • Cible: position de la cellule centrale uniquement")




# ## 3. Création du modèle GAT


# Créer le modèle avec Joint Encoder
in_channels = subgraphs_list[0].x.shape[1]

# Récupérer le nombre de gènes et protéines
n_genes = (adata_processed.var["feature_types"] == "Gene Expression").sum()
n_proteins = (adata_processed.var["feature_types"] == "Protein Expression").sum()

print(f"\n📊 Modalités biologiques:")
print(f"  • Gènes: {n_genes}")
print(f"  • Protéines: {n_proteins}")
print(f"  • Total features: {in_channels}")

# Choisir l'architecture: 'joint_encoder' ou 'standard'
use_joint_encoder = True  # Mettre False pour utiliser l'ancien modèle

if use_joint_encoder:
    print(f"\n🧬 Utilisation du Joint Encoder (ARN et Protéines séparés)")
    model = create_joint_encoder_model(
        n_genes=n_genes,
        n_proteins=n_proteins,
        model_type='large',     # 'base' ou 'large'
        rna_hidden=256,         # Encodeur ARN
        protein_hidden=128,     # Encodeur protéines
        joint_hidden=256,       # Représentation fusionnée
        gat_hidden=256,         # GAT layers
        heads=4,
        dropout=0.4
    )
else:
    print(f"\n📊 Utilisation du modèle standard (toutes features concaténées)")
    model = create_model(
        in_channels=in_channels,
        model_type='large',
        hidden_channels=256,
        heads=4,
        dropout=0.4
    )


# ## 4. Entraînement du modèle


# Créer le trainer

trainer = Trainer(
        model=model,
        subgraphs_list=subgraphs_list,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        batch_size=300,
        lr=0.001,
        weight_decay=5e-4,
        device=device
    )



# Entraîner le modèle
best_model_state = trainer.train(
    epochs=200,
    early_stopping_patience=20,
    verbose=True
)


# Visualiser l'historique d'entraînement
history = trainer.get_history()
plot_training_history(history, save_path='results/training_history.png')


# ## 5. Évaluation sur l'ensemble de test


# Prédire sur l'ensemble de test

# Pour les sous-graphes, utiliser la méthode spécifique
y_pred_test, y_true_test = trainer.predict_all(
    trainer.test_loader,
    denormalize=True,
    coords_scaler=coords_scaler
)



# Calculer les métriques
metrics, euclidean_distances = evaluate_predictions(
    y_true_test,
    y_pred_test,
    set_name='Test'
)


# Visualiser prédictions vs réalité
plot_predictions_vs_true(y_true_test, y_pred_test,
                        save_path='results/predictions_vs_true.png')


# Visualiser les positions spatiales
plot_spatial_predictions(y_true_test, y_pred_test, euclidean_distances,
                        save_path='results/spatial_predictions.png')


# Distribution des erreurs
plot_error_distribution(euclidean_distances,
                       save_path='results/error_distribution.png')


# Analyser les erreurs extrêmes
worst_cells, best_cells = analyze_extreme_errors(
    y_true_test,
    y_pred_test,
    euclidean_distances,
    top_n=10
)


# ## 6. Sauvegarder le modèle
# Créer le dossier results s'il n'existe pas
import os
os.makedirs('results', exist_ok=True)

# Sauvegarder le modèle
trainer.save_model('results/spatial_gat_model.pt')


# Sauvegarder les métriques dans un fichier CSV
import pandas as pd
metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv('results/test_metrics.csv', index=False)
print("✓ Métriques sauvegardées dans results/test_metrics.csv")