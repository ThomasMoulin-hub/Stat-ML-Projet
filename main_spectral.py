#!/user/tmm2219/.conda/envs/statml/bin/python

"""
Main script pour l'approche SPECTRALE.

Au lieu de prédire directement (x, y), le modèle prédit des coefficients
sur une base spectrale (Laplacian Eigenmaps du graphe K-NN).

Différences avec main.py:
1. Utilise spectral_preprocessing.py pour calculer les eigenvectors
2. Utilise model_spectral.py pour prédire les coefficients spectraux
3. Utilise train_spectral.py pour l'entraînement
4. Reconstruction des coordonnées via la base spectrale pour l'évaluation
"""

import spatialdata_io as sio
import torch
from scipy.sparse import issparse
from data_preprocessing import preprocess_adata
from spectral_preprocessing import build_spectral_subgraphs
from data_preprocessing_subgraph import create_subgraph_splits
from model_spectral import create_spectral_model
from train_spectral import SpectralSubgraphTrainer
from evaluate import (evaluate_predictions, plot_training_history,
                      plot_predictions_vs_true, plot_spatial_predictions,
                      plot_error_distribution, analyze_extreme_errors)
import os
import resource
import pickle
import json
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np
import pandas as pd


RLIMIT_NOFILE = resource.RLIMIT_NOFILE
soft_limit_initial, hard_limit_initial = resource.getrlimit(RLIMIT_NOFILE)
resource.setrlimit(RLIMIT_NOFILE, (hard_limit_initial, hard_limit_initial))

torch.multiprocessing.set_sharing_strategy('file_system')


def run_ddp_spectral(rank, world_size, subgraphs_path, train_indices, val_indices, test_indices,
                     spectral_basis_path, n_genes, n_proteins, n_spectral_dims, use_joint_encoder):
    """
    Fonction DDP pour entraînement distribué avec approche spectrale.
    """
    # Init process group
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    device = f'cuda:{rank}'
    torch.cuda.set_device(rank)

    # Charger les données
    subgraphs_list = torch.load(subgraphs_path, weights_only=False)
    with open(spectral_basis_path, 'rb') as f:
        spectral_basis = pickle.load(f)

    # Créer le modèle spectral
    if use_joint_encoder:
        model = create_spectral_model(
            n_genes=n_genes,
            n_proteins=n_proteins,
            n_spectral_dims=n_spectral_dims,
            model_type='large',
            use_joint_encoder=True,
            rna_hidden=256,
            protein_hidden=128,
            joint_hidden=400,
            gat_hidden=400,
            heads=4,
            dropout=0.4,
            use_cross_attention=True
        )
    else:
        in_channels = subgraphs_list[0].x.shape[1]
        model = create_spectral_model(
            in_channels=in_channels,
            n_spectral_dims=n_spectral_dims,
            model_type='large',
            use_joint_encoder=False,
            hidden_channels=256,
            heads=4,
            dropout=0.4
        )

    # Créer le trainer spectral
    trainer = SpectralSubgraphTrainer(
        model=model,
        subgraphs_list=subgraphs_list,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        spectral_basis=spectral_basis,
        batch_size=800,
        lr=0.001,
        weight_decay=5e-4,
        device=device,
        lambda_smooth=0.0,
        distributed=True,
        rank=rank,
        world_size=world_size
    )

    # Entraînement
    best_state = trainer.train(epochs=200, early_stopping_patience=15, verbose=True)

    # Sauvegarder uniquement sur rank 0
    if rank == 0:
        os.makedirs('results_spectral', exist_ok=True)
        trainer.save_model('results_spectral/spatial_gat_spectral_model.pt')

        # Historique d'entraînement
        history = trainer.get_history()
        plot_training_history(history, save_path='results_spectral/training_history.png')

        # Prédictions avec RECONSTRUCTION SPATIALE pour l'évaluation
        print(f"\n{'='*60}")
        print("ÉVALUATION SUR L'ENSEMBLE DE TEST")
        print(f"{'='*60}")

        # Prédire avec reconstruction spatiale
        y_pred_test, y_true_test = trainer.predict_all(
            trainer.test_loader,
            reconstruct_spatial=True  # ✅ Reconstruction pour obtenir (x, y)
        )

        print(f"• Coordonnées prédites (reconstruites): {y_pred_test.shape}")
        print(f"• Coordonnées spatiales vraies: {y_true_test.shape}")

        # Calculer les métriques
        metrics, euclidean_distances = evaluate_predictions(
            y_true_test,
            y_pred_test,
            set_name='Test (Spectral)'
        )

        # Visualiser prédictions vs réalité
        plot_predictions_vs_true(
            y_true_test,
            y_pred_test,
            save_path='results_spectral/predictions_vs_true_spectral.png'
        )

        # Visualiser les positions spatiales
        plot_spatial_predictions(
            y_true_test,
            y_pred_test,
            euclidean_distances,
            save_path='results_spectral/spatial_predictions_spectral.png'
        )

        # Distribution des erreurs
        plot_error_distribution(
            euclidean_distances,
            save_path='results_spectral/error_distribution_spectral.png'
        )

        # Analyser les erreurs extrêmes
        worst_cells, best_cells = analyze_extreme_errors(
            y_true_test,
            y_pred_test,
            euclidean_distances,
            top_n=10
        )

        # Sauvegarder les métriques dans un fichier CSV
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv('results_spectral/test_metrics_spectral.csv', index=False)
        print("✓ Métriques sauvegardées dans results_spectral/test_metrics_spectral.csv")

        # Sauvegarder les prédictions
        np.savez('results_spectral/predictions.npz',
                y_pred=y_pred_test,
                y_true=y_true_test,
                euclidean_distances=euclidean_distances)
        print(f"✓ Prédictions sauvegardées dans results_spectral/predictions.npz")

        # Sauvegarder l'historique
        with open('results_spectral/history.json', 'w') as f:
            json.dump(history, f, indent=2)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    print(f"{'='*60}")
    print("PIPELINE GNN AVEC APPROCHE SPECTRALE")
    print(f"{'='*60}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print("\n🎯 Approche: LAPLACIAN EIGENMAPS (SPECTRALE)")
    print("   • Au lieu de prédire (x, y), on prédit des coefficients spectraux")
    print("   • Base spectrale = vecteurs propres du Laplacien du graphe K-NN")
    print("   • Avantages: régularité, robustesse, cohérence topologique")
    print("   • Reconstruction: coords = eigenvectors @ spectral_coeffs")
    print(f"{'='*60}\n")

    # Chargement des données
    dataset_name = "Xenium_V1_Human_Kidney_FFPE_Protein_updated_outs/"
    xenium_path = "./data/" + dataset_name

    print("Chargement des données Xenium...")
    sdata = sio.xenium(xenium_path, gex_only=False, morphology_focus=False,
                       cells_boundaries=False, nucleus_boundaries=False,
                       cells_labels=False, nucleus_labels=False, cells_as_circles=True)

    adata = sdata.tables["table"]
    adata_processed = preprocess_adata(adata, normalize_genes=True, normalize_proteins=True)

    # Paramètres
    k_value = 49
    metric_value = 'euclidean'
    n_spectral_dims = 12  # Nombre de dimensions spectrales (8-16 recommandé)
    sigma = 1.0  # Paramètre du noyau gaussien pour le Laplacien
    subset_fraction = 1  # 🧪 MODE TEST: 10% des cellules (mettre 1.0 pour tout)

    cache_dir = 'cache_spectral_' + dataset_name
    os.makedirs(cache_dir, exist_ok=True)

    cache_key = f"subgraphs_spectral_k{k_value}_metric_{metric_value}_dims{n_spectral_dims}_sigma{sigma}_subset{int(subset_fraction*100)}"
    subgraphs_path = os.path.join(cache_dir, cache_key + '.pt')
    spectral_basis_path = os.path.join(cache_dir, cache_key + '_basis.pkl')
    splits_path = os.path.join(cache_dir, cache_key + '_splits.json')

    use_cache = (os.path.exists(subgraphs_path) and
                 os.path.exists(spectral_basis_path))

    if use_cache:
        print(f"🔁 Cache détecté: chargement depuis {cache_dir}/")
        subgraphs_list = torch.load(subgraphs_path, weights_only=False)
        with open(spectral_basis_path, 'rb') as f:
            spectral_basis = pickle.load(f)
        print(f"  ✓ {len(subgraphs_list)} sous-graphes chargés")
        print(f"  ✓ Base spectrale: {spectral_basis['n_components']} dimensions")
    else:
        print("🚀 Construction des sous-graphes spectraux...")

        # Extraire features et coordonnées
        if issparse(adata_processed.X):
            features = adata_processed.X.toarray()
        else:
            features = adata_processed.X

        spatial_coords = adata_processed.obsm["spatial"]

        print(f"\nDonnées:")
        print(f"  • Features: {features.shape}")
        print(f"  • Coordonnées: {spatial_coords.shape}")

        # Construire les sous-graphes spectraux
        # MODE TEST: Utiliser seulement 10% des cellules pour tester
        subgraphs_list, spectral_basis = build_spectral_subgraphs(
            features=features,
            spatial_coords=spatial_coords,
            k=k_value,
            metric=metric_value,
            n_spectral_dims=n_spectral_dims,
            sigma=sigma,
            subset_fraction=subset_fraction
        )

        # Sauvegarder
        torch.save(subgraphs_list, subgraphs_path)
        with open(spectral_basis_path, 'wb') as f:
            pickle.dump(spectral_basis, f)
        print(f"\n💾 Sous-graphes et base spectrale sauvegardés dans {cache_dir}/")

    # Créer/charger les splits
    if os.path.exists(splits_path):
        print("🔁 Chargement des splits depuis le cache")
        with open(splits_path, 'r') as f:
            splits_data = json.load(f)
        train_indices = splits_data['train_indices']
        val_indices = splits_data['val_indices']
        test_indices = splits_data['test_indices']
    else:
        print("⚙️ Création des splits")
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

    # Info sur les données
    data = subgraphs_list[0]
    print(f"\n📊 Structure des sous-graphes spectraux:")
    print(f"  • Nœuds par sous-graphe: {data.x.shape[0]}")
    print(f"  • Features par nœud: {data.x.shape[1]}")
    print(f"  • Arêtes: {data.edge_index.shape[1]}")
    print(f"  • Cibles spectrales (y_spectral): {data.y_spectral.shape}")
    print(f"  • Coordonnées spatiales originales (y_spatial): {data.y_spatial.shape}")

    # Récupérer n_genes et n_proteins
    n_genes = (adata_processed.var["feature_types"] == "Gene Expression").sum()
    n_proteins = (adata_processed.var["feature_types"] == "Protein Expression").sum()

    print(f"\n📊 Modalités biologiques:")
    print(f"  • Gènes: {n_genes}")
    print(f"  • Protéines: {n_proteins}")
    print(f"  • Total features: {n_genes + n_proteins}")

    # Libérer la mémoire
    del subgraphs_list

    # Entraînement distribué
    use_joint_encoder = True
    world_size = 4

    print(f"\n🚀 Lancement de l'entraînement distribué (DDP)")
    print(f"  • GPUs: {world_size}")
    print(f"  • Joint Encoder: {use_joint_encoder}")
    print(f"  • Dimensions spectrales: {n_spectral_dims}")

    mp.spawn(
        run_ddp_spectral,
        args=(world_size, subgraphs_path, train_indices, val_indices, test_indices,
              spectral_basis_path, n_genes, n_proteins, n_spectral_dims, use_joint_encoder),
        nprocs=world_size,
        join=True
    )

    print(f"\n{'='*60}")
    print("✓ ENTRAÎNEMENT TERMINÉ")
    print(f"{'='*60}")
    print("Les résultats sont dans le dossier: results_spectral/")
    print("\nProchaine étape: implémenter la reconstruction complète")
    print("des coordonnées spatiales via la base spectrale pour évaluation.")

