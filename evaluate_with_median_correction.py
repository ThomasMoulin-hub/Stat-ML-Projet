#!/user/tmm2219/.conda/envs/statml/bin/python
"""
Script pour évaluer le modèle entraîné sur la partie droite du dataset
avec correction de la médiane sur la coordonnée x.
"""

import torch
import pickle
import json
import os
import numpy as np
from torch_geometric.loader import DataLoader

from model_joint_encoder import create_joint_encoder_model
from model import create_model
from train_subgraph import ListDataset
from evaluate import (evaluate_predictions, plot_training_history,
                      plot_predictions_vs_true, plot_spatial_predictions,
                      plot_error_distribution, analyze_extreme_errors)


def load_model_and_evaluate(
    model_path,
    subgraphs_path,
    test_indices,
    scaler_path,
    metadata_path,
    output_dir='results_evaluation_median_corrected',
    use_joint_encoder=True,
    device='cuda'
):
    """
    Charge le modèle et évalue sur le test set avec correction de médiane.

    Args:
        model_path: Chemin vers le modèle sauvegardé (.pt)
        subgraphs_path: Chemin vers les sous-graphes
        test_indices: Indices du test set
        scaler_path: Chemin vers le scaler des coordonnées
        metadata_path: Chemin vers les métadonnées (n_genes, n_proteins)
        output_dir: Dossier de sortie pour les résultats
        use_joint_encoder: Utiliser le joint encoder ou non
        device: 'cuda' ou 'cpu'
    """

    print(f"{'='*60}")
    print(f"Évaluation avec correction de médiane sur coordonnée X")
    print(f"{'='*60}\n")

    # Vérifier que le device est disponible
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA non disponible, passage en CPU")
        device = 'cpu'

    print(f"Device utilisé: {device}")

    # Créer le dossier de sortie
    os.makedirs(output_dir, exist_ok=True)

    # Charger les métadonnées
    print(f"\n📂 Chargement des métadonnées depuis {metadata_path}")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        n_genes = metadata['n_genes']
        n_proteins = metadata['n_proteins']

    print(f"  • Gènes: {n_genes}")
    print(f"  • Protéines: {n_proteins}")

    # Charger les sous-graphes
    print(f"\n📂 Chargement des sous-graphes depuis {subgraphs_path}")
    subgraphs_list = torch.load(subgraphs_path, weights_only=False)
    print(f"  • Total sous-graphes: {len(subgraphs_list)}")

    # Charger le scaler
    print(f"\n📂 Chargement du scaler depuis {scaler_path}")
    with open(scaler_path, 'rb') as f:
        coords_scaler = pickle.load(f)

    # Créer le test loader
    test_dataset = ListDataset([subgraphs_list[i] for i in test_indices])
    test_loader = DataLoader(test_dataset, batch_size=600, shuffle=False)
    print(f"\n✓ Test set: {len(test_indices)} sous-graphes")

    # Obtenir in_channels depuis le premier sous-graphe
    in_channels = subgraphs_list[0].x.shape[1]
    print(f"  • Features d'entrée: {in_channels}")

    # Créer le modèle
    print(f"\n🔧 Création du modèle...")
    if use_joint_encoder:
        print("  • Architecture: Joint Encoder")
        model = create_joint_encoder_model(
            n_genes=n_genes,
            n_proteins=n_proteins,
            model_type='base',
            rna_hidden=256,
            protein_hidden=128,
            joint_hidden=400,
            gat_hidden=400,
            heads=6,
            dropout=0.3,
            use_cross_attention=False,
            use_global_pooling=False
        )
    else:
        print("  • Architecture: Standard GAT")
        model = create_model(
            in_channels=in_channels,
            model_type='large',
            hidden_channels=256,
            heads=4,
            dropout=0.4
        )

    # Charger les poids du modèle
    print(f"\n📥 Chargement du modèle depuis {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le fichier de modèle n'existe pas: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    # Gérer différents formats de checkpoint
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            # Format avec model_state_dict (nouveau format)
            print("  • Format de checkpoint: avec model_state_dict")
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            # Format alternatif
            print("  • Format de checkpoint: avec state_dict")
            state_dict = checkpoint['state_dict']
        else:
            # Le checkpoint est directement le state_dict
            print("  • Format de checkpoint: state_dict direct")
            state_dict = checkpoint
    else:
        # Format très ancien (peu probable)
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("✓ Modèle chargé avec succès")

    # Faire les prédictions directement (sans passer par SubgraphTrainer)
    print(f"\n🔮 Prédiction sur le test set...")
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch)

            # Extraire les prédictions et cibles pour les nœuds centraux
            batch_size = batch.num_graphs
            central_predictions = []
            central_targets = []

            for i in range(batch_size):
                start_idx = batch.ptr[i]
                central_predictions.append(pred[start_idx])
                central_targets.append(batch.y[start_idx])

            pred_central = torch.stack(central_predictions)
            target_central = torch.stack(central_targets)
            all_predictions.append(pred_central.cpu())
            all_targets.append(target_central.cpu())

    # Concaténer et convertir en numpy
    y_pred_test = torch.cat(all_predictions, dim=0).numpy()
    y_true_test = torch.cat(all_targets, dim=0).numpy()

    # Dénormaliser
    y_pred_test = coords_scaler.inverse_transform(y_pred_test)
    y_true_test = coords_scaler.inverse_transform(y_true_test)

    print(f"  • Prédictions: {y_pred_test.shape}")
    print(f"  • Vraies valeurs: {y_true_test.shape}")

    # CORRECTION: Ajouter la médiane de la coordonnée X
    print(f"\n🔧 Application de la correction de médiane sur X...")
    median_x_true = np.median(y_true_test[:, 0])
    median_x_pred = np.median(y_pred_test[:, 0])
    correction_x = median_x_true - median_x_pred

    print(f"  • Médiane X (vraie): {median_x_true:.2f}")
    print(f"  • Médiane X (prédite): {median_x_pred:.2f}")
    print(f"  • Correction appliquée: {correction_x:.2f}")

    y_pred_corrected = y_pred_test.copy()
    y_pred_corrected[:, 0] += correction_x

    # Sauvegarder les prédictions brutes et corrigées
    np.save(os.path.join(output_dir, 'predictions_raw.npy'), y_pred_test)
    np.save(os.path.join(output_dir, 'predictions_corrected.npy'), y_pred_corrected)
    np.save(os.path.join(output_dir, 'true_values.npy'), y_true_test)

    correction_info = {
        'median_x_true': float(median_x_true),
        'median_x_pred': float(median_x_pred),
        'correction_applied': float(correction_x)
    }
    with open(os.path.join(output_dir, 'correction_info.json'), 'w') as f:
        json.dump(correction_info, f, indent=2)

    print(f"✓ Prédictions sauvegardées dans {output_dir}/")

    # Évaluer AVANT correction
    print(f"\n{'='*60}")
    print("📊 MÉTRIQUES AVANT CORRECTION")
    print(f"{'='*60}")
    metrics_raw, euclidean_distances_raw = evaluate_predictions(
        y_true_test,
        y_pred_test,
        set_name='Test (brut)'
    )

    # Évaluer APRÈS correction
    print(f"\n{'='*60}")
    print("📊 MÉTRIQUES APRÈS CORRECTION DE MÉDIANE")
    print(f"{'='*60}")
    metrics_corrected, euclidean_distances_corrected = evaluate_predictions(
        y_true_test,
        y_pred_corrected,
        set_name='Test (corrigé)'
    )

    # Sauvegarder les métriques
    import pandas as pd

    metrics_df = pd.DataFrame([
        {**{'type': 'raw'}, **metrics_raw},
        {**{'type': 'corrected'}, **metrics_corrected}
    ])
    metrics_df.to_csv(os.path.join(output_dir, 'test_metrics_comparison.csv'), index=False)
    print(f"\n✓ Métriques sauvegardées dans {output_dir}/test_metrics_comparison.csv")

    # Visualisations pour les prédictions CORRIGÉES
    print(f"\n📊 Génération des visualisations...")

    plot_predictions_vs_true(
        y_true_test, y_pred_corrected,
        save_path=os.path.join(output_dir, 'predictions_vs_true_corrected.png')
    )

    plot_spatial_predictions(
        y_true_test, y_pred_corrected, euclidean_distances_corrected,
        save_path=os.path.join(output_dir, 'spatial_predictions_corrected.png')
    )

    plot_error_distribution(
        euclidean_distances_corrected,
        save_path=os.path.join(output_dir, 'error_distribution_corrected.png')
    )

    # Aussi générer les visualisations pour les prédictions BRUTES
    plot_predictions_vs_true(
        y_true_test, y_pred_test,
        save_path=os.path.join(output_dir, 'predictions_vs_true_raw.png')
    )

    plot_spatial_predictions(
        y_true_test, y_pred_test, euclidean_distances_raw,
        save_path=os.path.join(output_dir, 'spatial_predictions_raw.png')
    )

    plot_error_distribution(
        euclidean_distances_raw,
        save_path=os.path.join(output_dir, 'error_distribution_raw.png')
    )

    # Analyser les erreurs extrêmes (sur les prédictions corrigées)
    print(f"\n🔍 Analyse des erreurs extrêmes (prédictions corrigées)...")
    worst_cells, best_cells = analyze_extreme_errors(
        y_true_test,
        y_pred_corrected,
        euclidean_distances_corrected,
        top_n=10
    )

    print(f"\n{'='*60}")
    print(f"✅ ÉVALUATION TERMINÉE")
    print(f"{'='*60}")
    print(f"📁 Tous les résultats sont dans: {output_dir}/")
    print(f"  • Métriques comparatives: test_metrics_comparison.csv")
    print(f"  • Prédictions brutes: predictions_raw.npy")
    print(f"  • Prédictions corrigées: predictions_corrected.npy")
    print(f"  • Vraies valeurs: true_values.npy")
    print(f"  • Info correction: correction_info.json")
    print(f"  • Visualisations: *_raw.png et *_corrected.png")


if __name__ == '__main__':
    import sys

    # Configuration
    dataset_name = "Xenium_V1_Human_Kidney_FFPE_Protein_updated_outs/"
    cache_dir = f'cache_{dataset_name}'
    k_value = 99
    metric_value = 'euclidean'
    cache_key = f"subgraphs_k{k_value}_metric_{metric_value}"

    # Chemins des fichiers
    subgraphs_path = os.path.join(cache_dir, cache_key + '.pt')
    scaler_path = os.path.join(cache_dir, cache_key + '_scaler.pkl')
    metadata_path = os.path.join(cache_dir, cache_key + '_metadata.json')
    spatial_splits_path = os.path.join(cache_dir, cache_key + '_spatial_splits.json')

    # Déterminer le chemin du modèle
    # Priorité 1: argument en ligne de commande
    # Priorité 2: results_split_droite si existe
    # Priorité 3: results_rna_proteine_100knn
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        print(f"📍 Utilisation du modèle spécifié: {model_path}")
    elif os.path.exists('results_split_droite/spatial_gat_model.pt'):
        model_path = 'results_split_droite/spatial_gat_model.pt'
        print(f"📍 Utilisation du modèle depuis results_split_droite/")
    else:
        model_path = 'results_rna_proteine_100knn/spatial_gat_model.pt'
        print(f"⚠️  results_split_droite non trouvé, utilisation de: {model_path}")

    # Dossier de sortie
    output_dir = 'results_evaluation_median_corrected'

    # Vérifier que les fichiers existent
    required_files = [subgraphs_path, scaler_path, metadata_path, spatial_splits_path, model_path]
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        print("❌ Fichiers manquants:")
        for f in missing_files:
            print(f"  • {f}")
        print("\n⚠️  Veuillez vérifier les chemins et relancer le script.")
        exit(1)

    # Charger les indices de test
    print(f"📂 Chargement des splits depuis {spatial_splits_path}")
    with open(spatial_splits_path, 'r') as f:
        splits_data = json.load(f)
    test_indices = splits_data['test_indices']

    print(f"✓ Test set: {len(test_indices)} cellules")

    # Lancer l'évaluation
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    load_model_and_evaluate(
        model_path=model_path,
        subgraphs_path=subgraphs_path,
        test_indices=test_indices,
        scaler_path=scaler_path,
        metadata_path=metadata_path,
        output_dir=output_dir,
        use_joint_encoder=True,  # Mettre False si vous utilisez le modèle standard
        device=device
    )

