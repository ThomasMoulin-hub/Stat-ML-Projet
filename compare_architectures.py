"""
Script de comparaison entre le modèle standard et le modèle avec Joint Encoder.

Ce script entraîne les deux architectures et compare leurs performances.
"""

import torch
import time
from model import create_model
from model_joint_encoder import create_joint_encoder_model
from train_subgraph import SubgraphTrainer
from evaluate import evaluate_predictions
import matplotlib.pyplot as plt
import json
import os


def compare_architectures(subgraphs_list, train_indices, val_indices, test_indices,
                         coords_scaler, n_genes, n_proteins, device,
                         epochs=50, results_dir='results'):
    """
    Compare les performances du modèle standard vs Joint Encoder.

    Args:
        subgraphs_list: Liste des sous-graphes
        train_indices, val_indices, test_indices: Indices des splits
        coords_scaler: Scaler pour dénormaliser les coordonnées
        n_genes: Nombre de gènes
        n_proteins: Nombre de protéines
        device: Device (cuda/cpu)
        epochs: Nombre d'epochs d'entraînement
        results_dir: Dossier pour sauvegarder les résultats
    """

    os.makedirs(results_dir, exist_ok=True)
    in_channels = n_genes + n_proteins

    print("="*80)
    print("🔬 COMPARAISON DES ARCHITECTURES")
    print("="*80)

    results = {}

    # ============================================================
    # 1. MODÈLE STANDARD
    # ============================================================
    print("\n" + "="*80)
    print("📊 MODÈLE 1: STANDARD (toutes features concaténées)")
    print("="*80)

    model_standard = create_model(
        in_channels=in_channels,
        model_type='large',
        hidden_channels=256,
        heads=4,
        dropout=0.4
    ).to(device)

    trainer_standard = SubgraphTrainer(
        model=model_standard,
        subgraphs_list=subgraphs_list,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        batch_size=300,
        lr=0.001,
        weight_decay=5e-4,
        device=device
    )

    print("\n🎓 Entraînement du modèle standard...")
    start_time = time.time()
    trainer_standard.train(epochs=epochs, early_stopping_patience=20, verbose=True)
    train_time_standard = time.time() - start_time

    # Évaluation
    y_pred_test, y_true_test = trainer_standard.predict_all(
        trainer_standard.test_loader,
        denormalize=True,
        coords_scaler=coords_scaler
    )
    metrics_standard, distances_standard = evaluate_predictions(
        y_true_test, y_pred_test, set_name='Test Standard'
    )

    history_standard = trainer_standard.get_history()

    results['standard'] = {
        'metrics': metrics_standard,
        'train_time': train_time_standard,
        'history': history_standard,
        'params': model_standard.count_parameters()
    }

    print(f"\n✓ Modèle Standard terminé en {train_time_standard:.2f}s")
    print(f"  • MAE: {metrics_standard['mae']:.6f}")
    print(f"  • RMSE: {metrics_standard['rmse']:.6f}")
    print(f"  • Distance euclidienne moyenne: {metrics_standard['mean_euclidean_distance']:.6f}")

    # ============================================================
    # 2. MODÈLE JOINT ENCODER
    # ============================================================
    print("\n" + "="*80)
    print("🧬 MODÈLE 2: JOINT ENCODER (ARN et Protéines séparés)")
    print("="*80)

    model_joint = create_joint_encoder_model(
        n_genes=n_genes,
        n_proteins=n_proteins,
        model_type='large',
        rna_hidden=256,
        protein_hidden=128,
        joint_hidden=256,
        gat_hidden=256,
        heads=4,
        dropout=0.4
    ).to(device)

    trainer_joint = SubgraphTrainer(
        model=model_joint,
        subgraphs_list=subgraphs_list,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        batch_size=300,
        lr=0.001,
        weight_decay=5e-4,
        device=device
    )

    print("\n🎓 Entraînement du modèle Joint Encoder...")
    start_time = time.time()
    trainer_joint.train(epochs=epochs, early_stopping_patience=20, verbose=True)
    train_time_joint = time.time() - start_time

    # Évaluation
    y_pred_test, y_true_test = trainer_joint.predict_all(
        trainer_joint.test_loader,
        denormalize=True,
        coords_scaler=coords_scaler
    )
    metrics_joint, distances_joint = evaluate_predictions(
        y_true_test, y_pred_test, set_name='Test Joint Encoder'
    )

    history_joint = trainer_joint.get_history()

    results['joint_encoder'] = {
        'metrics': metrics_joint,
        'train_time': train_time_joint,
        'history': history_joint,
        'params': model_joint.count_parameters()
    }

    print(f"\n✓ Modèle Joint Encoder terminé en {train_time_joint:.2f}s")
    print(f"  • MAE: {metrics_joint['mae']:.6f}")
    print(f"  • RMSE: {metrics_joint['rmse']:.6f}")
    print(f"  • Distance euclidienne moyenne: {metrics_joint['mean_euclidean_distance']:.6f}")

    # ============================================================
    # 3. COMPARAISON
    # ============================================================
    print("\n" + "="*80)
    print("📊 COMPARAISON FINALE")
    print("="*80)

    print(f"\n{'Métrique':<30} {'Standard':<15} {'Joint Encoder':<15} {'Amélioration':<15}")
    print("-"*80)

    for metric_name in ['mae', 'rmse', 'mean_euclidean_distance', 'median_euclidean_distance']:
        val_std = metrics_standard[metric_name]
        val_joint = metrics_joint[metric_name]
        improvement = ((val_std - val_joint) / val_std) * 100

        print(f"{metric_name:<30} {val_std:<15.6f} {val_joint:<15.6f} {improvement:>+14.2f}%")

    print(f"\n{'Paramètres':<30} {results['standard']['params']:<15,} {results['joint_encoder']['params']:<15,}")
    print(f"{'Temps d\'entraînement (s)':<30} {train_time_standard:<15.2f} {train_time_joint:<15.2f}")

    # ============================================================
    # 4. VISUALISATIONS
    # ============================================================
    print(f"\n📈 Génération des visualisations...")

    # Courbes d'apprentissage comparées
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Train loss
    axes[0].plot(history_standard['train_loss'], label='Standard', linewidth=2)
    axes[0].plot(history_joint['train_loss'], label='Joint Encoder', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Train Loss')
    axes[0].set_title('Comparaison Train Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Val loss
    axes[1].plot(history_standard['val_loss'], label='Standard', linewidth=2)
    axes[1].plot(history_joint['val_loss'], label='Joint Encoder', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation Loss')
    axes[1].set_title('Comparaison Validation Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    comparison_path = os.path.join(results_dir, 'architecture_comparison.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Graphiques sauvegardés: {comparison_path}")
    plt.close()

    # Sauvegarder les résultats en JSON
    results_json = {
        'standard': {
            'metrics': {k: float(v) for k, v in metrics_standard.items()},
            'train_time': train_time_standard,
            'params': results['standard']['params'],
            'final_train_loss': float(history_standard['train_loss'][-1]),
            'final_val_loss': float(history_standard['val_loss'][-1]),
            'best_val_loss': float(min(history_standard['val_loss']))
        },
        'joint_encoder': {
            'metrics': {k: float(v) for k, v in metrics_joint.items()},
            'train_time': train_time_joint,
            'params': results['joint_encoder']['params'],
            'final_train_loss': float(history_joint['train_loss'][-1]),
            'final_val_loss': float(history_joint['val_loss'][-1]),
            'best_val_loss': float(min(history_joint['val_loss']))
        }
    }

    json_path = os.path.join(results_dir, 'architecture_comparison.json')
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"  ✓ Résultats JSON sauvegardés: {json_path}")

    # ============================================================
    # 5. RECOMMANDATION
    # ============================================================
    print("\n" + "="*80)
    print("💡 RECOMMANDATION")
    print("="*80)

    if metrics_joint['mae'] < metrics_standard['mae']:
        improvement_pct = ((metrics_standard['mae'] - metrics_joint['mae']) / metrics_standard['mae']) * 100
        print(f"\n🏆 Le modèle Joint Encoder est MEILLEUR!")
        print(f"   Amélioration MAE: {improvement_pct:.2f}%")
        print(f"   Recommandation: Utiliser l'architecture Joint Encoder pour votre projet.")
    else:
        print(f"\n📊 Le modèle Standard performe mieux pour ce dataset.")
        print(f"   Le Joint Encoder peut nécessiter plus d'epochs ou un tuning différent.")

    return results


if __name__ == "__main__":
    print("⚠️  Ce script doit être importé et appelé depuis main.py")
    print("    Ajoutez ces lignes dans main.py après la création des sous-graphes:")
    print()
    print("    from compare_architectures import compare_architectures")
    print("    results = compare_architectures(")
    print("        subgraphs_list, train_indices, val_indices, test_indices,")
    print("        coords_scaler, n_genes, n_proteins, device")
    print("    )")

