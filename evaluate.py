"""
Évaluation et visualisation des résultats du modèle.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd


def evaluate_predictions(y_true, y_pred, set_name='Test'):
    """
    Calcule les métriques d'évaluation.

    Args:
        y_true: Coordonnées réelles (n_samples, 2)
        y_pred: Coordonnées prédites (n_samples, 2)
        set_name: Nom de l'ensemble de données

    Returns:
        metrics: Dictionnaire avec les métriques
    """
    # Métriques globales
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    # R² pour chaque dimension
    r2_x = r2_score(y_true[:, 0], y_pred[:, 0])
    r2_y = r2_score(y_true[:, 1], y_pred[:, 1])
    r2_mean = (r2_x + r2_y) / 2

    # Distance euclidienne
    euclidean_distances = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1))
    mean_euclidean = np.mean(euclidean_distances)
    median_euclidean = np.median(euclidean_distances)

    metrics = {
        'MSE': mse,
        'MAE': mae,
        'R2_x': r2_x,
        'R2_y': r2_y,
        'R2_mean': r2_mean,
        'Mean_Euclidean_Distance': mean_euclidean,
        'Median_Euclidean_Distance': median_euclidean,
        'Max_Euclidean_Distance': np.max(euclidean_distances),
        'Min_Euclidean_Distance': np.min(euclidean_distances)
    }

    print(f"\n{'='*60}")
    print(f"Métriques d'évaluation - {set_name}")
    print(f"{'='*60}")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² (x): {r2_x:.4f}")
    print(f"R² (y): {r2_y:.4f}")
    print(f"R² (moyen): {r2_mean:.4f}")
    print(f"Distance euclidienne moyenne: {mean_euclidean:.2f}")
    print(f"Distance euclidienne médiane: {median_euclidean:.2f}")
    print(f"{'='*60}\n")

    return metrics, euclidean_distances


def plot_training_history(history, save_path=None):
    """
    Visualise l'historique d'entraînement.

    Args:
        history: Dictionnaire avec les métriques d'entraînement
        save_path: Chemin pour sauvegarder la figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('MSE Loss', fontsize=12)
    axes[0].set_title('Loss durant l\'entraînement', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # MAE
    axes[1].plot(history['train_mae'], label='Train MAE', linewidth=2)
    axes[1].plot(history['val_mae'], label='Val MAE', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('MAE', fontsize=12)
    axes[1].set_title('MAE durant l\'entraînement', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure sauvegardée: {save_path}")

    plt.show()


def plot_predictions_vs_true(y_true, y_pred, save_path=None):
    """
    Scatter plot comparant prédictions et valeurs réelles.

    Args:
        y_true: Coordonnées réelles (n_samples, 2)
        y_pred: Coordonnées prédites (n_samples, 2)
        save_path: Chemin pour sauvegarder la figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Coordonnée X
    axes[0].scatter(y_true[:, 0], y_pred[:, 0], alpha=0.3, s=5, c='blue')
    axes[0].plot([y_true[:, 0].min(), y_true[:, 0].max()],
                 [y_true[:, 0].min(), y_true[:, 0].max()],
                 'r--', linewidth=2, label='Ligne idéale')
    axes[0].set_xlabel('X réel', fontsize=12)
    axes[0].set_ylabel('X prédit', fontsize=12)
    axes[0].set_title('Prédictions vs Réalité (coordonnée X)', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Coordonnée Y
    axes[1].scatter(y_true[:, 1], y_pred[:, 1], alpha=0.3, s=5, c='green')
    axes[1].plot([y_true[:, 1].min(), y_true[:, 1].max()],
                 [y_true[:, 1].min(), y_true[:, 1].max()],
                 'r--', linewidth=2, label='Ligne idéale')
    axes[1].set_xlabel('Y réel', fontsize=12)
    axes[1].set_ylabel('Y prédit', fontsize=12)
    axes[1].set_title('Prédictions vs Réalité (coordonnée Y)', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure sauvegardée: {save_path}")

    plt.show()


def plot_spatial_predictions(y_true, y_pred, euclidean_distances, save_path=None):
    """
    Visualise les positions réelles et prédites dans l'espace spatial.

    Args:
        y_true: Coordonnées réelles (n_samples, 2)
        y_pred: Coordonnées prédites (n_samples, 2)
        euclidean_distances: Distances euclidiennes (n_samples,)
        save_path: Chemin pour sauvegarder la figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Positions réelles
    axes[0].scatter(y_true[:, 0], y_true[:, 1], alpha=0.5, s=1, c='blue')
    axes[0].set_xlabel('X', fontsize=12)
    axes[0].set_ylabel('Y', fontsize=12)
    axes[0].set_title('Positions réelles', fontsize=14, fontweight='bold')
    axes[0].set_aspect('equal')

    # Positions prédites
    axes[1].scatter(y_pred[:, 0], y_pred[:, 1], alpha=0.5, s=1, c='red')
    axes[1].set_xlabel('X', fontsize=12)
    axes[1].set_ylabel('Y', fontsize=12)
    axes[1].set_title('Positions prédites', fontsize=14, fontweight='bold')
    axes[1].set_aspect('equal')

    # Erreurs spatiales (heatmap)
    scatter = axes[2].scatter(y_true[:, 0], y_true[:, 1],
                             c=euclidean_distances, cmap='YlOrRd',
                             alpha=0.6, s=2)
    axes[2].set_xlabel('X', fontsize=12)
    axes[2].set_ylabel('Y', fontsize=12)
    axes[2].set_title('Erreur de prédiction (distance euclidienne)',
                     fontsize=14, fontweight='bold')
    axes[2].set_aspect('equal')
    cbar = plt.colorbar(scatter, ax=axes[2])
    cbar.set_label('Distance euclidienne', fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure sauvegardée: {save_path}")

    plt.show()


def plot_error_distribution(euclidean_distances, save_path=None):
    """
    Visualise la distribution des erreurs.

    Args:
        euclidean_distances: Distances euclidiennes (n_samples,)
        save_path: Chemin pour sauvegarder la figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogramme
    axes[0].hist(euclidean_distances, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].axvline(np.mean(euclidean_distances), color='red', linestyle='--',
                   linewidth=2, label=f'Moyenne: {np.mean(euclidean_distances):.2f}')
    axes[0].axvline(np.median(euclidean_distances), color='orange', linestyle='--',
                   linewidth=2, label=f'Médiane: {np.median(euclidean_distances):.2f}')
    axes[0].set_xlabel('Distance euclidienne', fontsize=12)
    axes[0].set_ylabel('Fréquence', fontsize=12)
    axes[0].set_title('Distribution des erreurs de prédiction', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Boxplot
    axes[1].boxplot(euclidean_distances, vert=True)
    axes[1].set_ylabel('Distance euclidienne', fontsize=12)
    axes[1].set_title('Boxplot des erreurs', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure sauvegardée: {save_path}")

    plt.show()


def analyze_extreme_errors(y_true, y_pred, euclidean_distances,
                          features=None, top_n=10):
    """
    Analyse les cellules avec les plus grandes et plus petites erreurs.

    Args:
        y_true: Coordonnées réelles
        y_pred: Coordonnées prédites
        euclidean_distances: Distances euclidiennes
        features: Features des cellules (optionnel)
        top_n: Nombre de cellules à analyser

    Returns:
        worst_cells, best_cells: Indices des cellules
    """
    # Cellules avec les plus grandes erreurs
    worst_indices = np.argsort(euclidean_distances)[-top_n:][::-1]

    # Cellules avec les plus petites erreurs
    best_indices = np.argsort(euclidean_distances)[:top_n]

    print(f"\n{'='*60}")
    print(f"Top {top_n} cellules avec les PLUS GRANDES erreurs:")
    print(f"{'='*60}")
    for idx in worst_indices:
        print(f"Cellule {idx}: Distance={euclidean_distances[idx]:.2f}, "
              f"Réel=({y_true[idx, 0]:.1f}, {y_true[idx, 1]:.1f}), "
              f"Prédit=({y_pred[idx, 0]:.1f}, {y_pred[idx, 1]:.1f})")

    print(f"\n{'='*60}")
    print(f"Top {top_n} cellules avec les PLUS PETITES erreurs:")
    print(f"{'='*60}")
    for idx in best_indices:
        print(f"Cellule {idx}: Distance={euclidean_distances[idx]:.2f}, "
              f"Réel=({y_true[idx, 0]:.1f}, {y_true[idx, 1]:.1f}), "
              f"Prédit=({y_pred[idx, 0]:.1f}, {y_pred[idx, 1]:.1f})")

    return worst_indices, best_indices

