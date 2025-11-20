"""
Pipeline d'entraînement du modèle GNN.
"""

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import time


class Trainer:
    """Classe pour gérer l'entraînement du modèle."""

    def __init__(self, model, data, train_mask, val_mask, test_mask,
                 lr=0.001, weight_decay=5e-4, device='cpu'):
        """
        Args:
            model: Modèle PyTorch
            data: Objet torch_geometric.data.Data
            train_mask, val_mask, test_mask: Masques booléens pour le split
            lr: Learning rate
            weight_decay: Régularisation L2
            device: 'cpu' ou 'cuda'
        """
        self.model = model.to(device)
        self.data = data.to(device)
        self.train_mask = train_mask.to(device)
        self.val_mask = val_mask.to(device)
        self.test_mask = test_mask.to(device)
        self.device = device

        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min',
                                          factor=0.5, patience=10, verbose=True)

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': []
        }

    def compute_loss_and_metrics(self, mask):
        """Calcule la loss MSE et le MAE pour un ensemble de données."""
        pred = self.model(self.data.x, self.data.edge_index)

        # Loss MSE (sur coordonnées normalisées)
        loss = F.mse_loss(pred[mask], self.data.y[mask])

        # MAE (sur coordonnées normalisées)
        mae = F.l1_loss(pred[mask], self.data.y[mask])

        return loss, mae, pred

    def train_epoch(self):
        """Entraîne le modèle pour une époque."""
        self.model.train()
        self.optimizer.zero_grad()

        loss, mae, _ = self.compute_loss_and_metrics(self.train_mask)

        loss.backward()
        self.optimizer.step()

        return loss.item(), mae.item()

    @torch.no_grad()
    def evaluate(self, mask):
        """Évalue le modèle sur un ensemble de données."""
        self.model.eval()
        loss, mae, pred = self.compute_loss_and_metrics(mask)
        return loss.item(), mae.item(), pred

    def train(self, epochs=200, early_stopping_patience=20, verbose=True):
        """
        Boucle d'entraînement complète.

        Args:
            epochs: Nombre d'époques maximum
            early_stopping_patience: Nombre d'époques sans amélioration avant arrêt
            verbose: Si True, affiche les métriques

        Returns:
            best_model_state: État du meilleur modèle
        """
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        print(f"\n{'='*60}")
        print(f"Début de l'entraînement sur {self.device}")
        print(f"{'='*60}\n")

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            # Entraînement
            train_loss, train_mae = self.train_epoch()

            # Validation
            val_loss, val_mae, _ = self.evaluate(self.val_mask)

            # Enregistrer l'historique
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_mae'].append(train_mae)
            self.history['val_mae'].append(val_mae)

            # Scheduler
            self.scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1

            # Affichage
            if verbose and (epoch == 1 or epoch % 10 == 0 or patience_counter == 0):
                print(f"Epoch {epoch:03d} | "
                      f"Train Loss: {train_loss:.4f} | Train MAE: {train_mae:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f} | "
                      f"Best: {best_val_loss:.4f}")

            # Arrêt anticipé
            if patience_counter >= early_stopping_patience:
                print(f"\n✓ Early stopping à l'époque {epoch}")
                break

        elapsed_time = time.time() - start_time
        print(f"\n✓ Entraînement terminé en {elapsed_time/60:.2f} minutes")
        print(f"✓ Meilleure validation loss: {best_val_loss:.4f}")

        # Charger le meilleur modèle
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return best_model_state

    @torch.no_grad()
    def predict(self, denormalize=False, coords_scaler=None):
        """
        Prédit les coordonnées pour toutes les cellules.

        Args:
            denormalize: Si True, dénormalise les prédictions
            coords_scaler: Scaler sklearn pour dénormaliser

        Returns:
            predictions: Coordonnées prédites (numpy array)
        """
        self.model.eval()
        pred = self.model(self.data.x, self.data.edge_index)

        predictions = pred.cpu().numpy()

        if denormalize and coords_scaler is not None:
            predictions = coords_scaler.inverse_transform(predictions)

        return predictions

    def get_history(self):
        """Retourne l'historique d'entraînement."""
        return self.history

    def save_model(self, path):
        """Sauvegarde le modèle."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, path)
        print(f"✓ Modèle sauvegardé: {path}")

    def load_model(self, path):
        """Charge un modèle sauvegardé."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        print(f"✓ Modèle chargé: {path}")


def compute_r2_score(y_true, y_pred):
    """
    Calcule le R² score.

    Args:
        y_true: Valeurs réelles (n_samples, 2)
        y_pred: Valeurs prédites (n_samples, 2)

    Returns:
        r2: R² score (calculé séparément pour x et y)
    """
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    r2 = 1 - (ss_res / ss_tot)
    return r2


def compute_euclidean_distance(y_true, y_pred):
    """
    Calcule la distance euclidienne entre prédictions et vraies coordonnées.

    Args:
        y_true: Coordonnées réelles (n_samples, 2)
        y_pred: Coordonnées prédites (n_samples, 2)

    Returns:
        distances: Distance euclidienne pour chaque cellule
    """
    return np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1))

