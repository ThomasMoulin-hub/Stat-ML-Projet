"""
Trainer pour le modèle avec sous-graphes locaux.
Chaque batch contient plusieurs sous-graphes indépendants.
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
import numpy as np
import time


class SubgraphTrainer:
    """Classe pour entraîner le modèle sur des sous-graphes locaux."""
    
    def __init__(self, model, subgraphs_list, train_indices, val_indices, test_indices,
                 batch_size=32, lr=0.001, weight_decay=5e-4, device='cpu'):
        """
        Args:
            model: Modèle PyTorch
            subgraphs_list: Liste de sous-graphes (Data objects)
            train_indices, val_indices, test_indices: Indices des sous-graphes
            batch_size: Taille des batches
            lr: Learning rate
            weight_decay: Régularisation L2
            device: 'cpu' ou 'cuda'
        """
        self.model = model.to(device)
        self.device = device
        
        # Créer les datasets
        train_subgraphs = [subgraphs_list[i] for i in train_indices]
        val_subgraphs = [subgraphs_list[i] for i in val_indices]
        test_subgraphs = [subgraphs_list[i] for i in test_indices]
        
        # Créer les DataLoaders
        self.train_loader = DataLoader(train_subgraphs, batch_size=batch_size, persistent_workers=True, num_workers=4, pin_memory=True, shuffle=True, prefetch_factor=10)
        self.val_loader = DataLoader(val_subgraphs, batch_size=batch_size, persistent_workers=True,num_workers=4, pin_memory=True, shuffle=False, prefetch_factor=10)
        self.test_loader = DataLoader(test_subgraphs, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=False, prefetch_factor=10)
        

        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min',
                                          factor=0.5, patience=10)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': []
        }
        
        print(f"✓ DataLoaders créés:")
        print(f"  Train: {len(train_subgraphs)} sous-graphes, {len(self.train_loader)} batches")
        print(f"  Val: {len(val_subgraphs)} sous-graphes, {len(self.val_loader)} batches")
        print(f"  Test: {len(test_subgraphs)} sous-graphes, {len(self.test_loader)} batches")
    
    def train_epoch(self):
        """Entraîne le modèle pour une époque."""
        self.model.train()
        total_loss = 0
        total_mae = 0
        n_batches = 0
        
        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            # Forward pass - prédire UNIQUEMENT la cellule centrale (nœud 0) de chaque sous-graphe
            pred = self.model(batch.x, batch.edge_index)
            
            # Extraire uniquement les prédictions des cellules centrales
            # Dans chaque sous-graphe, la cellule centrale est le nœud 0
            batch_size = batch.num_graphs
            central_predictions = []
            
            # batch.ptr indique où commence chaque graphe dans le batch
            for i in range(batch_size):
                start_idx = batch.ptr[i]
                # Nœud 0 de chaque sous-graphe = cellule centrale
                central_predictions.append(pred[start_idx])
            
            pred_central = torch.stack(central_predictions)
            
            # Loss MSE
            loss = F.mse_loss(pred_central, batch.y)
            
            # MAE
            mae = F.l1_loss(pred_central, batch.y)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_mae += mae.item()
            n_batches += 1
        
        return total_loss / n_batches, total_mae / n_batches
    
    @torch.no_grad()
    def evaluate(self, loader):
        """Évalue le modèle sur un DataLoader."""
        self.model.eval()
        total_loss = 0
        total_mae = 0
        n_batches = 0
        all_predictions = []
        all_targets = []
        
        for batch in loader:
            batch = batch.to(self.device)
            
            # Forward pass
            pred = self.model(batch.x, batch.edge_index)
            
            # Extraire les prédictions des cellules centrales
            batch_size = batch.num_graphs
            central_predictions = []
            
            for i in range(batch_size):
                start_idx = batch.ptr[i]
                central_predictions.append(pred[start_idx])
            
            pred_central = torch.stack(central_predictions)
            
            # Loss et MAE
            loss = F.mse_loss(pred_central, batch.y)
            mae = F.l1_loss(pred_central, batch.y)
            
            total_loss += loss.item()
            total_mae += mae.item()
            n_batches += 1
            
            # Sauvegarder pour analyse
            all_predictions.append(pred_central.cpu())
            all_targets.append(batch.y.cpu())
        
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        return total_loss / n_batches, total_mae / n_batches, all_predictions, all_targets
    
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
            val_loss, val_mae, _, _ = self.evaluate(self.val_loader)
            
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
    def predict_all(self, loader, denormalize=False, coords_scaler=None):
        """
        Prédit les coordonnées pour tous les sous-graphes d'un loader.
        
        Args:
            loader: DataLoader
            denormalize: Si True, dénormalise les prédictions
            coords_scaler: Scaler pour dénormaliser
        
        Returns:
            predictions, targets: Arrays numpy des prédictions et vraies valeurs
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        for batch in loader:
            batch = batch.to(self.device)
            pred = self.model(batch.x, batch.edge_index)
            
            # Extraire cellules centrales
            batch_size = batch.num_graphs
            central_predictions = []
            
            for i in range(batch_size):
                start_idx = batch.ptr[i]
                central_predictions.append(pred[start_idx])
            
            pred_central = torch.stack(central_predictions)
            
            all_predictions.append(pred_central.cpu())
            all_targets.append(batch.y.cpu())
        
        predictions = torch.cat(all_predictions, dim=0).numpy()
        targets = torch.cat(all_targets, dim=0).numpy()
        
        if denormalize and coords_scaler is not None:
            predictions = coords_scaler.inverse_transform(predictions)
            targets = coords_scaler.inverse_transform(targets)
        
        return predictions, targets
    
    @torch.no_grad()
    def predict(self, denormalize=False, coords_scaler=None):
        """
        Méthode pour compatibilité - prédit sur l'ensemble de test.

        Args:
            denormalize: Si True, dénormalise les prédictions
            coords_scaler: Scaler pour dénormaliser

        Returns:
            predictions: Array numpy des prédictions
        """
        predictions, _ = self.predict_all(self.test_loader, denormalize, coords_scaler)
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

