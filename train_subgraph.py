"""
Trainer pour le modèle avec sous-graphes locaux.
Chaque batch contient plusieurs sous-graphes indépendants.
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset, DistributedSampler

import numpy as np
import time
import os
import numpy as np
import time

class ListDataset(Dataset):
    """Wrapper minimal pour une liste de Data objects."""
    def __init__(self, lst):
        self.lst = lst
    def __len__(self):
        return len(self.lst)
    def __getitem__(self, idx):
        return self.lst[idx]


class SubgraphTrainer:
    """Classe pour entraîner le modèle sur des sous-graphes locaux."""

    def __init__(self, model, subgraphs_list, train_indices, val_indices, test_indices,
                 batch_size=32, lr=0.001, weight_decay=5e-4, device='cpu',
                 lambda_smooth=0.1, distributed=False, rank=0, world_size=1):
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.distributed = distributed

        # Gérer le model -> déplacer sur device + DDP wrap si demandé
        model = model.to(device)
        if self.distributed and self.world_size > 1:
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = DDP(model, device_ids=[int(device.split(':')[-1])])
        self.model = model

        self.lambda_smooth = lambda_smooth

        # Préparer datasets
        train_subgraphs = [subgraphs_list[i] for i in train_indices]
        val_subgraphs = [subgraphs_list[i] for i in val_indices]
        test_subgraphs = [subgraphs_list[i] for i in test_indices]

        # Wrap en dataset pour utiliser DistributedSampler
        train_dataset = ListDataset(train_subgraphs)
        val_dataset = ListDataset(val_subgraphs)
        test_dataset = ListDataset(test_subgraphs)

        if self.distributed and self.world_size > 1:
            train_sampler = DistributedSampler(train_dataset, num_replicas=self.world_size, rank=self.rank,
                                               shuffle=True)
            val_sampler = DistributedSampler(val_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False)
            test_sampler = DistributedSampler(test_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False)
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size,persistent_workers=False, sampler=train_sampler,
                                           num_workers=0, pin_memory=False)
            self.val_loader = DataLoader(val_dataset, batch_size=batch_size,persistent_workers=False, sampler=val_sampler,
                                         num_workers=0, pin_memory=False)
            self.test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler,
                                          num_workers=0, pin_memory=False)
            self.train_sampler = train_sampler
        else:
            self.train_loader = DataLoader(train_subgraphs, batch_size=batch_size, shuffle=True,
                                           num_workers=4, pin_memory=True)
            self.val_loader = DataLoader(val_subgraphs, batch_size=batch_size, shuffle=False,
                                         num_workers=2, pin_memory=True)
            self.test_loader = DataLoader(test_subgraphs, batch_size=batch_size, shuffle=False,
                                          num_workers=2, pin_memory=True)
            self.train_sampler = None

        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10)

        self.history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': [],
                        'train_smooth_loss': [], 'val_smooth_loss': []}

        if self.rank == 0:
            print(f"✓ DataLoaders créés (rank {self.rank}):")
            print(f"  Train: {len(train_dataset)} sous-graphes, {len(self.train_loader)} batches")
            print(f"  Val: {len(val_dataset)} sous-graphes, {len(self.val_loader)} batches")
            print(f"  Test: {len(test_dataset)} sous-graphes, {len(self.test_loader)} batches")

    def train_epoch(self, epoch=None):
        self.model.train()
        total_loss = total_mae = total_smooth_loss = total_main_loss = 0.0
        n_batches = 0

        if self.train_sampler is not None and epoch is not None:
            self.train_sampler.set_epoch(epoch)

        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(batch)
            main_loss = F.mse_loss(pred, batch.y)
            pred_source = pred[batch.edge_index[0]]
            pred_target = pred[batch.edge_index[1]]
            smooth_loss = torch.mean((pred_source - pred_target) ** 2)
            loss = main_loss + self.lambda_smooth * smooth_loss
            mae = F.l1_loss(pred, batch.y)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_main_loss += main_loss.item()
            total_mae += mae.item()
            total_smooth_loss += smooth_loss.item()
            n_batches += 1

        return total_loss / n_batches, total_mae / n_batches, total_smooth_loss / n_batches

    @torch.no_grad()
    def evaluate(self, loader):
        # identique à ta version, renvoie métriques et centrales
        self.model.eval()
        total_loss = total_mae = total_smooth_loss = 0.0
        n_batches = 0
        all_central_predictions = []
        all_central_targets = []

        for batch in loader:
            batch = batch.to(self.device)
            pred = self.model(batch)
            main_loss = F.mse_loss(pred, batch.y)
            mae = F.l1_loss(pred, batch.y)
            pred_source = pred[batch.edge_index[0]]
            pred_target = pred[batch.edge_index[1]]
            smooth_loss = torch.mean((pred_source - pred_target) ** 2)
            loss = main_loss + self.lambda_smooth * smooth_loss

            total_loss += loss.item()
            total_mae += mae.item()
            total_smooth_loss += smooth_loss.item()
            n_batches += 1

            batch_size = batch.num_graphs
            for i in range(batch_size):
                start_idx = batch.ptr[i]
                all_central_predictions.append(pred[start_idx].cpu())
                all_central_targets.append(batch.y[start_idx].cpu())

        all_central_predictions = torch.stack(all_central_predictions)
        all_central_targets = torch.stack(all_central_targets)

        return total_loss / n_batches, total_mae / n_batches, all_central_predictions, all_central_targets, total_smooth_loss / n_batches

    def train(self, epochs=200, early_stopping_patience=20, verbose=True):
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        start_time = time.time()
        for epoch in range(1, epochs + 1):
            train_loss, train_mae, train_smooth = self.train_epoch(epoch)
            # Validation (si distributed : on évalue sur la partition du rank ;
            # simplification : on évalue localement et on laisse rank 0 afficher)
            val_loss, val_mae, _, _, val_smooth = self.evaluate(self.val_loader)

            # Ne pas agrèger les métriques sur tous les ranks pour garder l'exemple simple.
            # On sauvegarde le modèle si rank==0
            if self.rank == 0:
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['train_mae'].append(train_mae)
                self.history['val_mae'].append(val_mae)
                self.history['train_smooth_loss'].append(train_smooth)
                self.history['val_smooth_loss'].append(val_smooth)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # sauvegarde état (si DDP -> model.module)
                    if hasattr(self.model, 'module'):
                        best_model_state = self.model.module.state_dict().copy()
                    else:
                        best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1

                if verbose and (epoch == 1 or epoch % 10 == 0 or patience_counter == 0 or epoch == 2):
                    print(f"Epoch {epoch:03d} | "
                          f"Train Loss: {train_loss:.4f} (MSE+λ·Smooth) | Train MAE: {train_mae:.4f} | "
                          f"Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f} | "
                          f"Smooth: {train_smooth:.4f}/{val_smooth:.4f} | "
                          f"Best: {best_val_loss:.4f}")

            # Synchroniser tous les processes pour être sûrs que le rank 0 ait fini d'écrire/lecture
            if self.distributed and self.world_size > 1:
                torch.distributed.barrier()

            if self.rank == 0 and patience_counter >= early_stopping_patience and epoch == 50:
                print(f"\n✓ Early stopping à l'époque {epoch}")
                break

        elapsed = time.time() - start_time
        if self.rank == 0:
            print(f"\n✓ Entraînement terminé en {elapsed / 60:.2f} minutes - Best val: {best_val_loss:.4f}")

        # Charger le meilleur modèle (localement dans ce processus si rank==0)
        if best_model_state is not None and self.rank == 0:
            if hasattr(self.model, 'module'):
                self.model.module.load_state_dict(best_model_state)
            else:
                self.model.load_state_dict(best_model_state)

        return best_model_state
    
    @torch.no_grad()
    def predict_all(self, loader, denormalize=False, coords_scaler=None):
        # identique à ta version, en s'assurant que self.model.eval() est bon
        self.model.eval()
        all_predictions = []
        all_targets = []
        for batch in loader:
            batch = batch.to(self.device)
            pred = self.model(batch)
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
        state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
        torch.save({'model_state_dict': state, 'optimizer_state_dict': self.optimizer.state_dict(), 'history': self.history}, path)
        if self.rank == 0:
            print(f"✓ Modèle sauvegardé: {path}")

    def load_model(self, path):
        ckpt = torch.load(path, map_location=self.device)
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(ckpt['model_state_dict'])
        else:
            self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.history = ckpt.get('history', self.history)
        if self.rank == 0:
            print(f"✓ Modèle chargé: {path}")


