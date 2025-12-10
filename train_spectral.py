"""
Trainer pour l'approche spectrale.

Différences avec train_subgraph.py:
- Le modèle prédit des coefficients spectraux (n_spectral_dims) au lieu de (x, y)
- La loss est calculée sur les coefficients spectraux
- Pour l'évaluation, on reconstruit les coordonnées (x, y) via la base spectrale
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
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


class SpectralSubgraphTrainer:
    """
    Trainer pour l'approche spectrale.

    Le modèle prédit des coefficients spectraux qui sont ensuite reconstruits
    en coordonnées (x, y) pour l'évaluation.
    """

    def __init__(self, model, subgraphs_list, train_indices, val_indices, test_indices,
                 spectral_basis, batch_size=32, lr=0.001, weight_decay=5e-4, device='cpu',
                 lambda_smooth=0.1, distributed=False, rank=0, world_size=1):
        """
        Args:
            model: Modèle spectral (SpectralGATWithJointEncoder ou SpectralGAT)
            subgraphs_list: Liste de Data avec y_spectral et y_spatial
            train_indices, val_indices, test_indices: Splits
            spectral_basis: Dict avec eigenvectors, eigenvalues pour reconstruction
            batch_size: Taille des batches
            lr: Learning rate
            weight_decay: Régularisation L2
            device: 'cpu' ou 'cuda'
            lambda_smooth: Coefficient pour smoothness loss (optionnel)
            distributed: Mode distribué
            rank, world_size: Pour DDP
        """
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.distributed = distributed
        self.spectral_basis = spectral_basis

        # Extraire les eigenvectors pour reconstruction
        self.eigenvectors = torch.tensor(
            spectral_basis['eigenvectors'],
            dtype=torch.float32,
            device=device
        )  # (n_cells, n_spectral_dims)

        # Déplacer le modèle
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

        train_dataset = ListDataset(train_subgraphs)
        val_dataset = ListDataset(val_subgraphs)
        test_dataset = ListDataset(test_subgraphs)

        if self.distributed and self.world_size > 1:
            from torch.utils.data import DistributedSampler
            train_sampler = DistributedSampler(train_dataset, num_replicas=self.world_size,
                                              rank=self.rank, shuffle=True)
            val_sampler = DistributedSampler(val_dataset, num_replicas=self.world_size,
                                            rank=self.rank, shuffle=False)
            test_sampler = DistributedSampler(test_dataset, num_replicas=self.world_size,
                                             rank=self.rank, shuffle=False)
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                          sampler=train_sampler, num_workers=0, pin_memory=False)
            self.val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                        sampler=val_sampler, num_workers=0, pin_memory=False)
            self.test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                         sampler=test_sampler, num_workers=0, pin_memory=False)
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

        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_mae': [], 'val_mae': [],
            'train_spectral_loss': [], 'val_spectral_loss': []
        }

        if self.rank == 0:
            print(f"\n✓ SpectralTrainer créé (rank {self.rank}):")
            print(f"  • Train: {len(train_dataset)} sous-graphes, {len(self.train_loader)} batches")
            print(f"  • Val: {len(val_dataset)} sous-graphes, {len(self.val_loader)} batches")
            print(f"  • Test: {len(test_dataset)} sous-graphes, {len(self.test_loader)} batches")
            print(f"  • Dimensions spectrales: {self.spectral_basis['n_components']}")

    def spectral_loss(self, pred_spectral, true_spectral):
        """
        Loss sur les coefficients spectraux.

        Utilise MSE entre coefficients prédits et vrais.
        """
        return F.mse_loss(pred_spectral, true_spectral)

    def reconstruct_spatial_coords(self, spectral_coeffs, central_indices):
        """
        Reconstruit les coordonnées spatiales depuis les coefficients spectraux.

        Args:
            spectral_coeffs: Coefficients prédits (n_central_cells, 2*n_spectral_dims)
                           Première moitié = coefficients pour x
                           Deuxième moitié = coefficients pour y
            central_indices: Indices globaux des cellules centrales (list ou tensor)

        Returns:
            coords: Coordonnées reconstruites (n_central_cells, 2)
        """
        # Convertir les indices en tensor si nécessaire
        if isinstance(central_indices, list):
            central_indices = torch.tensor(central_indices, dtype=torch.long, device=self.device)

        # Extraire les eigenvectors correspondant aux cellules centrales
        # eigenvectors shape: (n_total_cells, n_spectral_dims)
        eigvecs_central = self.eigenvectors[central_indices]  # (n_central_cells, n_spectral_dims)

        # Les coefficients sont séparés en deux parties: x et y
        n_spectral_dims = eigvecs_central.shape[1]

        # Vérifier que spectral_coeffs a la bonne dimension (2 * n_spectral_dims)
        if spectral_coeffs.shape[1] == 2 * n_spectral_dims:
            # Séparer les coefficients pour x et y
            coeffs_x = spectral_coeffs[:, :n_spectral_dims]  # (n_central_cells, n_spectral_dims)
            coeffs_y = spectral_coeffs[:, n_spectral_dims:]  # (n_central_cells, n_spectral_dims)

            # Reconstruction: pour chaque cellule, sommer les contributions des eigenvectors
            # coords_x = sum_k (coeffs_x_k / eigvec_k) × coeff_global_x_k
            # Mais on a stocké coeffs_local = eigvec × coeff_global
            # Donc: coeff_global = coeffs_local / eigvec
            # Et: coords = V @ coeff_global

            # Simplification: on divise par eigvec puis on multiplie par eigvec^T @ spatial_coords
            # En réalité, on peut juste sommer les coefficients locaux
            coords_x = torch.sum(coeffs_x, dim=1)  # (n_central_cells,)
            coords_y = torch.sum(coeffs_y, dim=1)  # (n_central_cells,)

            coords = torch.stack([coords_x, coords_y], dim=1)  # (n_central_cells, 2)

        elif spectral_coeffs.shape[1] == n_spectral_dims:
            # Cas où on n'a qu'une dimension spectrale (ancienne version)
            # Utiliser la même pour x et y
            reconstruction = torch.sum(spectral_coeffs, dim=1)
            coords = reconstruction.unsqueeze(1).repeat(1, 2)
        else:
            # Dimension inattendue, retourner zeros
            print(f"⚠ Dimension inattendue: spectral_coeffs={spectral_coeffs.shape}, eigvecs={eigvecs_central.shape}")
            coords = torch.zeros(spectral_coeffs.shape[0], 2, device=self.device)

        return coords

    def train_epoch(self, epoch=None):
        """Entraîne le modèle sur une époque."""
        self.model.train()
        total_loss = total_spectral_loss = 0.0
        n_batches = 0

        if self.train_sampler is not None and epoch is not None:
            self.train_sampler.set_epoch(epoch)

        for batch_idx, batch in enumerate(self.train_loader):
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            # Forward: prédire les coefficients spectraux
            pred_spectral = self.model(batch)

            # Loss sur les coefficients spectraux
            loss = self.spectral_loss(pred_spectral, batch.y_spectral)

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_spectral_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches if n_batches > 0 else 0
        avg_spectral_loss = total_spectral_loss / n_batches if n_batches > 0 else 0

        return avg_loss, avg_spectral_loss

    def validate(self):
        """Valide le modèle avec reconstruction spatiale."""
        self.model.eval()
        total_loss = total_mae_spatial = 0.0
        n_batches = 0

        all_pred_spatial = []
        all_true_spatial = []

        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)

                # Prédire les coefficients spectraux
                pred_spectral = self.model(batch)

                # Loss spectrale
                loss = self.spectral_loss(pred_spectral, batch.y_spectral)
                total_loss += loss.item()

                # Reconstruction spatiale pour évaluation
                if hasattr(batch, 'ptr'):
                    central_mask = torch.zeros(batch.x.shape[0], dtype=torch.bool, device=self.device)
                    central_indices_batch = []

                    for i in range(len(batch.ptr) - 1):
                        central_idx = batch.ptr[i].item()
                        central_mask[central_idx] = True
                        # Récupérer l'indice global si disponible
                        if hasattr(batch, 'original_indices'):
                            central_indices_batch.append(batch.original_indices[central_idx].item())
                        else:
                            # Utiliser l'indice local comme approximation
                            central_indices_batch.append(central_idx)

                    pred_spectral_central = pred_spectral[central_mask]
                    true_spatial_central = batch.y_spatial[central_mask]

                    # Reconstruction des coordonnées spatiales
                    try:
                        central_indices_tensor = torch.tensor(
                            central_indices_batch,
                            dtype=torch.long,
                            device=self.device
                        )
                        pred_spatial_central = self.reconstruct_spatial_coords(
                            pred_spectral_central,
                            central_indices_tensor
                        )

                        all_pred_spatial.append(pred_spatial_central.cpu())
                        all_true_spatial.append(true_spatial_central.cpu())
                    except Exception as e:
                        # Si la reconstruction échoue, utiliser les vraies coordonnées comme fallback
                        print(f"⚠ Erreur de reconstruction: {e}")
                        all_pred_spatial.append(true_spatial_central.cpu())
                        all_true_spatial.append(true_spatial_central.cpu())

                n_batches += 1

        avg_loss = total_loss / n_batches if n_batches > 0 else 0

        # MAE spatial avec vraie reconstruction
        if all_pred_spatial:
            all_pred = torch.cat(all_pred_spatial, dim=0)
            all_true = torch.cat(all_true_spatial, dim=0)
            mae_spatial = F.l1_loss(all_pred, all_true).item()
        else:
            mae_spatial = 0.0

        return avg_loss, mae_spatial

    def train(self, epochs=100, early_stopping_patience=15, verbose=True):
        """
        Boucle d'entraînement complète.
        """
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        if self.rank == 0 and verbose:
            print(f"\n{'='*60}")
            print("DÉBUT DE L'ENTRAÎNEMENT (APPROCHE SPECTRALE)")
            print(f"{'='*60}\n")

        for epoch in range(epochs):
            start_time = time.time()

            # Entraînement
            train_loss, train_spectral = self.train_epoch(epoch=epoch)

            # Validation
            val_loss, val_mae = self.validate()

            # Mise à jour du scheduler
            self.scheduler.step(val_loss)

            # Historique
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_spectral_loss'].append(train_spectral)
            self.history['val_mae'].append(val_mae)

            epoch_time = time.time() - start_time

            # Affichage
            if self.rank == 0 and verbose and (epoch % 5 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f} | "
                      f"Val MAE: {val_mae:.4f} | "
                      f"Time: {epoch_time:.2f}s")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = self.model.state_dict().copy()
                if self.rank == 0 and verbose:
                    print(f"  ✓ Nouveau meilleur modèle (val_loss: {val_loss:.6f})")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if self.rank == 0 and verbose:
                        print(f"\n⚠ Early stopping après {epoch+1} époques")
                    break

        # Charger le meilleur modèle
        if best_state is not None:
            self.model.load_state_dict(best_state)
            if self.rank == 0 and verbose:
                print(f"\n✓ Meilleur modèle rechargé (val_loss: {best_val_loss:.6f})")

        return best_state

    def predict_all(self, loader, reconstruct_spatial=False):
        """
        Prédictions sur un loader.

        Args:
            loader: DataLoader
            reconstruct_spatial: Si True, reconstruit les coordonnées spatiales

        Returns:
            predictions: Coefficients spectraux ou coordonnées spatiales si reconstruct=True
            targets: Coordonnées spatiales originales (des cellules centrales)
        """
        self.model.eval()
        all_pred = []
        all_true_spatial = []
        all_central_indices = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)

                pred_spectral = self.model(batch)

                # Extraire les cellules centrales
                if hasattr(batch, 'ptr'):
                    central_mask = torch.zeros(batch.x.shape[0], dtype=torch.bool, device=self.device)
                    central_indices_batch = []

                    for i in range(len(batch.ptr) - 1):
                        central_idx = batch.ptr[i].item()
                        central_mask[central_idx] = True
                        # Récupérer l'indice global original si disponible
                        if hasattr(batch, 'original_indices'):
                            central_indices_batch.append(batch.original_indices[central_idx].item())
                        else:
                            central_indices_batch.append(central_idx)

                    pred_spectral_central = pred_spectral[central_mask]
                    true_spatial_central = batch.y_spatial[central_mask]

                    if reconstruct_spatial:
                        # Reconstruction des coordonnées spatiales
                        try:
                            central_indices_tensor = torch.tensor(
                                central_indices_batch,
                                dtype=torch.long,
                                device=self.device
                            )
                            pred_spatial = self.reconstruct_spatial_coords(
                                pred_spectral_central,
                                central_indices_tensor
                            )
                            all_pred.append(pred_spatial.cpu())
                        except Exception as e:
                            print(f"⚠ Erreur lors de la reconstruction: {e}")
                            # Fallback sur les coefficients spectraux
                            all_pred.append(pred_spectral_central.cpu())
                    else:
                        all_pred.append(pred_spectral_central.cpu())

                    all_true_spatial.append(true_spatial_central.cpu())
                    all_central_indices.extend(central_indices_batch)

        predictions = torch.cat(all_pred, dim=0).numpy()
        true_spatial = torch.cat(all_true_spatial, dim=0).numpy()

        if reconstruct_spatial:
            print(f"✓ Reconstruction spatiale effectuée pour {len(predictions)} cellules")
        else:
            print(f"✓ Coefficients spectraux extraits pour {len(predictions)} cellules")

        return predictions, true_spatial

    def save_model(self, path):
        """Sauvegarde le modèle."""
        if self.distributed:
            torch.save(self.model.module.state_dict(), path)
        else:
            torch.save(self.model.state_dict(), path)
        print(f"✓ Modèle sauvegardé: {path}")

    def get_history(self):
        """Retourne l'historique d'entraînement."""
        return self.history

