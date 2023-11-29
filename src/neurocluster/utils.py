"""Utility functions and classes for neuroCluster

Including training class and custom loss functions

"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from numba import jit
from torch import nn, optim
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange

# lbeta removed 21/02/2022


class VAELoss:
    def __init__(self, kld_weight=1.0):
        self.recon_loss = nn.BCELoss(reduction="none")
        self.kld_weight = kld_weight

    def kld_loss(self, model_output):
        _, mu, log_var = model_output
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
        return kld_loss

    def __call__(self, model_output, x, average=True):
        x_recon, mu, log_var = model_output
        reconstruction_loss = self.recon_loss(x_recon, x).sum(dim=-1)
        kld_loss = self.kld_loss(model_output)
        loss = reconstruction_loss + self.kld_weight * kld_loss
        if average:
            return loss.mean(dim=0)
        else:
            return loss


# VAMPVAE removed 21/02/2022


class FLOWVAELoss:
    def __init__(self, kld_weight=1.0):
        self.recon_loss = nn.BCELoss(reduction="none")
        self.kld_weight = kld_weight

    def kld_loss(self, model_output):
        _, mu, log_var, z, log_p_z = model_output
        # log_q = -0.5 * torch.sum(log_var +1.837+ torch.pow(z - mu, 2) / log_var.exp(), dim=-1)
        J = log_var.size(-1)
        log_q = -0.5 * (2.837 * J + log_var.sum(dim=-1))
        kld_loss = -(log_p_z - log_q)
        return kld_loss

    def __call__(self, model_output, x, average=True):
        x_recon, _, _, _, _ = model_output
        reconstruction_loss = self.recon_loss(x_recon, x).sum(dim=-1)
        kld_loss = self.kld_loss(model_output)
        loss = reconstruction_loss + self.kld_weight * kld_loss
        if average:
            return loss.mean(dim=0)
        else:
            return loss


# CatVAELoss removed 21/02/2022


class HIVAELoss:
    def __init__(self, partitions, kld_weight=1.0, weights=None):
        self.kld_weight = kld_weight
        self.partitions = partitions
        self.weights = weights
        self.mu_idxs, self.var_idxs, self.out_cat_idxs = self._get_output_idxs()
        self.num_idxs, self.in_cat_idxs = self._get_input_idxs()

    def kld_loss(self, model_output):
        _, mu, log_var = model_output
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
        return kld_loss

    def recon_loss(self, x_out, x_target, mask):
        x_target[mask] = 0.0
        num_losses = torch.stack(
            [
                F.gaussian_nll_loss(
                    x_out[:, j],
                    x_target[:, i],
                    torch.exp(x_out[:, k]),
                    reduction="none",
                )
                for i, (j, k) in zip(self.num_idxs, zip(self.mu_idxs, self.var_idxs))
            ],
            dim=1,
        )
        cat_losses = torch.stack(
            [
                F.cross_entropy(x_out[:, i], x_target[:, j].long(), reduction="none")
                for i, j in zip(self.out_cat_idxs, self.in_cat_idxs)
            ],
            dim=1,
        )
        all_losses = torch.cat((num_losses, cat_losses), dim=1)

        if self.weights is not None:
            all_losses = all_losses.mul(self.weights)
        all_losses[mask] = 0.0
        return all_losses

    def _get_output_idxs(self):
        return _get_output_num_cat_idxs(self.partitions)

    def _get_input_idxs(self):
        num_idxs = _get_numerical_idxs(self.partitions)
        cat_idxs, _ = _get_categorical_idxs(self.partitions)
        return num_idxs, cat_idxs

    def __call__(self, model_output, target, average=True):
        x_out, mu, log_var = model_output
        x_target, mask = target
        reconstruction_loss = self.recon_loss(x_out, x_target, mask).sum(dim=-1)
        kld_loss = self.kld_loss(model_output)
        loss = reconstruction_loss + self.kld_weight * kld_loss
        if average:
            return loss.mean(dim=0)
        else:
            return loss


class FLOWHIVAELoss(HIVAELoss):
    def __init__(self, partitions, kld_weight=1.0, weights=None):
        super().__init__(partitions, kld_weight, weights)

    def kld_loss(self, model_output):
        _, mu, log_var, z, log_p_z = model_output
        J = log_var.size(-1)
        log_q = -0.5 * (2.837 * J + log_var.sum(dim=-1))
        kld_loss = -(log_p_z - log_q)
        return kld_loss

    def __call__(self, model_output, target, average=True):
        x_out, mu, log_var, _, _ = model_output
        x_target, mask = target
        reconstruction_loss = self.recon_loss(x_out, x_target, mask).sum(dim=-1)
        kld_loss = self.kld_loss(model_output)
        loss = reconstruction_loss + self.kld_weight * kld_loss
        if average:
            return loss.mean(dim=0)
        else:
            return loss


# BBVAELoss removed 21/02/2022


class Trainer:
    def __init__(
        self,
        loss_criterion="mse",
        use_gpu=True,
        lr=1e-4,
        weight_decay=0.0,
        kld_weight=1.0,
        model_dir=".",
        is_hivae=False,
    ):
        self.device = torch.device(
            "cuda" if (torch.cuda.is_available() and use_gpu) else "cpu"
        )
        self.lr = lr
        self.wd = weight_decay
        self.optimizer = None
        self.loss_criterion = loss_criterion
        self.loss_types = {
            "mse": nn.MSELoss(),
            "bce": nn.BCELoss(),
            "cross_entropy": nn.CrossEntropyLoss(),
            "vae": VAELoss(kld_weight=kld_weight),
        }
        if loss_criterion in self.loss_types.keys():
            self.loss_fn = self.loss_types[loss_criterion]
        else:
            self.loss_fn = loss_criterion
        self.model_dir = Path(model_dir)
        self.is_hivae = is_hivae

    def __call__(
        self,
        model,
        train_loader,
        valid_loader,
        max_epochs=100,
        track_kld=False,
        track_best_model=False,
        print_loss=False,
        tqdm=True,
    ):
        start_time = time.time()
        model = model.to(self.device)
        self.optimizer = optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.wd
        )
        train_losses, valid_losses = [], []

        kld_valid_loss = []

        current_best_loss = 1e7

        n_training_batches = len(train_loader)
        n_validation_batches = len(valid_loader)

        iterator = trange(max_epochs) if tqdm else range(max_epochs)
        for epoch in iterator:
            train_loss = self.train_step(model, train_loader)
            epoch_train_loss = train_loss / n_training_batches
            train_losses.append(epoch_train_loss)
            valid_loss = self.validation_step(model, valid_loader)
            epoch_valid_loss = valid_loss / n_validation_batches
            valid_losses.append(epoch_valid_loss)

            if track_kld:
                kld_loss = self.kld_track_step(model, valid_loader)
                epoch_kld_loss = kld_loss / n_validation_batches
                kld_valid_loss.append(epoch_kld_loss)

            if print_loss:
                if track_kld:
                    print(
                        "====> Epoch: {} train loss: {:.4f} valid loss: {:.4f} kld loss: {:.4f}".format(
                            epoch, epoch_train_loss, epoch_valid_loss, epoch_kld_loss
                        )
                    )
                else:
                    print(
                        "====> Epoch: {} train loss: {:.4f} valid loss: {:.4f}".format(
                            epoch, epoch_train_loss, epoch_valid_loss
                        )
                    )

            if track_best_model and (epoch_valid_loss < current_best_loss):
                current_best_loss = epoch_valid_loss
                self.save_model("model-best", model)

        model.cpu()
        end_time = time.time()
        print("training time: {} seconds".format(end_time - start_time))

        loss_dict = {"train_losses": train_losses, "valid_losses": valid_losses}
        if track_kld:
            loss_dict["kld_losses"] = kld_valid_loss

        self.save_model("model-last", model)

        return loss_dict

    def train_step(self, model, train_loader):
        train_loss = 0
        model.train()
        for data, target in train_loader:
            data, target = self._data_to_device(data), self._data_to_device(target)
            self.optimizer.zero_grad()
            output = model(data)
            batch_loss = self.loss_fn(output, target)
            batch_loss.backward()
            train_loss += batch_loss.item()
            self.optimizer.step()
        return train_loss

    def validation_step(self, model, validation_loader):
        valid_loss = 0
        model.eval()
        for data, target in validation_loader:
            data, target = self._data_to_device(data), self._data_to_device(target)
            output = model(data)
            batch_loss = self.loss_fn(output, target)
            valid_loss += batch_loss.item()
        return valid_loss

    def kld_track_step(self, model, validation_loader):
        kld_loss = 0
        model.eval()
        for data, target in validation_loader:
            data, target = self._data_to_device(data), self._data_to_device(target)
            output = model(data)
            batch_kld_loss = (
                self.loss_fn.kld_loss(output).mean(dim=0)
                if self.is_hivae
                else self.loss_fn.kld_loss(output)
            )
            kld_loss += batch_kld_loss.item()
        return kld_loss

    def _data_to_device(self, data):
        if self.is_hivae:
            data_x, data_y = data
            data_x, data_y = data_x.to(self.device), data_y.to(self.device)
            return data_x, data_y
        else:
            data = data.to(self.device)
            return data

    def save_model(self, subpath, model):
        if self.model_dir.exists() and self.model_dir.is_dir():
            model_path = self.model_dir / subpath
            model_path.mkdir(parents=False, exist_ok=True)
            torch.save(model.state_dict(), model_path / "model.bin")
            model_config = {
                "class": str(type(model)),
                "class_name": type(model).__name__,
            }
            with open(model_path / "model_config.json", "w") as file:
                json.dump(model_config, file)
        else:
            raise Exception("Model directory not found")

    def test(self, model, test_loader):
        model.eval()
        with torch.no_grad():
            test_loss = self.validation_step(model, test_loader)

        return test_loss


@jit(nopython=True, parallel=True)
def numba_cluster_feat_mccs(cluster_id, feature_vecs, cluster_vec):
    """numba-accelerated matthews correlation coefficient calculation
    for finding base features most heavily correlated with cluster ids"""
    mccs = []
    for feature_idx in range(feature_vecs.shape[1]):
        in_cluster = cluster_vec == cluster_id
        has_feature = feature_vecs[:, feature_idx] > 0.0
        tp = np.logical_and(in_cluster, has_feature).sum()
        fp = np.logical_and(np.logical_not(in_cluster), has_feature).sum()
        tn = np.logical_and(
            np.logical_not(in_cluster), np.logical_not(has_feature)
        ).sum()
        fn = np.logical_and(in_cluster, np.logical_not(has_feature)).sum()
        num = tp * tn - fp * fn
        denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        MCC = num / denom
        mccs.append(MCC)
    mccs = np.array(mccs)
    return mccs


class HIVAEDataset(Dataset):
    """Custom pytorch dataset for use with HIVAE models"""

    def __init__(self, X, mask):
        super().__init__()
        self.data = self._check_data(X)
        self.mask = self._check_data(mask)

    def _check_data(self, tensor):
        if isinstance(tensor, torch.Tensor) and (
            tensor.dtype in [torch.float32, torch.bool]
        ):
            return tensor
        else:
            raise TypeError(
                "Invalid type, only 32bit floating-point and boolean pytorch tensors allowed"
            )

    def __len__(self):
        return self.data.size()[0]

    def __getitem__(self, idx):
        data_sample = self.data[idx]
        mask_sample = self.mask[idx]
        return (data_sample, mask_sample), (data_sample, mask_sample)


def estimate_LL(model, dataset, lossfunc, n_samples=100, is_hivae=False):
    """Use importance sampling to estimate the
    log-likelihood of a model
    """
    model.eval()
    with torch.no_grad():
        elbos = []
        for i in tqdm(range(n_samples)):
            if is_hivae:
                sample_output = model((dataset, torch.isnan(dataset)))
                sample_elbo = -lossfunc(
                    sample_output, (dataset, torch.isnan(dataset)), average=False
                )
            else:
                sample_output = model(dataset)
                sample_elbo = -lossfunc(sample_output, dataset, average=False)
            elbos.append(sample_elbo)
    elbos = torch.stack(elbos, dim=-1)
    sample_iwll = torch.logsumexp(elbos, 1)
    sample_iwll = sample_iwll - torch.log(
        n_samples * torch.ones(size=sample_iwll.size())
    )
    return sample_iwll.mean(), sample_iwll.std()


def _get_output_num_cat_idxs(partitions):
    num_idxs = _get_numerical_idxs(partitions)
    mu_idxs = [2 * i for i, _ in enumerate(num_idxs)]
    var_idxs = [2 * i + 1 for i, _ in enumerate(num_idxs)]
    cat_idxs, n_classes = _get_categorical_idxs(partitions)
    new_cat_idxs = []
    j = max(var_idxs)
    for n in cat_idxs:
        new_cat_idxs.append(slice(j, j + n_classes[n]))
        j += n_classes[n]
    return mu_idxs, var_idxs, new_cat_idxs


def _get_numerical_idxs(partition_dict):
    cont_dict = partition_dict["continuous"]
    return sorted([item["indices"].start for item in cont_dict.values()])


def _get_categorical_idxs(partition_dict):
    cat_dict = partition_dict["categorical"]
    index_classes = {
        cat_dict["indices"].start: cat_dict["n_categories"]
        for cat, cat_dict in cat_dict.items()
    }
    return sorted(list(index_classes.keys())), index_classes
