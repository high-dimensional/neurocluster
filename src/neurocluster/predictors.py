"""Predictor classes holding pytorch models for neuroNLP pipes

This module holds pytorch classifier models
used by several of pipes of the pipes.py module.
"""

import json
import pickle
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from sklearn.metrics import calinski_harabasz_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm, trange

from neurocluster.utils import (Trainer, _get_categorical_idxs,
                                _get_numerical_idxs, _get_output_num_cat_idxs)
from neurocluster.vectorizers import EntityCountVectorizer


class MLP(nn.Module):
    """A n-layer multi-layer perceptron

    An implementation of a n-layer MLP with
    ELU activation and batch normalization
    """

    def __init__(self, layer_sizes):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.mlp = self._create_model(self.layer_sizes)

    def _create_model(self, layer_sizes):
        layer_list = []
        previous_size = layer_sizes[0]
        j = 0
        for i, size in enumerate(layer_sizes[1:-1]):
            layer_list.append((f"fc{i}", nn.Linear(previous_size, size)))
            layer_list.append((f"bn{i}", nn.BatchNorm1d(size)))
            layer_list.append((f"elu{i}", nn.ELU()))
            previous_size = size
            j = i
        layer_list.append((f"fc{j+1}", nn.Linear(previous_size, layer_sizes[-1])))
        layers = OrderedDict(layer_list)
        return nn.Sequential(layers)

    def forward(self, x):
        return self.mlp(x)


# AE5 removed 21/02/2022


class AE(nn.Module):
    """Deep Autoencoder

    Arbitrary-depth autoencoder
    with batch normalization and ELU
    activation on the hidden layers
    """

    def __init__(self, layer_sizes, sigmoid_output=False):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.sigmoid_output = sigmoid_output
        self.encoder = MLP(self.layer_sizes)
        self.decoder = MLP(list(reversed(self.layer_sizes)))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return torch.sigmoid(self.decode(z)) if self.sigmoid_output else self.decode(z)


# VAE5 removed 21/02/2022


class VAE(nn.Module):
    """Variational Autoencoder

    An arbitrary-depth VAE model Bernoulli decoder
    It uses MLPs to parameterize the encoder and decoder.
    """

    def __init__(self, layer_sizes):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.encoder = MLP(self.layer_sizes[:-1])
        self.decoder = MLP(list(reversed(self.layer_sizes)))
        self.fc_mu = nn.Linear(self.layer_sizes[-2], self.layer_sizes[-1])
        self.fc_logvar = nn.Linear(self.layer_sizes[-2], self.layer_sizes[-1])

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, x):
        y = self.encoder(x)
        return self.fc_mu(y), self.fc_logvar(y)

    def decode(self, z):
        return torch.sigmoid(self.decoder(z))

    def sample(self, n_samples, parameters=False):
        z = torch.randn(n_samples, self.layer_sizes[-1])
        samples = self.decode(z)
        return samples if parameters else torch.bernoulli(samples)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        output = self.decode(z)
        return output, mu, logvar

    def reconstruct(self, x):
        output, _, _ = self.forward(x)
        return output


# VAMPVAE removed 21/02/2022


class FLOW(nn.Module):
    """RealNVP Normalizing flow model

    Based on the implementation of Dinh. 2017
    The base distribution is a unit Normal.
    """

    def __init__(self, input_size, flow_layers=1, mask_proportion=0.5):
        super().__init__()
        self.input_size = input_size
        self.layers = flow_layers
        self.mask_n = int(mask_proportion * self.input_size)
        self.non_mask_n = int(self.input_size - self.mask_n)
        self.scalers = nn.ModuleList(
            [
                MLP([self.mask_n, self.mask_n, self.non_mask_n])
                for i in range(self.layers)
            ]
        )
        self.shifters = nn.ModuleList(
            [
                MLP([self.mask_n, self.mask_n, self.non_mask_n])
                for i in range(self.layers)
            ]
        )

    def forward(self, x):
        log_jac = 0.0
        switch = True
        for layer_shift, layer_scale in zip(self.shifters, self.scalers):
            x_a, x_b = self.split(x, switch)
            scl = layer_scale(x_a)
            sft = layer_shift(x_a)
            e_b = sft + x_b * torch.exp(scl)
            e_a = x_a
            log_jac += layer_scale(x_a).sum(dim=-1)
            x = self.concat(e_a, e_b, switch)
            switch = not switch

        log_p_e = torch.sum(-0.5 * (1.837 + torch.pow(x, 2)), dim=-1)
        log_prob = log_p_e + log_jac
        return x, log_prob

    def concat(self, x_a, x_b, switch):
        if switch:
            return torch.cat([x_a, x_b], dim=-1)
        else:
            return torch.cat([x_b, x_a], dim=-1)

    def split(self, x, switch):
        if switch:
            return x[:, : self.mask_n], x[:, self.mask_n :]
        else:
            return x[:, self.non_mask_n :], x[:, : self.non_mask_n]

    def inverse(self, e):
        switch = self.layers % 2 == 1
        for layer_shift, layer_scale in zip(
            reversed(self.shifters), reversed(self.scalers)
        ):
            x_a, x_b = self.split(e, switch)
            scl = layer_scale(x_a)
            sft = layer_shift(x_a)
            e_b = (x_b - sft) * torch.exp(-scl)
            e_a = x_a
            e = self.concat(e_a, e_b, switch)
            switch = not switch
        return e

    def sample(self, N):
        e = torch.randn(N, self.input_size)
        return self.inverse(e)


class FLOWVAE(VAE):
    """A bernoulli VAE with a normalizing flow prior"""

    def __init__(self, layer_sizes, flow_layers=2, mask_proportion=0.5):
        super().__init__(layer_sizes)
        self.prior = FLOW(
            layer_sizes[-1], flow_layers=flow_layers, mask_proportion=mask_proportion
        )

    def sample(self, n_samples, probs=False):
        z = self.prior.sample(n_samples)
        samples = self.decode(z)
        return samples if probs else torch.bernoulli(samples)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        _, prior_log_prob = self.prior(z)
        output = self.decode(z)
        return output, mu, logvar, z, prior_log_prob

    def reconstruct(self, x):
        output, _, _, _, _ = self.forward(x)
        return output


# BBVAE5 removed 21/02/2022
# DocEmbeddingModel removed 16/11/23

# CatVAE removed 21/02/2022


class HIVAE(nn.Module):
    """Basic HIVAE implementation

    Based on Nazabal et al. 2018
    """

    def __init__(self, hidden_layer_sizes, partition_dict):
        super().__init__()
        self.partition_dict = partition_dict
        self.hidden_layer_sizes = hidden_layer_sizes
        self.input_size, self.output_sizes = self._get_input_output_sizes(
            self.partition_dict
        )
        (
            self.out_mu_idxs,
            self.out_var_idxs,
            self.out_cat_idxs,
        ) = _get_output_num_cat_idxs(self.partition_dict)
        self.encoder = MLP([self.input_size] + self.hidden_layer_sizes[:-1])
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_layer_sizes[-1], self.hidden_layer_sizes[-2]),
            nn.BatchNorm1d(self.hidden_layer_sizes[-2]),
            nn.ELU(),
        )
        self.fc_mu = nn.Linear(self.hidden_layer_sizes[-2], self.hidden_layer_sizes[-1])
        self.fc_logvar = nn.Linear(
            self.hidden_layer_sizes[-2], self.hidden_layer_sizes[-1]
        )
        self.cat_idxs, self.cat_index_classes = self._get_cat_idxs(self.partition_dict)
        self.num_idxs = self._get_cont_idxs(self.partition_dict)
        self.decoder_heads = MultiHeadLayer(
            self.hidden_layer_sizes[-2], self.output_sizes
        )

    def _preprocess_X(self, X, mask):
        """replace nans in continuous features with zero,
        one-hot categorical features"""
        num_mask = mask[:, self.num_idxs]
        num_X = X[:, self.num_idxs]
        num_X[num_mask] = 0.0
        cat_X = X[:, self.cat_idxs]
        cat_X = torch.cat(
            [self._one_hot(cat_X[:, i], i) for i in range(cat_X.shape[1])], dim=1
        )
        return torch.cat((num_X, cat_X), dim=1)

    def _one_hot(self, X, index):
        """one hot categorical column of data"""
        new_index = self.cat_idxs[0] + index
        n_classes = self.cat_index_classes[new_index]
        new_X = torch.nan_to_num(X, nan=n_classes).long()
        one_hot_cats = F.one_hot(new_X, num_classes=n_classes + 1)
        one_hot_cats[:, -1] = 0
        return one_hot_cats.float()

    def _get_cat_idxs(self, partition_dict):
        return _get_categorical_idxs(partition_dict)

    def _get_cont_idxs(self, partition_dict):
        return _get_numerical_idxs(partition_dict)

    def _get_input_output_sizes(self, partition_dict):
        n_num_features = len(partition_dict["continuous"])
        n_extra_nan_features = len(partition_dict["categorical"])
        total_categories = sum(
            [
                cat_dict["n_categories"]
                for cat_dict in partition_dict["categorical"].values()
            ]
        )
        total_input_size = n_num_features + n_extra_nan_features + total_categories
        num_out_sizes = [2 for i in range(n_num_features)]
        index_n_cat = [
            (d["indices"].start, d["n_categories"])
            for d in partition_dict["categorical"].values()
        ]
        cat_out_sizes = sorted(index_n_cat, key=lambda x: x[0])
        cat_out_sizes = [j for i, j in cat_out_sizes]
        output_sizes = num_out_sizes + cat_out_sizes
        return total_input_size, output_sizes

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, x, mask):
        x_processed = self._preprocess_X(x, mask)
        y = self.encoder(x_processed)
        return self.fc_mu(y), self.fc_logvar(y)

    def decode(self, z):
        shared_y = self.decoder(z)
        multi_head_output = self.decoder_heads(shared_y)
        return multi_head_output

    def sample(self, n_samples):
        z = torch.randn(n_samples, self.hidden_layer_sizes[-1])
        samples = self.decode(z)
        return self._sample_output_dists(samples)

    def reconstruct(self, x_mask_tuple, return_cat_prob=False):
        raw_output, _, _ = self.forward(x_mask_tuple)
        return self._sample_output_dists(raw_output, return_cat_prob=return_cat_prob)

    def _sample_output_dists(self, raw_output, return_cat_prob=False):
        mu, logvar = raw_output[:, self.out_mu_idxs], raw_output[:, self.out_var_idxs]
        cont_samples = torch.normal(mu, torch.sqrt(torch.exp(logvar)))
        if return_cat_prob:
            cat_samples = torch.cat(
                [
                    F.softmax(raw_output[:, idxs], dim=1).squeeze().float()
                    for idxs in self.out_cat_idxs
                ],
                dim=1,
            )
        else:
            cat_samples = torch.stack(
                [
                    torch.multinomial(F.softmax(raw_output[:, idxs], dim=1), 1)
                    .squeeze()
                    .float()
                    for idxs in self.out_cat_idxs
                ],
                dim=1,
            )
        return torch.cat((cont_samples, cat_samples), dim=1)

    def fill_missing(self, x_mask_tuple):
        reconstruction = self.reconstruct(x_mask_tuple, return_cat_prob=False)
        x, mask = x_mask_tuple
        x[mask] = reconstruction[mask]
        return x

    def forward(self, x_mask_tuple):
        x, mask = x_mask_tuple
        mu, logvar = self.encode(x, mask)
        z = self.reparameterize(mu, logvar)
        output = self.decode(z)
        return output, mu, logvar


class MultiHeadLayer(nn.Module):
    """A simple multi-head layer that takes
    a shared input that is read by multiple
    head networks, with a concatenated total output

    NOTE: batchnorm and activation function is applied
    first, then the fully-connnected layer, as we assume
    input is raw layer output.
    """

    def __init__(self, input_size, output_sizes):
        super().__init__()
        self.input_size = input_size
        self.output_sizes = output_sizes
        self.mlp_list = self._make_module_list()

    def _make_module_list(self):
        modules = []
        for outsize in self.output_sizes:
            module = nn.Sequential(
                nn.Linear(self.input_size, 2 * outsize),
                nn.BatchNorm1d(2 * outsize),
                nn.ELU(),
                nn.Linear(2 * outsize, outsize),
            )
            modules.append(module)
        module_list = nn.ModuleList(modules)
        return module_list

    def forward(self, x):
        return torch.cat([mlp(x) for mlp in self.mlp_list], dim=1)


class FLOWHIVAE(HIVAE):
    """A HIVAE with a normalizing flow prior"""

    def __init__(
        self, hidden_layer_sizes, partition_dict, flow_layers=2, mask_proportion=0.5
    ):
        super().__init__(hidden_layer_sizes, partition_dict)
        self.prior = FLOW(
            self.hidden_layer_sizes[-1],
            flow_layers=flow_layers,
            mask_proportion=mask_proportion,
        )

    def sample(self, n_samples):
        z = self.prior.sample(n_samples)
        samples = self.decode(z)
        return self._sample_output_dists(samples)

    def reconstruct(self, x_mask_tuple, return_cat_prob=False):
        raw_output, _, _, _, _ = self.forward(x_mask_tuple)
        return self._sample_output_dists(raw_output, return_cat_prob=return_cat_prob)

    def forward(self, x_mask_tuple):
        x, mask = x_mask_tuple
        mu, logvar = self.encode(x, mask)
        z = self.reparameterize(mu, logvar)
        _, prior_log_prob = self.prior(z)
        output = self.decode(z)
        return output, mu, logvar, z, prior_log_prob


class AttentionLayer(nn.Module):
    """Basic attention layer

    Given a sequence of N X M dimensional inputs, outputs
    a M-dimensional vector which is the sum of input vectors
    weighted by the attention mechanism.

    Hierarchical Attention Networks for Document Classification,
    Yang et. al., DOI: 10.18653/v1/N16-1174

    Attributes
    ----------
    attention_size : int
        the number of features in the input
    """

    def __init__(self, input_size):
        """
        Parameters
        ----------
        input_size : int
            the number of input features, to set the attention size
        """
        super(AttentionLayer, self).__init__()
        self.attention_size = input_size
        self.linear = nn.Linear(input_size, input_size)
        self.context = nn.Linear(input_size, 1, bias=False)

    def forward(self, X):
        """
        Parameters
        ----------
        X : torch tensor, shape (batch_size, seq_len, input_size)
             the input to the model

        Returns
        -------
        weighted_X : torch tensor, shape (batch_size, input_size)
            the predicted output from the layer, the weighted sum input vectors
        attention_weights : torch tensor, shape (batch_size, seq_len)
            the attention weights for each vector in the input sequence
        """
        u = torch.tanh(self.linear(X))
        attention_scores = self.context(u)
        attention_weights = F.softmax(attention_scores, dim=1)
        weighted_X = X * attention_weights
        return weighted_X.sum(dim=1), attention_weights


class AttentionAE(nn.Module):
    """Attention Autoencoder

    Arbitrary-depth autoencoder
    with batch normalization and ELU
    activation on the hidden layers.

    Can take arbitrary-length sequence of vectors
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        hidden_size_2,
        latent_size,
        output_size,
        sigmoid_output=False,
    ):
        super().__init__()
        self.attention_layer = AttentionLayer(input_size)
        self.sigmoid_output = sigmoid_output
        self.encoder = MLP([input_size, hidden_size, hidden_size_2, latent_size])
        self.decoder = MLP([latent_size, hidden_size_2, hidden_size, output_size])

    def encode(self, x):
        y, _ = self.attention_layer(x)
        return self.encoder(y)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return torch.sigmoid(self.decode(z)) if self.sigmoid_output else self.decode(z)
