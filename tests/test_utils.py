import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from neurocluster.predictors import FLOWHIVAE, FLOWVAE, HIVAE, VAE
from neurocluster.utils import (FLOWHIVAELoss, FLOWVAELoss, HIVAEDataset,
                                HIVAELoss, VAELoss, estimate_LL)


def test_hivaeloss():
    partitions = {
        "continuous": {
            "cont1": {"indices": slice(0, 1)},
            "cont2": {"indices": slice(3, 4)},
        },
        "categorical": {
            "cat1": {"indices": slice(1, 2), "n_categories": 3},
            "cat2": {"indices": slice(2, 3), "n_categories": 2},
        },
    }
    model = HIVAE([4, 3, 2], partitions)
    data = np.array(
        [
            [0.1, 1.0, np.nan, 7.2],
            [0.3, np.nan, 1.0, 0.2],
            [3.1, 2.0, 0.0, 9.3],
            [np.nan, 0.0, 0.0, 2.82],
        ]
    )
    mask = np.isnan(data)
    data, mask = torch.from_numpy(data).float(), torch.from_numpy(mask)
    model_output = model((data, mask))
    recon, _, _ = model_output
    loss_fn = HIVAELoss(partitions, weights=torch.Tensor([0.0, 1.0, 1.0, 0.0]))
    assert loss_fn.mu_idxs == [0, 2]
    assert loss_fn.var_idxs == [1, 3]
    assert loss_fn.out_cat_idxs == [slice(3, 6), slice(6, 8)]
    assert loss_fn.num_idxs == [0, 3]
    assert loss_fn.in_cat_idxs == [1, 2]
    recon_loss = loss_fn.recon_loss(recon, data, mask)
    loss = loss_fn(model_output, (data, mask))
    assert recon_loss.size() == torch.Size([4, 4])
    assert loss > 0.1


def test_hivaedataset():
    data = torch.randn(10, 6)
    mask = torch.rand(10, 6).bool()
    dataset = HIVAEDataset(data, mask)
    assert len(dataset) == 10
    sampled_x, sampled_y = dataset[4]
    assert sampled_x[0].size() == torch.Size([6])
    loader = DataLoader(dataset, batch_size=5)
    sampled_x, sampled_y = next(iter(loader))
    assert sampled_x[0].size() == torch.Size([5, 6])


def test_flowvaeloss():
    data = torch.randn(10, 23)
    model = FLOWVAE([23, 4, 2])
    lossfn = FLOWVAELoss()
    output = model(data)
    loss = lossfn(output, data)
    assert loss.size() == torch.Size([])


def test_flowhivaeloss():
    partitions = {
        "continuous": {
            "cont1": {"indices": slice(0, 1)},
            "cont2": {"indices": slice(3, 4)},
        },
        "categorical": {
            "cat1": {"indices": slice(1, 2), "n_categories": 3},
            "cat2": {"indices": slice(2, 3), "n_categories": 2},
        },
    }
    model = FLOWHIVAE([4, 3, 2], partitions)
    data = np.array(
        [
            [0.1, 1.0, np.nan, 7.2],
            [0.3, np.nan, 1.0, 0.2],
            [3.1, 2.0, 0.0, 9.3],
            [np.nan, 0.0, 0.0, 2.82],
        ]
    )
    mask = np.isnan(data)
    data, mask = torch.from_numpy(data).float(), torch.from_numpy(mask)
    model_output = model((data, mask))
    recon, _, _, _, _ = model_output
    loss_fn = FLOWHIVAELoss(partitions, weights=torch.Tensor([0.0, 1.0, 1.0, 0.0]))
    assert loss_fn.mu_idxs == [0, 2]
    assert loss_fn.var_idxs == [1, 3]
    assert loss_fn.out_cat_idxs == [slice(3, 6), slice(6, 8)]
    assert loss_fn.num_idxs == [0, 3]
    assert loss_fn.in_cat_idxs == [1, 2]
    recon_loss = loss_fn.recon_loss(recon, data, mask)
    loss = loss_fn(model_output, (data, mask))
    assert recon_loss.size() == torch.Size([4, 4])
    assert loss > 0.1


def test_estimate_LL():
    model = VAE([10, 3, 2])
    loss = VAELoss()
    data = torch.randn(20, 10)
    LLmean, llstd = estimate_LL(model, data, loss)
    assert LLmean < 0.0
