import numpy as np
import pytest
import spacy
import torch
from numpy import ndarray
from sklearn.cluster import KMeans
from spacy.tokens import DocBin
from torch.cuda import device

from neurocluster.predictors import *
from neurocluster.vectorizers import EntityCountVectorizer

report_sample = [
    "Clinical Indications: Post-op tumour excision. ?residual tumour. pre and post contrast MRI head please on Thursday. Findings: Comparison is made with the previous MR studies. There has been interval resection of the previously shown enhancing tumour centred on the superior aspects of the posterior ethmoid and sphenoid sinuses, with involvement of the anterior cranial fossa. Heterogeneous signal is demonstrated within the surgical bed, along with areas of faint T1 shortening and curvilinear enhancement, which are likely postsurgical in nature at this stage. Allowing for these changes, no residual or recurrent tumour is convincingly demonstrated, although this will be clarified on subsequent follow-up imaging. Note is made of mild thickening and enhancement of the anteroinferior aspect of the falx cerebri. The remaining intracranial appearances are stable. The previous left frontal resection cavity is again shown. Note is also again made of a few non-specific foci of T2 hyperintensity within the cerebral white matter. Postsurgical changes are noted in the paranasal sinuses."
]


def test_mlp():
    mlp = MLP([20, 3, 2])
    data = torch.randn(10, 20)
    out = mlp(data)
    assert out.size() == (10, 2)


def test_hivae():
    data = np.array(
        [[0.1, 1.0, np.nan], [0.3, np.nan, 1.0], [3.1, 2.0, 0.0], [np.nan, 0.0, 0.0]]
    )
    mask = np.isnan(data)
    data, mask = torch.from_numpy(data).float(), torch.from_numpy(mask)
    partitions = {
        "continuous": {"cont1": {"indices": slice(0, 1)}},
        "categorical": {
            "cat1": {"indices": slice(1, 2), "n_categories": 3},
            "cat2": {"indices": slice(2, 3), "n_categories": 2},
        },
    }

    layers = [5, 3, 2]
    model = HIVAE(layers, partitions)
    output, _, _ = model((data, mask))
    assert model.cat_idxs == [1, 2]
    assert model.input_size == sum(model.output_sizes) + 1
    assert output.shape == (4, 7)
    samples = model.sample(4)
    assert samples.size() == data.size()
    assert model.reconstruct((data, mask)).size() == data.size()
    assert model.reconstruct((data, mask), return_cat_prob=True).size() == torch.Size(
        [4, 6]
    )
    filled_data = model.fill_missing((data, mask))
    assert filled_data.size() == data.size()
    assert (filled_data[0, 2] == 0.0) or (filled_data[0, 2] == 1.0)


def test_hivae_flow():
    data = np.array(
        [[0.1, 1.0, np.nan], [0.3, np.nan, 1.0], [3.1, 2.0, 0.0], [np.nan, 0.0, 0.0]]
    )
    mask = np.isnan(data)
    data, mask = torch.from_numpy(data).float(), torch.from_numpy(mask)
    partitions = {
        "continuous": {"cont1": {"indices": slice(0, 1)}},
        "categorical": {
            "cat1": {"indices": slice(1, 2), "n_categories": 3},
            "cat2": {"indices": slice(2, 3), "n_categories": 2},
        },
    }

    layers = [5, 3, 2]
    model = FLOWHIVAE(layers, partitions)
    output, _, _, _, _ = model((data, mask))
    assert model.cat_idxs == [1, 2]
    assert model.input_size == sum(model.output_sizes) + 1
    assert output.shape == (4, 7)
    samples = model.sample(4)
    assert samples.size() == data.size()
    assert model.reconstruct((data, mask)).size() == data.size()
    assert model.reconstruct((data, mask), return_cat_prob=True).size() == torch.Size(
        [4, 6]
    )
    filled_data = model.fill_missing((data, mask))
    assert filled_data.size() == data.size()
    assert (filled_data[0, 2] == 0.0) or (filled_data[0, 2] == 1.0)


def test_flowvae():
    flow = FLOWVAE([20, 3, 2])
    data = torch.randn(10, 20)
    recon, mu, logvar, z, logp = flow(data)
    samples = flow.sample(10)
    assert recon.shape == (10, 20)
    assert samples.shape == (10, 20)
    flow = flow.to("cuda")
    data = data.to("cuda")
    gpu_output = flow.reconstruct(data)
    assert gpu_output.device == torch.device("cuda:0")


def test_flow():
    flow = FLOW(10, flow_layers=3, mask_proportion=0.4)
    data = torch.randn(15, 10)
    z, log = flow(data)
    assert z.shape == data.shape
    samples = flow.sample(4)
    assert samples.shape == (4, 10)


def test_attention_layer():
    att_layer = AttentionLayer(300)
    vecs = torch.randn((1, 10, 300))
    vals, weights = att_layer(vecs)
    assert weights.shape == torch.Size([1, 10, 1])
    assert vals.shape == torch.Size([1, 300])
