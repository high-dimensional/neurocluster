#!/usr/bin/env python
"""Report Autoencoder training script.

This scripts reads a docbin of processed reports and trains an AE model on the agglomerated features.
The AE model embeds the data into a 2D latent space, and saves this data in extra columns in the CSV
alongside the model weights.
"""
import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
import torch
# from neuronlp.custom_pipes import *
from sklearn.model_selection import train_test_split
from spacy.tokens import DocBin
from torch.utils.data import DataLoader, TensorDataset

from neurocluster.predictors import AE
from neurocluster.utils import Trainer
from neurocluster.vectorizers import EntityCountVectorizer, FeatureClusterer


def load_input(args):
    """load in data or take input from stdin"""
    bin = DocBin().from_disk(args.docbin)
    df = pd.read_csv(args.data, low_memory=False)
    model = spacy.load(args.model)
    docs = bin.get_docs(model.vocab)
    return df, docs, model


def vectorize_data(docs):
    """perform the necessary transformation on the input data"""

    classes_to_model = [
        "pathology-cerebrovascular",
        "pathology-congenital-developmental",
        "pathology-csf-disorders",
        "pathology-endocrine",
        "pathology-haemorrhagic",
        "pathology-infectious",
        "pathology-inflammatory-autoimmune",
        "pathology-ischaemic",
        "pathology-metabolic-nutritional-toxic",
        "pathology-musculoskeletal",
        "pathology-neoplastic-paraneoplastic",
        "pathology-neurodegenerative-dementia",
        "pathology-opthalmological",
        "pathology-traumatic",
        "pathology-treatment",
        "pathology-vascular",
    ]
    vectr = EntityCountVectorizer(
        class_names=classes_to_model,
        ngram_range=(1, 3),
        binary=True,
        max_features=7000,
        min_df=5,
    )
    vecs = vectr.fit_transform(docs)
    vocab = vectr.internal_vectorizer.vocabulary_
    return vecs, vocab, vectr


def agglomerate_vectors(data, model, vocab, n_groups=100):
    word_vec_func = lambda x: model(x).vector
    clus = FeatureClusterer(n_groups)
    clus.fit(vocab, word_vec_func)
    vec_array = np.array(data)
    agglo_vecs = clus.transform(vec_array)
    return agglo_vecs, clus


def train_model(vectors, n_epochs=10, batch_size=256):
    output_size = vectors.shape[1]
    modl = AE([output_size, 256, 64, 2], sigmoid_output=True)
    trainX, validX = train_test_split(vectors)
    torch_train, torch_valid = (
        torch.from_numpy(trainX).float(),
        torch.from_numpy(validX).float(),
    )

    trainset = TensorDataset(torch_train, torch_train)
    validset = TensorDataset(torch_valid, torch_valid)
    trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
    validloader = DataLoader(validset, shuffle=True, batch_size=batch_size)

    trainer = Trainer(
        loss_criterion="mse",
        lr=1e-4,
        use_gpu=True,
        weight_decay=0.0,
    )

    losses = trainer(
        modl,
        trainloader,
        validloader,
        max_epochs=n_epochs,
        track_best_model=False,
        print_loss=True,
        tqdm=False,
    )
    return modl


def plot_result(data_to_plot):
    """plot the results"""
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.scatterplot(
        data=data_to_plot,
        x="_X",
        y="_Y",
        hue="asserted-pathology-treatment",
        s=11,
        ax=ax,
        legend=True,
    )
    return fig


def output_results(dir_, vectorizer, cluster_feat, model, data, figure):
    """output analysis, save to file or send to stdout"""
    with open(dir_ / "report_vectorizer.pkl", "wb") as file:
        pickle.dump(vectorizer.__dict__, file)
    with open(dir_ / "feature_clusterer.pkl", "wb") as file:
        pickle.dump(cluster_feat.__dict__, file)
    torch.save(model.state_dict(), dir_ / "model.bin")
    data.to_csv(dir_ / "report_data.csv", index=False)
    figure.savefig(dir_ / "embedding.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("docbin", help="docbin of processed spacy docs", type=Path)
    parser.add_argument(
        "model", help="path to nlp model - must have word vectors", type=Path
    )
    parser.add_argument("data", help="path to source data CSV", type=Path)
    parser.add_argument(
        "-a",
        "--aggroups",
        help="the number of feature aggregation groups",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "-o", "--outdir", help="output directory", type=Path, default=Path.cwd()
    )
    parser.add_argument(
        "--merge_agglomeration",
        help="concatenate both the input and agglomeration features",
        action="store_true",
    )
    args = parser.parse_args()
    if not args.outdir.exists():
        args.outdir.mkdir()

    data, docs, model = load_input(args)
    transformed_data, vocab, vectorizer = vectorize_data(docs)
    agglomerated_data, agglomerator = agglomerate_vectors(
        transformed_data, model, vocab, n_groups=args.aggroups
    )
    if args.merge_agglomeration:
        agglomerated_data = np.concatenate(
            (transformed_data, agglomerated_data), axis=1
        )
    modl = train_model(agglomerated_data, n_epochs=300, batch_size=1024)
    test_input = torch.from_numpy(agglomerated_data).float()
    embedded_vectors = modl.encode(test_input).detach().numpy()
    data["_X"] = embedded_vectors[:, 0]
    data["_Y"] = embedded_vectors[:, 1]
    fig = plot_result(data)
    output_results(args.outdir, vectorizer, agglomerator, modl, data, fig)
    print(args.outdir)


if __name__ == "__main__":
    main()
