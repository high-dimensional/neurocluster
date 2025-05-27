import numpy as np
import pandas as pd
import pytest
import spacy
from negspacy.negation import Negex

from neurocluster.vectorizers import (EntityCountVectorizer, FeatureClusterer,
                                      MultiTransformer)

report_sample = [
    "placeholder"
]


def test_multitransformer():
    data = {
        "C": ["T", "Y", "T", np.nan, "Y", "Y", "Y"],
        "A": ["cat1", "cat2", "cat3", "cat1", "cat1", "cat2", np.nan],
        "B": [0.1, -0.5, 6, 10, 2, 1, np.nan],
        "D": [np.nan, 0.1, 0.001, 454, 1.0, 5.0, 9.0],
    }
    df = pd.DataFrame(data=data).astype(
        {"A": "category", "B": float, "C": "category", "D": float}
    )
    vectorizer = MultiTransformer(standard_scaling=True)
    vectorizer.fit(df, lognormal_columns=["D"])
    fitted_data = vectorizer.transform(df, return_mask=False)
    assert fitted_data.shape == (7, 9)
    inverse_df = vectorizer.inverse_transform(fitted_data)
    print(inverse_df.head())
    assert len(inverse_df.columns.difference(df.columns)) == 0
    assert all([a == b for a, b in zip(inverse_df.columns, vectorizer.feature_names)])
    assert len(inverse_df["A"].compare(df["A"])) == 0
    assert all([inverse_df[ocol].dtype == df[ocol].dtype for ocol in df.columns])
    assert vectorizer.output_indices == {
        "categorical": {
            "A": {"indices": slice(4, 7), "n_categories": 3},
            "C": {"indices": slice(2, 4), "n_categories": 2},
        },
        "continuous": {"B": {"indices": slice(0, 1)}, "D": {"indices": slice(1, 2)}},
    }
    vectorizer2 = MultiTransformer(ordinal_categories=True, standard_scaling=False)
    fitted_data2, mask = vectorizer2.fit_transform(df, return_mask=True)
    print(fitted_data2)
    assert False
    inverted_data = vectorizer2.inverse_transform(fitted_data2)
    assert mask.dtype == bool
    assert np.all(np.isnan(fitted_data2[mask]))
    assert fitted_data2.shape == (7, 4)
    assert vectorizer2.output_indices == {
        "categorical": {
            "A": {"indices": slice(3, 4), "n_categories": 3},
            "C": {"indices": slice(2, 3), "n_categories": 2},
        },
        "continuous": {"B": {"indices": slice(0, 1)}, "D": {"indices": slice(1, 2)}},
    }

    assert all(
        [a == b for a, b in zip(vectorizer2.feature_names, ["B", "D", "C", "A"])]
    )


def test_feature_clustering():
    test_nlp = lambda x: np.random.choice([1.0, 0.0], size=(10,))
    test_X = np.random.choice([1.0, 0.0], size=(14, 20))
    test_vocab = {j: i for i, j in enumerate("abcdefghik")}
    clus = FeatureClusterer(3)
    clus.fit(test_vocab, test_nlp)
    new_X = clus.transform(test_X)
    assert new_X.shape == (14, 3)


def test_feature_cluster_load():
    clus = FeatureClusterer(3)
    clus.load_from_pickle("./models/path_model_030123_v2")
    vec_array = np.random.randn(2, 1854)
    agglo_vecs = clus.transform(vec_array)
    assert agglo_vecs.shape == (2, 100)


def test_vectorizer():
    vec = EntityCountVectorizer()
    vec.load_from_pickle("./models/path_model_030123_v2")
    nlp = spacy.load(
        "./models/path_model_030123_v2/en_full_neuro_model-1.8/en_full_neuro_model/en_full_neuro_model-1.8"
    )
    doc = nlp(report_sample[0])
    vecs = vec.transform([doc])
    assert vecs.shape[1] == 1854
