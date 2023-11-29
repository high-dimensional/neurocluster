"""Vectorizer classes for creating vectorizied Doc objects

These classes convert an iterable of spacy Doc objects to
a series numerical feature vectors.
"""

import pickle
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.preprocessing import (KBinsDiscretizer, MultiLabelBinarizer,
                                   OneHotEncoder, OrdinalEncoder,
                                   PowerTransformer, StandardScaler)


class BaseVectorizer(BaseEstimator, TransformerMixin):
    """Base vectorizer class

    Parent class for implimentations of spacy doc vectorization
    classes. fit/transform methods are not implemented and this
    class exists only to be inherited by other classes.

    Attributes
    ----------
    pathology_types : list of str
        the pathology types included in the featurization
    section_types : list of str
        the section types included in the featurization
    """

    def __init__(self, pathology_only=False, body_only=True):
        """
        Parameters
        ----------
        body_only : bool, optional
            whether to include the whole report in featurization or just entities in the body
        pathology_only : bool, optional
            whether to use both PATHOLOGY and DESCRIPTOR types in featurization
        """
        super().__init__()
        self.section_types = (
            ["body"]
            if body_only
            else ["body", "header", "indications", "metareport", "tail"]
        )
        self.pathology_types = (
            ["PATHOLOGY"] if pathology_only else ["PATHOLOGY", "DESCRIPTOR"]
        )

    def _are_docs_valid(self, docs):
        """checks if all the docs have the required attributes"""
        return (
            True if all([len(e._.cui) > 1 for doc in docs for e in doc.ents]) else False
        )

    def _get_entities(self, docs):
        """gets all pathology-location entity pairs"""
        entity_pairs = []
        path_filter = lambda e: (e.label_ in self.pathology_types) and (
            not e._.is_negated
        )
        for doc in docs:
            doc_pair_list = []
            for ent in filter(path_filter, doc.ents):
                doc_pair_list.append((ent, None))
                for loc in ent._.relation:
                    doc_pair_list.append((ent, loc))
            entity_pairs.append(doc_pair_list)
        return entity_pairs

    def fit(self, docs, y=None):
        return self

    def transform(self, docs):
        raise NotImplementedError


class EntityCountVectorizer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        class_names=["PATHOLOGY"],
        ngram_range=(1, 1),
        stop_words=[
            "to",
            "for",
            "or",
            "of",
            "on",
            "at",
            "of the",
            "and",
            "the",
            "with",
            "in",
            "by",
        ],
        max_features=None,
        binary=False,
        suppress_normal=False,
        suppress_laterality=False,
        min_df=1,
    ):
        super().__init__()
        self.class_names = class_names
        self.normals_stop_words = [
            "normal",
            "limits",
            "appearance",
            "appearances",
            "intracranial",
            "parenchymal",
        ]
        self.binary = (binary,)
        self.laterality_stop_words = ["left", "right"]
        new_stop_words = self.set_stop_words(
            stop_words, suppress_normal, suppress_laterality
        )
        self.internal_vectorizer = CountVectorizer(
            stop_words=new_stop_words,
            ngram_range=ngram_range,
            max_features=max_features,
            binary=True,
            min_df=min_df,
        )

    def set_stop_words(self, stop_words, suppress_normal, suppress_laterality):
        total_stop_words = stop_words
        if suppress_normal:
            total_stop_words.extend(self.normals_stop_words)
        if suppress_laterality:
            total_stop_words.extend(self.laterality_stop_words)
        return total_stop_words

    def _get_doc_entities(self, docs):
        doc_entities = [
            [e.text for e in doc.ents if e.label_ in self.class_names if not e._.negex]
            for doc in docs
        ]
        return doc_entities

    def fit(self, docs, y=None):
        doc_entities = self._get_doc_entities(docs)
        all_entities = [w for doc in doc_entities for w in doc]
        self.internal_vectorizer.fit(all_entities)
        return self

    def transform(self, docs):
        doc_entities = self._get_doc_entities(docs)
        docs_array = np.stack(
            [
                self.internal_vectorizer.transform(doc).sum(axis=0)
                for doc in doc_entities
            ],
            axis=0,
        )
        if self.binary:
            return (docs_array >= 1.0).astype(float)
        else:
            return docs_array

    def fit_transform(self, docs):
        doc_entities = self._get_doc_entities(docs)
        all_entities = [w for doc in doc_entities for w in doc]
        self.internal_vectorizer.fit(all_entities)
        docs_array = np.stack(
            [
                self.internal_vectorizer.transform(doc).sum(axis=0)
                for doc in doc_entities
            ],
            axis=0,
        )
        if self.binary:
            return (docs_array >= 1.0).astype(float)
        else:
            return docs_array

    def load_from_pickle(self, model_path):
        """load the object __dict__ from a pickle file"""
        with open(Path(model_path) / "report_vectorizer_dict.pkl", "rb") as f:
            self.__dict__.update(pickle.load(f))
        return self


class FeatureClusterer(BaseEstimator, TransformerMixin):
    """Feature agglomeration for token n-gram features"""

    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        self.clusterer = KMeans(n_clusters=output_size)
        self.grouped_vocab = (
            None  # grouped_vocab[vocab_group_name] = list of input_vocab for this group
        )
        self.index_map = None  # index_map[input_vocab_index] = vocab_group_index
        self.reversed_vocab = None

    def fit(self, vocab, word2vec_model):
        self.reversed_vocab = {j: i for i, j in vocab.items()}
        feature_vectors = self.get_vectors(vocab, word2vec_model)
        self.index_map = self.clusterer.fit_predict(feature_vectors)
        self.grouped_vocab = defaultdict(list)
        for i, j in enumerate(self.index_map):
            self.grouped_vocab[j].append(self.reversed_vocab[i])
        return self

    def get_vectors(self, vocab, word2vec):
        return np.array([word2vec(self.reversed_vocab[i]) for i in range(len(vocab))])

    def transform(self, X):
        new_X = np.zeros((X.shape[0], self.output_size))
        for i, v in enumerate(self.index_map):
            rows_with_feature = X[:, i] > 0
            if rows_with_feature.any():
                new_X[rows_with_feature, v] = 1.0
        return new_X

    def load_from_pickle(self, model_path):
        """load the object __dict__ from a pickle file"""
        with open(Path(model_path) / "feature_clusterer_dict.pkl", "rb") as f:
            self.__dict__.update(pickle.load(f))
        return self


def merge_feature_vectors(transformed_data, agglomerated_data):
    agglomerated_data = np.concatenate((transformed_data, agglomerated_data), axis=1)
    return agglomerated_data


## EntityTfidfVectorizer removed 15-11-22

# ClusteringVectorizer removed 15/11/22

# OntologyVectorizer removed 30/06/2022

# CustomDfVectorizer removed 21/02/2022

# MultiVectorizer removed 21/02/2022


class MultiTransformer:
    """Custom df column vectorizer

    Converts a pandas dataframe made up
    of categorical and continuous columns
    into a numpy array.

    Categorical columns are one-hot encoded,
    continuous columns are scaled.

    We assume missing data is represented by Nan values.
    Nan entries are represented by zeros in the output.

    Ensure the categorical columns are recorded as "category" type
    Ensure continuous columns are recorded as "float" type
    """

    def __init__(self, ordinal_categories=False, standard_scaling=False):
        self.ordinal_categories = ordinal_categories
        self.standard_scaling = standard_scaling
        self.missing_checker = MissingIndicator(
            features="all", sparse=False, missing_values=np.nan
        )
        self.transformer = self._make_column_transformer()
        self.lognormal_columns = []

    def _make_column_transformer(self):
        scaler = (
            StandardScaler()
            if self.standard_scaling
            else PowerTransformer(standardize=True)
        )
        encoder = (
            OrdinalEncoder()
            if self.ordinal_categories
            else OneHotEncoder(sparse=False, handle_unknown="ignore")
        )
        transformer = ColumnTransformer(
            transformers=[
                (
                    "continuous",
                    scaler,
                    make_column_selector(dtype_include="float64"),
                ),
                (
                    "categorical",
                    encoder,
                    make_column_selector(dtype_include="category"),
                ),
            ]
        )

        return transformer

    def fit(self, df, lognormal_columns=[]):
        df_copy = df.copy()
        if lognormal_columns:
            if not self.standard_scaling:
                raise Exception(
                    "Internal transformer uses a Power transformer that can deal with log normal data, specifying log normal columns is only necessary when using the standard scaler"
                )
            cols_in_df = df_copy.columns.intersection(pd.Index(lognormal_columns))
            if len(cols_in_df):
                self.lognormal_columns = cols_in_df
                df_copy.loc[:, cols_in_df] = np.log(df_copy.loc[:, cols_in_df])
            else:
                raise Exception("Lognormal column names not present in dataframe")
        self.transformer.fit(df_copy)
        return self

    def transform(self, df, return_mask=False):
        df_copy = df.copy()
        if len(self.lognormal_columns):
            df_copy.loc[:, self.lognormal_columns] = np.log(
                df_copy.loc[:, self.lognormal_columns]
            )
        transformed_data = self.transformer.transform(df_copy)
        if return_mask:
            missing_mask = self.get_mask(transformed_data)
            return transformed_data, missing_mask
        else:
            return transformed_data

    def expand_mask(self, mask):
        raise NotImplementedError(
            "This vectorizer does not produce missing masks for one-hot encoded categories yet"
        )

    def get_mask(self, df):
        missing_mask = self.missing_checker.fit_transform(df)
        if not self.ordinal_categories:
            missing_mask = self.expand_mask(missing_mask)
        return missing_mask

    def fit_transform(self, df, return_mask=False, lognormal_columns=[]):
        self.fit(df, lognormal_columns)
        output = self.transform(df, return_mask)
        return output

    def inverse_transform(self, X):
        out_idx = self.transformer.output_indices_
        categorical_X, continuous_X = (
            X[:, out_idx["categorical"]],
            X[:, out_idx["continuous"]],
        )
        feature_names = self.feature_names
        cat_transformer = self.transformer.transformers_[1][1]
        cont_transformer = self.transformer.transformers_[0][1]
        inverted_cat_data = cat_transformer.inverse_transform(categorical_X)
        inverted_cont_data = cont_transformer.inverse_transform(continuous_X)
        cont_df = pd.DataFrame(
            data=inverted_cont_data,
            columns=feature_names[out_idx["continuous"]],
            dtype=float,
        )
        if len(self.lognormal_columns):
            cont_df.loc[:, self.lognormal_columns] = np.exp(
                cont_df.loc[:, self.lognormal_columns]
            )
        cat_df = pd.DataFrame(
            data=inverted_cat_data, columns=feature_names[out_idx["categorical"]]
        )
        cat_types = {col: "category" for col in cat_df.columns}
        cat_df = cat_df.astype(cat_types)
        # inverted_data = np.concatenate([inverted_cont_data, inverted_cat_data], axis=1)
        return pd.concat(
            [cont_df, cat_df], axis=1
        )  # ,ignore_index=True)#pd.DataFrame(data=inverted_data, columns=feature_names, )

    @property
    def output_indices(self):
        cat_transformer = self.transformer.transformers_[1]
        cont_transformer = self.transformer.transformers_[0]
        categories = cat_transformer[1].categories_
        no_na_categories = [np.array([i for i in cat if i == i]) for cat in categories]
        cat_name, cat_feats, cat_classes = (
            cat_transformer[0],
            cat_transformer[1].feature_names_in_,
            no_na_categories,
        )
        cont_name, cont_feats = (
            cont_transformer[0],
            cont_transformer[1].feature_names_in_,
        )
        cont_slices = {
            name: {"indices": slice(i, i + 1)} for i, name in enumerate(cont_feats)
        }
        n_conts = len(cont_feats)
        if self.ordinal_categories:
            cat_slices = {
                name: {
                    "indices": slice(n_conts + i, n_conts + i + 1),
                    "n_categories": len(cats),
                }
                for i, (name, cats) in enumerate(zip(cat_feats, cat_classes))
            }
        else:
            cat_slices = {}
            running_idx = n_conts
            for name, cats in zip(cat_feats, cat_classes):
                cat_slices[name] = {
                    "indices": slice(running_idx, running_idx + len(cats)),
                    "n_categories": len(cats),
                }
                running_idx += len(cats)
        output_indices = {cat_name: cat_slices, cont_name: cont_slices}
        return output_indices

    @property
    def feature_names(self):
        cat_transformer = self.transformer.transformers_[1]
        cont_transformer = self.transformer.transformers_[0]
        cat_names = cat_transformer[1].feature_names_in_
        cont_names = cont_transformer[1].feature_names_in_
        columns = np.concatenate([cont_names, cat_names])
        return columns
