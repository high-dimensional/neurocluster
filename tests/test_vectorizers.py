import numpy as np
import pandas as pd
import pytest
import spacy
from negspacy.negation import Negex

from neurocluster.vectorizers import (EntityCountVectorizer, FeatureClusterer,
                                      MultiTransformer)

report_sample = [
    "MRI Head & Neck Clinical Indications: Post-op tumour excision. ?residual tumour. pre and post contrast MRI head please on Thursday. Findings: Comparison is made with the previous MR studies dated 16/02 17 and 09/01/2018. There has been interval resection of the previously shown enhancing tumour centred on the superior aspects of the posterior ethmoid and sphenoid sinuses, with involvement of the anterior cranial fossa. Heterogeneous signal is demonstrated within the surgical bed, along with areas of faint T1 shortening and curvilinear enhancement, which are likely postsurgical in nature at this stage. Allowing for these changes, no residual or recurrent tumour is convincingly demonstrated, although this will be clarified on subsequent follow-up imaging. There is sign of multiple sclerosis in the frontal lobe. Note is made of mild thickening and enhancement of the anteroinferior aspect of the falx cerebri. The remaining intracranial appearances are stable. The previous left frontal resection cavity is again shown. Note is also again made of a few non-specific foci of T2 hyperintensity within the cerebral white matter. Postsurgical changes are noted in the paranasal sinuses. WM/ Dr Sachit Shah Consultant Neuroradiologist neurorad@uclh.nhs.uk",
    "Clinical Indications: Post-op tumour excision. ?residual tumour. pre and post contrast MRI head please on Thursday. Findings: Comparison is made with the previous MR studies dated 16/02 17 and 09/01/2018. There has been interval resection of the previously shown enhancing tumour centred on the superior aspects of the posterior ethmoid and sphenoid sinuses, with involvement of the anterior cranial fossa. Heterogeneous signal is demonstrated within the surgical bed, along with areas of faint T1 shortening and curvilinear enhancement, which are likely postsurgical in nature at this stage. Allowing for these changes, no residual or recurrent tumour is convincingly demonstrated, although this will be clarified on subsequent follow-up imaging. Note is made of mild thickening and enhancement of the anteroinferior aspect of the falx cerebri. The remaining intracranial appearances are stable. The previous left frontal resection cavity is again shown. Note is also again made of a few non-specific foci of T2 hyperintensity within the cerebral white matter. Postsurgical changes are noted in the paranasal sinuses. WM/ Dr Sachit Shah Consultant Neuroradiologist neurorad@uclh.nhs.uk",
    "There is moderate dilatation of the third and lateral ventricles, distension of the third ventricular recesses and mild enlargement of the pituitary fossa with depression of the glandular tissue. Appearances are in keeping with hydrocephalus and the mild associated frontal and peritrigonal transependymal oedema indicates an ongoing active element. No cause for hydrocephalus is demonstrated and the fourth ventricle and aqueduct are normal in appearance. Note is made of T1 shortening within the floor of the third ventricle. On the T2 gradient echo there is a bulbous focus of reduced signal in this region that is inseparable from the terminal basilar artery but likely to be lying anterosuperiorly. The combination of features is most likely to reflect a small incidental dermoid within the third ventricle with fat and calcified components. This could be confirmed by examination of the CT mentioned on the request form. If the presence of fat and calcium is not corroborated on the plain CT, a time-of-flight MR sequence or CTA would be prudent to exclude the small possibility of a vascular abnormality in this region. Dr. Matthew Adams",
    "Clinical details: L1 conus SOL. MRI brain/whole spine to rule out other lesions. Findings: There is bony asymmetry of the occipital bones and a small lipoma within the nuchal soft tissues. The intracranial appearances are otherwise normal. Unchanged appearances of the previously demonstrated enhancing lumbar mass lesion. Note is again made of mild spinal cord compression at T12-L1 secondary to degenerative disc changes. No other spinal or intracranial lesions are identified. KW",
    "Clinical Indications for MRI - White matter lesions on earlier scan has phospholipuid syndrome but ? MS Will need contrast Findings: Comparison is made with the previous scan performed 6 April 2016. Stable appearances of the diffuse and confluent bilateral white matter high T2/FLAIR signal changes. A few more conspicuous focal lesions in the periventricular and juxta-cortical white matter are again demonstrated and unchanged. There is no evidence of signal changes in the posterior fossa structures. The imaged cervical cord returns normal signal. There is no evidence of pathological enhancement. Summary: Stable appearances of the supratentorial white matter signal changes. Although some lesions appear more conspicuous in the periventricular and juxtacortical regions, there is no significant lesion burden to characterise dissemination in space at this time point. Dr Kelly Pegoretti Consultant Neuroradiologist email: neurorad@uclh.nhs.uk",
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
