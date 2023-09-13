import numpy as np
import pytest

from index.index import Index

np.random.seed(27)


def test_initialisation():
    dimension = 10
    embeddings = np.random.rand(10, 11)

    with pytest.raises(ValueError) as exc_info:
        index = Index(embeddings=embeddings, dimension=dimension, normalise=False)

    assert (
        str(exc_info.value)
        == f"Expected embeddings of dimension {dimension} but got {embeddings.shape[1]}"
    )

    embeddings_correct = np.random.rand(10, 10)
    index = Index(embeddings=embeddings_correct, dimension=dimension, normalise=True)

    assert len(index) == len(embeddings_correct)

    embeddings_unnormalised = np.array(
        [
            [
                0.15291948,
                0.29996861,
                0.45176148,
                0.19056872,
                0.65453686,
                0.61923943,
                0.64575399,
                0.15695894,
                0.84477893,
                0.91151869,
            ],
            [
                0.88766168,
                0.50003678,
                0.1762516,
                0.67199951,
                0.58559397,
                0.08123755,
                0.06319375,
                0.0760311,
                0.3199756,
                0.55600202,
            ],
            [
                0.37418972,
                0.74486978,
                0.14142433,
                0.55343016,
                0.58774383,
                0.04822796,
                0.22015045,
                0.28131,
                0.22465193,
                0.05507964,
            ],
            [
                0.90079836,
                0.81287259,
                0.3836313,
                0.59169538,
                0.90809283,
                0.1955902,
                0.88167246,
                0.13599261,
                0.11642354,
                0.18371848,
            ],
            [
                0.79795983,
                0.44441992,
                0.50138439,
                0.02061258,
                0.19624977,
                0.04626845,
                0.30993094,
                0.53831361,
                0.69843272,
                0.1865773,
            ],
            [
                0.17878129,
                0.17595442,
                0.04217433,
                0.62908872,
                0.39394527,
                0.22698598,
                0.82921537,
                0.77054606,
                0.32488408,
                0.65313149,
            ],
            [
                0.47742867,
                0.32482545,
                0.6003544,
                0.49423264,
                0.74311799,
                0.30445699,
                0.8235536,
                0.70106103,
                0.19251904,
                0.51238843,
            ],
            [
                0.28463922,
                0.43990004,
                0.09757145,
                0.06348822,
                0.47554289,
                0.7172874,
                0.20215388,
                0.49335816,
                0.30058124,
                0.75789675,
            ],
            [
                0.56020627,
                0.5024675,
                0.39965748,
                0.14681318,
                0.2930046,
                0.67610837,
                0.37827243,
                0.84564952,
                0.10335147,
                0.02385907,
            ],
            [
                0.01677684,
                0.07008122,
                0.61222488,
                0.02328426,
                0.12180433,
                0.68747517,
                0.54473236,
                0.63845266,
                0.49192671,
                0.55479406,
            ],
        ]
    )

    embeddings_normalised = np.array(
        [
            [
                0.03089635,
                0.06060664,
                0.09127537,
                0.03850313,
                0.13224477,
                0.12511316,
                0.13047025,
                0.0317125,
                0.1706819,
                0.18416622,
            ],
            [
                0.17934607,
                0.10102907,
                0.03561045,
                0.13577298,
                0.11831532,
                0.0164135,
                0.01276787,
                0.01536157,
                0.06464892,
                0.11233647,
            ],
            [
                0.07560252,
                0.15049593,
                0.02857383,
                0.11181684,
                0.11874969,
                0.00974414,
                0.04447992,
                0.05683679,
                0.04538941,
                0.01112847,
            ],
            [
                0.18200025,
                0.16423544,
                0.07751012,
                0.11954807,
                0.18347405,
                0.03951768,
                0.17813599,
                0.02747639,
                0.02352259,
                0.03711908,
            ],
            [
                0.16122242,
                0.08979205,
                0.10130134,
                0.00416463,
                0.03965095,
                0.00934823,
                0.06261946,
                0.10876264,
                0.14111363,
                0.03769669,
            ],
            [
                0.03612156,
                0.03555041,
                0.00852104,
                0.12710314,
                0.07959399,
                0.04586099,
                0.16753739,
                0.15568365,
                0.06564064,
                0.13196082,
            ],
            [
                0.09646125,
                0.0656288,
                0.12129757,
                0.09985638,
                0.15014199,
                0.06151349,
                0.16639346,
                0.14164466,
                0.03889718,
                0.10352464,
            ],
            [
                0.05750944,
                0.08887884,
                0.01971365,
                0.01282737,
                0.09608024,
                0.14492309,
                0.04084383,
                0.0996797,
                0.06073042,
                0.15312794,
            ],
            [
                0.11318591,
                0.10152018,
                0.0807481,
                0.02966262,
                0.05919961,
                0.13660315,
                0.0764274,
                0.1708578,
                0.02088147,
                0.00482056,
            ],
            [
                0.00338965,
                0.01415944,
                0.12369592,
                0.00470443,
                0.02460975,
                0.13889973,
                0.11005951,
                0.12899507,
                0.09939048,
                0.11209241,
            ],
        ]
    )

    index = Index(
        embeddings_unnormalised, len(embeddings_unnormalised[0]), normalise=True
    )

    assert np.allclose(index.embeddings, embeddings_normalised)


def test_add_valid_single_vector():
    dimension = 10
    embeddings = np.random.rand(10, 10)
    index = Index(embeddings, dimension)

    new_vector_1 = np.random.rand(1, 10)
    index.add_vector(new_vector_1)

    assert len(index) == 11
    assert len(index.embeddings) == 11


def test_add_valid_multiple_vectors():
    dimension = 10
    embeddings = np.random.rand(10, 10)
    index = Index(embeddings, dimension)

    new_vector_2 = np.random.rand(5, 10)
    index.add_vector(new_vector_2)

    assert len(index) == 15
    assert len(index.embeddings) == 15


def test_add_valid_vector_with_different_dimension():
    dimension = 10
    embeddings = np.random.rand(10, 10)
    index = Index(embeddings, dimension)

    new_vector_3 = np.random.rand(10)
    index.add_vector(new_vector_3)

    assert len(index) == 11
    assert len(index.embeddings) == 11


def test_add_invalid_vector_dimension_mismatch():
    dimension = 10
    embeddings = np.random.rand(10, 10)
    index = Index(embeddings, dimension)

    new_vector_4 = np.random.rand(1, 11)

    with pytest.raises(ValueError) as exc_info:
        index.add_vector(new_vector_4)

    assert (
        str(exc_info.value)
        == f"Expected vector of dimension {dimension} but got {new_vector_4.shape[1]}"
    )


def test_add_invalid_vector_dimension_too_large():
    dimension = 10
    embeddings = np.random.rand(10, 10)
    index = Index(embeddings, dimension)

    new_vector_5 = np.random.rand(20)

    with pytest.raises(ValueError) as exc_info:
        index.add_vector(new_vector_5)

    assert (
        str(exc_info.value)
        == f"Expected vector of dimension {dimension} but got {len(new_vector_5)}"
    )


def test_query_ideal_case():
    embeddings = np.random.rand(11, 10)
    dimension = 10

    query_1 = embeddings[0]
    index = Index(embeddings, dimension)

    k = 3
    res1, ans1 = index.get_similarity(query_1, k)
    assert len(res1) == k
    assert len(res1.shape) == 1
    assert ans1.shape[0] == k
    assert ans1.shape[1] == dimension


def test_query_multi_vector():
    embeddings = np.random.rand(11, 10)
    dimension = 10

    query_2 = embeddings[:5]
    index = Index(embeddings, dimension)

    k = 3
    with pytest.raises(NotImplementedError) as exc_info:
        index.get_similarity(query_2, k)

    assert str(exc_info.value) == "Multi-vector query not supported yet."


def test_query_dimension_mismatch():
    embeddings = np.random.rand(11, 10)
    dimension = 10

    query_3 = np.random.rand(1, 10)
    index = Index(embeddings, dimension)

    k = 1
    res3, ans3 = index.get_similarity(query_3, k)
    assert len(res3) == k
    assert len(res3.shape) == 1
    assert ans3.shape[0] == k
    assert ans3.shape[1] == dimension


def test_query_incompatible_vector():
    embeddings = np.random.rand(11, 10)
    dimension = 10

    query_4 = np.random.rand(11)
    index = Index(embeddings, dimension)

    k = 1
    with pytest.raises(ValueError) as exc_info:
        index.get_similarity(query_4, k)

    assert (
        str(exc_info.value)
        == f"Expected vector of dimension {dimension} but got {query_4.shape[0]}"
    )


def test_query_k_zero():
    embeddings = np.random.rand(11, 10)
    dimension = 10

    query_1 = embeddings[0]
    index = Index(embeddings, dimension)

    k = 0
    res1, ans1 = index.get_similarity(query_1, k)
    assert len(res1) == k
    assert len(res1.shape) == 1
    assert ans1.shape[0] == k
    assert ans1.shape[1] == dimension


def test_query_k_negative():
    embeddings = np.random.rand(11, 10)
    dimension = 10

    query_1 = embeddings[0]
    index = Index(embeddings, dimension)

    k = -20
    with pytest.raises(ValueError) as exc_info:
        index.get_similarity(query_1, k)

    assert str(exc_info.value) == f"Expected k>0 got k={k}"


def test_query_k_greater_than_index_length():
    embeddings = np.random.rand(11, 10)
    dimension = 10

    query_1 = embeddings[0]
    index = Index(embeddings, dimension)

    k = 12
    res1, ans1 = index.get_similarity(query_1, k)
    assert len(res1) == len(index)
    assert len(res1.shape) == 1
    assert ans1.shape[0] == len(index)
    assert ans1.shape[1] == dimension
