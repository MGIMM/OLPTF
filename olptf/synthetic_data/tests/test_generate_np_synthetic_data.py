import numpy as np
import pandas as pd

from ..generate_np_synthetic_data import (
    generate_np_risk_model,
    generate_predictor,
    generate_dict_of_np,
)


def test_generate_np_risk_model():
    sigma = 0.02
    F, D = generate_np_risk_model(n_factors=3, n_features=10, sigma=sigma)
    cov = F @ F.T + D
    # assert if risk model is of desired vol
    assert np.isclose(np.diag(cov)[0], np.power(sigma, 2))
    # 1D case


def test_predictor():
    T = 10
    N = 3
    pred = generate_predictor(T=T, N=N, time_scale=5)
    assert pred.shape == (T, N)


def test_generate_dict_of_np():
    T = 10
    N = 3
    K = 2
    data = generate_dict_of_np(T=T, N=N, K=K, time_scale_predictor=20)
    assert data["predictor"].shape == (T, N)
