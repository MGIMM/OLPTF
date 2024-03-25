import numpy as np
import pandas as pd
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

from olptf.synthetic_data.generate_synthetic_data import generate_jnp_risk_model, generate_predictor, generate_dict_of_jnp

key = random.key(0)


def test_risk_model():
    key = random.key(0)
    sigma = 0.02
    F, D = generate_jnp_risk_model(key=key, n_factors=3, n_features=10, sigma=sigma)
    cov = F @ F.T + D
    # assert if risk model is of desired vol
    assert jnp.isclose(jnp.diag(cov)[0],jnp.pow(sigma, 2))
    # 1D case
    F, D = generate_jnp_risk_model(key=key, n_factors=1, n_features=1, sigma=sigma)

def test_predictor():
    T = 10
    N = 3
    pred = generate_predictor(key=key, T=T, N = N, time_scale=5)
    assert pred.shape == (T, N)

def test_generate_dict_of_jnp():
    T = 10
    N = 3 
    K = 2
    data = generate_dict_of_jnp(key=key, T=T, N=N, K=K, time_scale_predictor=20) 
    assert data["predictor"].shape == (T,N)

