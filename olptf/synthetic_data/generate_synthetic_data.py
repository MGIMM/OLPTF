import numpy as np
import pandas as pd
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random


def generate_jnp_risk_model(key, n_features, n_factors, sigma=None):
    """Generate factor risk model according to:
    cov = FF' + D

    Args:
        n_features (int): number of features.
        n_factors (int): number of factors.
        sigma (float, optional): typical volatility. Defaults to None.

    Returns
    -------
    tuple (F,D)

    """
    N = n_features
    K = n_factors

    if K == 0:
        F = jnp.zeros((N, 0))
    else:
        key, subkey = random.split(key)
        F = random.normal(key=subkey, shape=(N, K))
        # orthogonalize and normalize
        F = jnp.linalg.svd(F)[0][:, :K]
        if K >= 1:
            F = F.at[:, 0].set(F[:, 0] * jnp.sqrt(10))
        if K >= 2:
            F = F.at[:, 1].set(F[:, 1] * jnp.sqrt(5))
    D = jnp.ones(N)

    if sigma:
        cov = F @ F.T + jnp.diag(D)
        vol = jnp.sqrt(jnp.diag(cov))
        D = D / vol**2 * sigma**2
        F = F / vol[:, None] * sigma
    return F, D


def generate_predictor(
    key,
    T,
    N,
    time_scale,
):
    key, subkey = random.split(key)
    predictor = random.normal(key=subkey, shape=(T, N))
    rho = 1.0 - 1.0 / time_scale
    factor_ = jnp.sqrt(1.0 - rho**2) / (1.0 - rho)
    predictor = predictor.at[1:, :].set(predictor[1:, :] * factor_)
    # TODO: use jnp backend
    predictor = (
        pd.DataFrame(predictor).ewm(alpha=1.0 / time_scale, adjust=False).mean().values
    )
    return jnp.asarray(predictor)


def generate_dict_of_jnp(
    key,
    T=500,
    N=100,
    K=2,
    sigma=0.02,
    beta=None,
    sharpe=2.0,
    turnover=None,
    linear_cost=None,
    predictor=None,
    time_scale_predictor=None,
):
    """Generate data of return model:
    r = beta p + eps

    Args:
        key (jax random key): jax random key.
        T (int, optional): number of days. Defaults to 500.
        N (int, optional): number of instruments. Defaults to 100.
        K (int, optional): number of factors in risk model. Defaults to 2.
        sigma (float, optional): volatility. Defaults to 0.02.
        beta (float, optional): pred scale. Defaults to None.
        sharpe (float, optional): shapre ratio of synthetic predictor. Defaults to 2.
        turnover (float, optional): turnover. Defaults to None.
        linear_cost (float, optional): linear cost. Defaults to None.
        predictor (ndarray, optional): synthetic predictor, when not provided, predictor is generated according to time_scale_predictor. Defaults to None.
        time_scale_predictor (float, optional): time scale of synthetic ar(1) predictor, should be None when predictor is provided. Defaults to None.

    Return:
        dict: synthetic data.

    """
    assert K <= N, "invalid factor numbers!"
    key, subkey = random.split(key)
    F, D = generate_jnp_risk_model(key=subkey, n_factors=K, n_features=N, sigma=sigma)
    cov = F @ F.T + jnp.diag(D)
    # use np for linalg for float64 precision
    key, subkey = random.split(key)
    noise_rets = random.normal(key=subkey, shape=(T, N)).dot(jnp.linalg.cholesky(cov).T)

    # generate cost data
    if turnover is None:
        key, subkey = random.split(key)
        turnover = jnp.power(
            10, random.uniform(key=subkey, minval=6.0, maxval=9.0, shape=(N,))
        )

    vol = jnp.sqrt(cov.diagonal())
    gamma = 3.0
    quad_cost = gamma * vol / turnover

    if predictor is None:
        assert (
            time_scale_predictor is not None
        ), "please provide time_scale_predictor or predictor!"
        key, subkey = random.split(key)
        predictor = generate_predictor(
            key=subkey, T=T, N=N, time_scale=time_scale_predictor
        )
    else:
        assert predictor.shape == (T, N), f"predictor should be of shape ({T},{N})."

    # compute beta
    if beta is None:
        # # norm_predictor_daily == False
        cov_pred = predictor.T @ predictor / len(predictor)
        norm_pred = np.sqrt(np.trace(np.linalg.inv(cov) @ cov_pred))
        #norm_pred = jnp.sqrt((predictor * predictor.dot(jnp.linalg.inv(cov))).sum(1)).reshape(-1,1)
        
        assert sharpe is not None, "please provide sharpe or beta."
        daily_sharpe = sharpe / np.sqrt(252.0)
        beta = daily_sharpe / norm_pred
    else:
        assert sharpe is None, "sharpe and beta cannot be both specified."
    if jnp.isscalar(beta):
        beta = jnp.tile(beta, N)

    rets = beta * predictor + noise_rets

    aum = 1e9

    dates = pd.date_range("19000101", periods=T, freq="B")

    data = {
        "dates": dates,
        "vol": vol,
        "covariance_factors": F,
        "residual_variance": D,
        "predictor": predictor,
        "quad_cost": quad_cost,
        "turnover": turnover,
        "rets": rets,
        "beta": beta,
        "aum": aum,
    }

    if linear_cost is not None:
        data.update({"linear_cost": linear_cost})
    return data
