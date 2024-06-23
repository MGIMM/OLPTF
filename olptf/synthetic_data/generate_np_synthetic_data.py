import numpy as np
import pandas as pd
import numpy.random as random


def generate_np_risk_model(n_features, n_factors, sigma=None):
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
        F = np.zeros((N, 0))
    else:
        F = random.normal(size=(N, K))
        # orthogonalize and normalize
        F = np.linalg.svd(F)[0][:, :K]
        if K >= 1:
            F[:, 0] *= np.sqrt(10)
        if K >= 2:
            F[:, 1] *= np.sqrt(5)
    D = np.ones(N)

    if sigma:
        cov = F @ F.T + np.diag(D)
        vol = np.sqrt(np.diag(cov))
        D = D / vol**2 * sigma**2
        F = F / vol[:, None] * sigma
    return F, D


def generate_predictor(
    T,
    N,
    time_scale,
):
    predictor = random.normal(size=(T, N))
    rho = 1.0 - 1.0 / time_scale
    factor_ = np.sqrt(1.0 - rho**2) / (1.0 - rho)
    predictor[1:, :] *= factor_
    # TODO: use np backend
    predictor = (
        pd.DataFrame(predictor).ewm(alpha=1.0 / time_scale, adjust=False).mean().values
    )
    return np.asarray(predictor)


def generate_dict_of_np(
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
    F, D = generate_np_risk_model(n_factors=K, n_features=N, sigma=sigma)
    cov = F @ F.T + np.diag(D)
    # use np for linalg for float64 precision
    noise_rets = random.normal(size=(T, N)).dot(np.linalg.cholesky(cov).T)

    # generate cost data
    if turnover is None:
        turnover = np.power(10, random.uniform(low=6.0, high=9.0, size=(N,)))

    vol = np.sqrt(cov.diagonal())

    if predictor is None:
        assert (
            time_scale_predictor is not None
        ), "please provide time_scale_predictor or predictor!"
        predictor = generate_predictor(T=T, N=N, time_scale=time_scale_predictor)
    else:
        assert predictor.shape == (T, N), f"predictor should be of shape ({T},{N})."

    # compute beta
    if beta is None:
        # # norm_predictor_daily == False
        cov_pred = predictor.T @ predictor / len(predictor)
        norm_pred = np.sqrt(np.trace(np.linalg.inv(cov) @ cov_pred))
        # norm_pred = np.sqrt((predictor * predictor.dot(np.linalg.inv(cov))).sum(1)).reshape(-1,1)

        assert sharpe is not None, "please provide sharpe or beta."
        daily_sharpe = sharpe / np.sqrt(252.0)
        beta = daily_sharpe / norm_pred
    else:
        assert sharpe is None, "sharpe and beta cannot be both specified."
    if np.isscalar(beta):
        beta = np.tile(beta, N)

    rets = beta * predictor + noise_rets

    aum = 1e9
    eqt_codes = np.array(["eqt" + str(i) for i in range(N)])

    dates = pd.date_range("19000101", periods=T, freq="B")
    mask = random.uniform(size=(T, N))
    mask = pd.DataFrame(mask, index=dates, columns=eqt_codes)
    mask = mask.ewm(100).mean() > 0.5

    # put mask on every thing
    predictor = pd.DataFrame(predictor, index=dates, columns=eqt_codes)
    predictor[~mask] = np.nan

    eigen_codes = np.array(["eigen" + str(i) for i in range(K)])
    # risk model is unmasked
    F = pd.DataFrame(F, index=eqt_codes, columns=eigen_codes)
    D = pd.Series(D, index=eqt_codes)
    vol = pd.Series(vol, index=eqt_codes)

    data = {
        "dates": dates,
        "vol": vol,
        "covariance_factors": F,
        "residual_variance": D,
        "predictor": predictor,
        "turnover": turnover,
        "rets": rets,
        "beta": beta,
        "aum": aum,
        "mask": mask,
    }

    if linear_cost is not None:
        data.update({"linear_cost": linear_cost})
    return data
