import pandas as pd
from pandas.tseries.offsets import BDay


def synthetic_data_stream(data, config=None):
    dates = data["dates"]
    for i, date in enumerate(dates):
        pred = data["predictor"][i]
        F = data["covariance_factors"]
        D = data["residual_variance"]
        # static quad cost
        gamma = data["quad_cost"]
        vol = data["vol"]
        turnover = data["turnover"]
        rets = data["rets"][i]

        stream = {
            "date": date,
            "predictor": pred,
            "vol": vol,
            "covariance_factors": F,
            "residual_variance": D,
            "quad_cost": gamma,
            "turnover": turnover,
            "rets": rets,
        }
        if config is not None:
            assert isinstance(config, dict), "config shoud be a dict!"
            for key, value in config.items():
                stream.update({key: value})
        yield stream
