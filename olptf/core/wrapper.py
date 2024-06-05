from dataclasses import dataclass
from copy import deepcopy
from typing import Any, List
from .abstract import AbstractAgent, AbstractEnv
from .concrete import Agent
import inspect


# function-agent wrapper
@dataclass
class WrapperAgent(Agent):
    f: Any
    params: dict = None
    defaults: dict = None

    def __post_init__(self):
        super().__post_init__()
        self._effective_params = dict()

    def update_keys(self, input_keys):
        self.input_keys = input_keys

    def act(self):
        self.effective_params = self.params
        effective_signature = {**self.obs, **self.effective_params}
        return self.f(**effective_signature)

    @property
    def effective_params(self):
        return self._effective_params

    @effective_params.setter
    def effective_params(self, params):
        _params = deepcopy(self.defaults) or dict()
        _params.update(params)
        self._effective_params = _params


def make_agent(f, params, input_keys):
    signature = inspect.signature(f)
    default_params = {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty and k not in input_keys
    }
    agent = WrapperAgent(
        f=f,
        params=params,
        defaults=default_params,
    )
    agent.update_keys(input_keys=input_keys)
    return agent

