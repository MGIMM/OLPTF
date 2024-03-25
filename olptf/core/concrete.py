from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, List
from .abstract import AbstractAgent, AbstractEnv
import inspect

BASE_FIELDS = [
    "attrs_",
    "act",
    "abc_impl",
    "attrs",
]


@dataclass
class Agent(AbstractAgent):
    def __post_init__(self):
        super().__post_init__()
        for attr_ in self.__dir__():
            if not attr_.endswith("_") and attr_ not in BASE_FIELDS:
                self.attrs.update({attr_: self.__getattribute__(attr_)})

    def __call__(self, obs):
        return self.act(obs) if obs is not None else None

    def act(self, obs) -> Any: ...


@dataclass
class PipelineAgent(Agent):
    agents: List

    def __post_init__(self):
        super().__post_init__()
        self.flatten_agents()

    def flatten_agents(self):
        agents_ = list()
        for agent in self.agents:
            if hasattr(agent, "agents"):
                for agent_ in agent.agents:
                    agents_.append(agent_)
                    self.attrs.update({agent_.__class__.__name__: agent_.attrs})
            else:
                agent_.append(agent)
        self.agents = agents_

    def act(self, obs):
        actions = dict()
        for agent in self.agents:
            action = agent(obs)
            if action is not None:
                obs.update(action)
                actions.update(action)
        return actions


@dataclass
class Env(AbstractEnv):
    data_stream: Any = None

    def __post_init__(self):
        super().__post_init__()
        if self.data_stream is not None:
            self.iter_ = iter(self.data_stream)
        self.state_ = {}

    def observation(self):
        if self.data_stream is not None:
            data = next(self.iter_)
            self.state.update(data)
        return self.state

    def step(self):
        try:
            obs = self.observation()
            done = False
        except StopIteration:
            obs = None
            done = True
        return obs, done

    @property
    def state(self):
        return self.state_

    def update(self, action):
        if action is not None:
            self.state.update(action)


# function-agent wrapper
@dataclass
class WrapperAgent(Agent):
    f: Any
    params: dict
    defaults: dict

    def __post_init__(self):
        self.input_keys = set()
        self.output_keys = set()
        super().__post_init__()

    def update_keys(self, input_keys, output_keys=None):
        self.input_keys = set(input_keys)
        if output_keys is not None:
            self.output_keys = set(output_keys)

    def act(self, obs):
        effective_obs = {k: obs[k] for k in self.input_keys}
        effective_signature = {**effective_obs, **self.params}
        return self.f(**effective_signature)


def make_agent(f, params, input_keys=None, output_keys=None):
    params_list_ = set()
    input_keys_ = set()
    signature_ = inspect.signature(f)
    for p in signature_.parameters:
        if str(p) not in input_keys:
            params_list_.update({p})
        else:
            input_keys_.update({p})
    default_params_ = {
        k: v.default
        for k, v in signature_.parameters.items()
        if v.default is not inspect.Parameter.empty and k not in input_keys_
    }
    agent = WrapperAgent(
        f=f,
        params=params,
        defaults=default_params_,
    )
    agent.update_keys(input_keys=input_keys, output_keys=output_keys)
    return agent


def train(agent, env):
    done = False
    while not done:
        obs, done = env.step()
        action = agent(obs)
        env.update(action)
