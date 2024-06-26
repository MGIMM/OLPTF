import os
import time
import pickle
from dataclasses import dataclass
from collections.abc import Iterable
from .abstract import AbstractAgent, AbstractEnv


@dataclass
class Agent(AbstractAgent):
    def __post_init__(self):
        super().__post_init__()
        self._obs = dict()
        self._input_keys = set()
        self._effective_input_keys = set()
        self._output_keys = set()
        self._label = self.__repr__()
        self._log = dict()
        self._runtime_counter = 0
        self._time_id = None
        self.update_attrs()

    def __init_subclass__(cls, keys: list or set = None, **kwargs):
        """initialization of input_keys when creating the class.

        Args:
            keys (list or set): keys
        """
        super().__init_subclass__(**kwargs)
        if keys is not None:
            cls.input_keys = set(keys)

    def __call__(self, state: dict) -> dict:
        """wrapper of observe and act.

        Args:
            state (dict): state

        Returns:
            dict: action
        """
        if state is not None:
            lt = time.localtime()
            self._time_id = (
                f"id-{self._runtime_counter}-{lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}"
            )
            t_start = time.time()

            # only entries in self.input_keys are loaded, c.f. setter of self.obs
            self.obs = state
            action = self.act()

            runtime = time.time() - t_start
            self.log = {"runtime": {self._time_id: runtime}}
            self._runtime_counter += 1
        else:
            action = None
        if action is not None:
            assert isinstance(
                action, dict
            ), f"action of {self.label} should be dict or None."
            self.output_keys = set(action.keys())
        return action

    def act(self) -> dict: ...

    @property
    def input_keys(self):
        return self._input_keys

    @input_keys.setter
    def input_keys(self, keys):
        if not isinstance(keys, set):
            keys = set(keys)
        self._input_keys = keys
        self.update_attrs()

    @property
    def effective_input_keys(self):
        return self._effective_input_keys

    @effective_input_keys.setter
    def effective_input_keys(self, keys):
        if not isinstance(keys, set):
            keys = set(keys)
        self._effective_input_keys = keys
        self.update_attrs()

    @property
    def output_keys(self):
        return self._output_keys

    @output_keys.setter
    def output_keys(self, keys):
        if not isinstance(keys, set):
            keys = set(keys)
        self._output_keys = keys
        self.update_attrs()

    @property
    def obs(self):
        return self._obs

    @obs.setter
    def obs(self, state):
        self.effective_input_keys = set()
        assert isinstance(state, dict), "state should be dict."
        self.effective_input_keys = self.input_keys.intersection(state.keys())
        self._obs.update({k: state[k] for k in self.effective_input_keys})
        self.update_attrs()

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label):
        assert isinstance(label, str), "label should be str."
        self._label = label

    @property
    def log(self):
        return self._log

    @log.setter
    def log(self, info):
        for field, info_dict in info.items():
            try:
                self._log[field].update(info_dict)
            except KeyError:
                self._log[field] = dict()
                self._log[field].update(info_dict)


@dataclass
class PipelineAgent(Agent):
    agents: list
    flatten: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.flatten:
            self.flatten_agents()
        self.input_keys = set()
        for agent_ in self.agents:
            self.input_keys.update(agent_.input_keys)

    def flatten_agents(self):
        """flatten agents in self.agents."""
        _agents = list()
        for agent in self.agents:
            if hasattr(agent, "agents"):
                for agent_ in agent.agents:
                    _agents.append(agent_)
                    self.attrs.update({agent_.__class__.__name__: agent_.attrs})
            else:
                _agents.append(agent)
        self.agents = _agents

    def act(self) -> dict:
        """action of a sequence of agents.

        output of former Agent only updates self.obs,
        and can be used by following agents.
        """
        actions = dict()
        for agent in self.agents:
            action = agent(self.obs)
            if action is not None:
                self.obs.update(action)
                actions.update(action)
        return actions


@dataclass
class Env(AbstractEnv):
    stream: Iterable = None
    init_state: dict = None

    def __post_init__(self):
        super().__post_init__()
        if self.stream is not None:
            self._iter = iter(self.stream)
        self._state = self.init_state or dict()

    def update_stream(self):
        if self.stream is not None:
            data = next(self._iter)
            self.state.update(data)
        return self.state

    def step(self):
        try:
            _state = self.update_stream()
            done = False
        except StopIteration:
            _state = None
            done = True
        return _state, done

    @property
    def state(self):
        return self._state

    def interaction(self, action):
        if action is not None:
            self.state.update(action)


# online dynamic
def train(agent: Agent or PipelineAgent, env: Env):
    done = False
    while not done:
        state, done = env.step()
        action = agent(state)
        env.interaction(action)


# debugging
def save(obj, path):
    dirname = os.path.abspath(os.path.dirname(path))
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(path, "wb") as file:
        pickle.dump(obj, file)


def load(path):
    pickle.load(open(path, "rb"))
