from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict

BASE_SKIP_FIELDS = [
    "abc_impl",
    "act",
    "attrs",
    "_attrs",
    "update_attrs",
    "update_keys",
]


@dataclass
class Base(ABC):
    def __post_init__(self):
        self._attrs = set()

    @property
    def attrs(self):
        return self._attrs

    def update_attrs(self):
        for _attr in self.__dir__():
            if (
                not _attr.endswith("_")
                and not _attr.startswith("_")
                and _attr not in BASE_SKIP_FIELDS
            ):
                self.attrs.update({_attr})


@dataclass
class AbstractAgent(Base):
    @abstractmethod
    def __call__(self, obs: Dict) -> Dict:
        """wrapper: obs -> agent -> action"""
        ...

    @abstractmethod
    def act(self) -> Dict:
        """self.obs -> action"""
        ...

    @property
    @abstractmethod
    def obs(self):
        ...

    @property
    @abstractmethod
    def input_keys(self):
        """state -> input_keys -> self.obs"""
        ...


@dataclass
class AbstractEnv(Base):
    @abstractmethod
    def update_stream(self) -> Dict:
        ...

    @abstractmethod
    def step(self) -> Dict:
        ...

    @property
    @abstractmethod
    def state(self):
        ...

    @abstractmethod
    def interaction(self):
        ...