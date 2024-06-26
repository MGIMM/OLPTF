from ..concrete import (
    Env,
    Agent,
    PipelineAgent,
    train,
)
from dataclasses import dataclass
import time


@dataclass
class LinearOperation(Agent, keys=["x", "theta"]):
    intercept: float

    def act(self):
        x = self.obs["x"]
        # default value for a data key in the flow
        theta = self.obs.get("theta", 1.0)
        action = dict(x=x * theta + self.intercept)
        return action


def test_agent():
    add3 = LinearOperation(intercept=3)
    state = {"x": 1.0}
    action = add3(state)
    assert action == {"x": 4.0}
    assert add3.input_keys == {"x", "theta"}
    assert add3.effective_input_keys == {"x"}
    assert add3.output_keys == {"x"}


def test_train():
    def data_stream():
        for i in range(5):
            yield {"x": i}

    add3 = LinearOperation(intercept=3)
    env = Env(data_stream())
    train(add3, env)


def test_pipeline_agent():
    def data_stream():
        for i in range(5):
            yield {"x": i, "theta": 0.5}

    add3 = LinearOperation(intercept=3)
    add5 = LinearOperation(intercept=5)
    agent = PipelineAgent([add3, add5])
    env = Env(stream=data_stream())
    train(agent, env)
    assert env.state["x"] == 7.5


def test_log():
    add3 = LinearOperation(intercept=3)
    state = {"x": 1.0}
    action = add3(state)
    state = {"x": 2.0}
    action = add3(state)
    assert len(add3.log["runtime"]) == 2
