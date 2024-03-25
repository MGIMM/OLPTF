from AbstractOnlineStructure.concrete import (
    Env,
    Agent,
    PipelineAgent,
    train,
)

from dataclasses import dataclass

@dataclass
class AddNum(Agent):
    num:int

    def act(self, obs):
        a = obs.get("a",1.)
        action = dict(a=a+self.num)
        return action

def test_train():
    def data_stream():
        for i in range(20):
            yield {"a": i}
    add3 = AddNum(num=3)
    env = Env(data_stream())
    train(add3, env)

def test_pipeline_agent():
    def data_stream():
        for i in range(20):
            yield {"a": i}
    add3 = AddNum(num=3)
    add5 = AddNum(num=5)
    agent = PipelineAgent([add3, add5])
    env = Env(data_stream())
    train(agent, env)