# ---
# title: olptf-01-intro 
# parent: 2024 
# collapse: 3
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# # overview 
#
# A online framework/fancy for-loop that separates data flow and algo management. 
#
#
# **license**
#
# The actual core part of the codes is shorter than this notebook (<400 lines), so do not hesitate if you prefer to just copy-paste it and make your own version.

# %% [markdown]
# # core mechanism
#
# There are 3 important ingredients that are essential to the pkg `olptf`: `Env`,`Agent`,`train`. 
#
# For all the imported key ingredients that are covered in this doc, one can use:

# %%
from olptf.core import Agent, Env, PipelineAgent, train, viz_flow, make_agent
from copy import deepcopy

# %% [markdown]
# The function `train()` basically explains how this framework operates.
# ```
# def train(agent, env):
#     done = False
#     while not done:
#         state, done = env.step()
#         action = agent(state)
#         env.interaction(action)
#
# ```

# %%
import networkx as nx
graph = nx.DiGraph()
graph.add_node("env")
graph.add_edge("env","env.step()")
graph.add_edge("env.step()","env.state")
graph.add_edge("env.state","agent(state)")
graph.add_edge("agent(state)", "agent.obs")
graph.add_edge("agent.obs","agent.act(agent.obs)")
graph.add_edge("agent.act(agent.obs)", "action")
graph.add_edge("action","env.interaction(action)")
graph.add_edge("env.interaction(action)", "env")
nx.write_network_text(graph, ascii_only=False)

# %% [markdown]
# Here, `env` should be understood as the data flow manager, and `agent` is basically the function/algo that precess the data in the flow.
#
# Such a structure seems useless and redundant when the task is easy, however, it can be useful if you have a complicated workflow that deals with daily/yearly for loop in a complicated manner.
#
# In the present notebook, we concentrate on simple examples to show how to get started with this framework.

# %% [markdown]
# # basic concept of agent and state
#
# A `state` is a dictionary that collects all the data entries that are processed or passed in the data loop.
#
# We start with a minimal state `{"number":1}`, we want to create a agent that that takes the number in it and output an action that adds a parametrized parameter `theta` to it.
#
# The `input_keys` should be defined when defining the class. We can, however, change it when the object is created. One can use any format that can be converted to a `set` to define it.
#
# After this step, the required entries will be loaded in `agent.obs[<data entry>]`.
#
# To get the output, i.e., `action`, one should use `agent(state)`, just like a function.

# %%
from dataclasses import dataclass

@dataclass
class AddNumber(Agent, keys = ["number"]):
    theta: float = None 
    def act(self):
        number = self.obs["number"]
        action = {
            "added_number":self.theta +  number
        }
        return action

state = {"number":1}
add_number = AddNumber(theta=10)
add_number(state)

# %% [markdown]
# ### management of data key entries
#
# After being executed for once, the agent `add_number` will automatically generate `agent.output_keys` and `agent.effective_input_keys`.

# %%
add_number.attrs #this collects meaningful attrs, it is good for debugging

# %% [markdown]
# It is possible to require key entries that is not presented in the state, and after being executed, the actual imported data key entries will be kept in `agent.effective_input_keys`. 
# This allows us to generalize and recycle the agent for multiple use cases.

# %%
from dataclasses import dataclass

@dataclass
class AddNumber(Agent, keys = ["number","sth_non_exist"]):
    theta: float = None 
    def act(self):
        number = self.obs["number"]
        action = {
            "added_number":self.theta +  number
        }
        return action

state = {"number":1}
add_number = AddNumber(theta=10)
add_number(state)
print("input_keys",add_number.input_keys)
print("effective_input_keys",add_number.effective_input_keys)


# %%
def foo(a, b=2, c=3, d=None):
    return {"output": a + b + c}


agent = make_agent(f=foo, params={"b": 3}, input_keys=["a","d"])
assert agent({"a": 3})["output"] == 9
assert agent.defaults == {"b": 2, "c": 3}
assert agent.params == {"b": 3}
assert agent.input_keys == {"a", "d"}  # d is unused input_key
assert agent.effective_input_keys == {"a"}
assert agent.output_keys == {"output"}

# %% [markdown]
# ### function agent wrapper
#
# If you do not like to write class and you prefer to create simple functions that operate on data, there is a wrapper in
# `from olptf.core import make_agent`
#
# Basically, you need to write a function that has the desired input keys in its signature, and you need to tell the wrapper what are the params you want it to take in a `dict`.
#
# Then, the wrapper with create an `agent` object and put all the important information in its attributes.
#
# You can put default value for input keys in the definition of your function, they will not be tracked as default params in the `WrapperAgent` object.

# %%
state = {"number":1}
def add_function(number, theta = 10, not_used = None):
    return {"added_number":theta+number}
add_function_agent = make_agent(add_function, params={"theta":5}, input_keys={"number","not_used"})
add_function_agent(state)

# %% [markdown]
# All key info of this function will be stocked in the attributes of this generated agent object.

# %%
add_function_agent.attrs

# %%
# since 'not_used' is in input keys, it is not counted as parameters
add_function_agent.defaults

# %%
add_function_agent.output_keys

# %%
add_function_agent.params

# %%
add_function_agent.effective_input_keys

# %% [markdown]
# ### PipelineAgent 
#
# The essential ingredient that allows us to do complicated things and to modularize them is realized through the class `PipelineAgent`. Roughly speaking, it is a sequence of agents that are executed one by one. The latter ones are able to use the outputs of the former ones. Let us do a sequence of basic algebraic manipulations to number to illustrate the idea.

# %%
state = {"number":1}
add_number = AddNumber(theta=10)
def mul_function(added_number, beta = 1, not_used = None):
    return {"multiplied_number":beta*added_number}
mul_agent = make_agent(mul_function,input_keys=["added_number"], params={"beta":5})
agents = [add_number, mul_agent] 
agent = PipelineAgent(agents=agents)
agent(state)

# %% [markdown]
# Once the input data keys are well prepared for each agent, there is no need to pay additional attention to the pipeline agent.
# Said differently, one should pay attention to each individual `agent` when doing dev. Once we make sure they work as intended, we can achieve complicated things by putting them together.

# %%
agent.attrs

# %% [markdown]
# ### viz flow
#
# There is a viz tool available for executed `PipelineAgent` object in `from olptf.core import viz_flow`.
# One has to make sure that the agent is executed on a given state, so that it has the right output_keys. In a more fundamental sense, one needs the information of the state to do the inference on the data flow.
#
# It can be helpful in debugging phase when the flow becomes complicated. By default, `agent.label` is equivalent to `agent.__repr__()`, which is generated when using `dataclass` to define the class.

# %%
viz_flow(agent)

# %% [markdown]
# So the benefit of using the class version is that it can have a prettier name in this viz by default.
#
# Nevertheless, one is able to manually set the label of an agent. This is in particular useful for wrapper agents.
# By default, the label is set to be `self.__repr__()`, and this is partially why it is recommended to use `@dataclass` wrapper.

# %%
agents = [add_number, mul_agent] 
mul_agent.label = "mul_agent()"
agent = PipelineAgent(agents=agents)
agent(state)
viz_flow(agent)

# %% [markdown]
# If you only want to have an idea on the dependence of the agents, one can use `show_data_keys=False`.
#
# This is useful when you only want to make sure no agents are connected with the wrong parent, since data key flows may be quite distracting sometimes. 
#
# In particular, one may wants to combine some agents when there are some block structure that is repeated several times in the whole pipeline.

# %%
viz_flow(agent, show_data_keys=False)

# %% [markdown]
# When there is ambiguity on the inflowed data keys, `viz_flow()` will automatically add a `(from <agent.label>)` to make it clearer.

# %%

@dataclass
class LinearOperation(Agent, keys= ["x","theta"]):
    intercept: float

    def act(self):
        x = self.obs["x"]
        # default value for a data key in the flow
        theta = self.obs.get("theta",1.0)
        action = dict(x=x*theta + self.intercept)
        return action
def data_stream():
    for i in range(5):
        yield {"x": i, "theta":0.5}

add3 = LinearOperation(intercept=3)
add5 = LinearOperation(intercept=5)
agent = PipelineAgent([add3, add5])
env = Env(stream=data_stream())
train(agent, env)
viz_flow(agent)

# %% [markdown]
# # data stream and env
#
# In our daily routine, a data stream is basically a for loop with yield function.
# A data stream feeds the env with new data for each iteration, and the data can be processed by algos/agents.

# %%
from tqdm.notebook import tqdm
def foo_stream():
    for i in tqdm(range(10)):
        state = {"number":i}
        yield state

env = Env(stream=foo_stream(), init_state=None)

# %%
state, done = env.step()
state

# %%
state, done = env.step()
state

# %% [markdown]
# By using `train()` function, we let agent and env interact with each other.

# %%
env = Env(stream=foo_stream(), init_state=None)
def mul_function(added_number, beta = 1, not_used = None):
    return {"multiplied_number":beta*added_number}
mul_agent = make_agent(mul_function,input_keys=["added_number"], params={"beta":5})
mul_agent.label = "mul()"
agents = [AddNumber(20), mul_agent] 
agent = PipelineAgent(agents=agents)
train(env=env, agent=agent)

# %% [markdown]
# After the training, the `env` and `agent ` keep every well defined entries in their attributes.

# %%
env.state

# %%
agent.output_keys

# %%
viz_flow(agent)

# %% [markdown]
# All the executed agent will keep the "trace" of the training in their attributes. For example, one can check the `.output_keys` of an agent that has been put in the pipeline. 

# %%
mul_agent.output_keys


# %% [markdown]
# You can easily create some monitor/dumper `agent` without `action`. 
# In practice, one can track the "results" in either `env` or a dumper `agent`. 

# %%
def foo_stream():
    for i in tqdm(range(10)):
        state = {"number":i}
        yield state

env = Env(stream=foo_stream(), init_state=None)
# first agent
add_number = AddNumber(20)
# second agent
def mul_function(added_number, beta = 1, not_used = None):
    return {"multiplied_number":beta*added_number}
mul_agent = make_agent(mul_function,input_keys=["added_number"], params={"beta":5})
mul_agent.label = "mul()"
# third agent
@dataclass
class Dumper(Agent, keys = ["added_number"]):
    counter:int=0
    def __post_init__(self):
        super().__post_init__()
        self.results = dict()
    def act(self):
        self.counter += 1
        self.results.update({self.counter:deepcopy(self.obs)})

dumper = Dumper()
# pipeline agent
agents = [add_number, mul_agent, dumper] 
agents = [add_number, dumper] 
agent = PipelineAgent(agents=agents)
# train
train(env=env, agent=agent)

# %%
viz_flow(agent)

# %%
dumper.results


# %% [markdown]
# ## good practice
#
# The idea of this framework is to modularize a complicated for-loop that separates data flow and algo management. So, when doing dev, the idea is to:
#
# * write and test every modular separately, to make sure the in and out are working as intended.
# * one is alway able to write and test a partial pipeline with `viz_flow()`.  
# * we check with the static state and ensure that the data keys flow as intended. 
# * when using monitor/dumper agent, one should pay attention to `deepcopy()`. In this sense, it's more convenient to store things in `state`, where one can initialize everything with `init_state` in `env`. This is in particular interesting when working with `jax`, where the performace is very sensitive to the change of `shape`.
# * the data key flow is essentially only visualized for the first level of the `state`, and one should use this to simplify the logic of data flow in the pipeline.
# * since `viz_flow()` outputs text-based viz, it is possible to use it in debugging mode.
#
# We present here a relatively easy way to dump your results in the state.

# %%
def foo_stream():
    for i in tqdm(range(10)):
        state = {
            "index":str(i),
            "number":i,
            }
        yield state

env = Env(stream=foo_stream(), init_state={"results":dict()})
# first agent
add_number = AddNumber(20)
# second agent
def mul_function(added_number, beta = 1, not_used = None):
    return {"multiplied_number":beta*added_number}
mul_agent = make_agent(mul_function,input_keys=["added_number"], params={"beta":5})
mul_agent.label = "mul()"
# third agent
@dataclass
class Collector(Agent, keys = ["multiplied_number","index","results"]):
    def act(self):
        output = self.obs["multiplied_number"]
        results = self.obs["results"]
        ind = self.obs["index"]
        results.update({ind:output})
        action = {"results":results}
        return action

collector = Collector()
# pipeline agent
agents = [add_number, mul_agent, collector] 
agent = PipelineAgent(agents=agents)
# train
train(env=env, agent=agent)

# %%
env.state["results"]

# %%
viz_flow(agent)

