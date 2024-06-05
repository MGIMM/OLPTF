from ..wrapper import (
    make_agent,
)


def foo(a, b=2, c=3):
    return {"output": a + b + c}


def test_function_wrapper():
    agent = make_agent(f=foo, params={"b": 3}, input_keys=["a", "d"])
    assert agent({"a": 3})["output"] == 9
    assert agent.defaults == {"b": 2, "c": 3}
    assert agent.params == {"b": 3}
    assert agent.input_keys == {"a", "d"}  # d is unused input_key
    assert agent.effective_input_keys == {"a"}
    assert agent.output_keys == {"output"}
