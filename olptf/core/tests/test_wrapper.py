from olptf.core.concrete import (
    make_agent,
)

def test_function_wrapper():
    def foo(a, b = 2, c = 3):
        return {"output":a+b+c}

    agent = make_agent(f = foo, params = {"b":3}, input_keys = ["a"], output_keys = ["output"])
    assert agent({"a":3})["output"] == 9
    assert agent.attrs["defaults"] == {"b":2, "c":3}
    assert agent.attrs["params"] == {"b":3}
    assert agent.input_keys == {"a"}
    assert agent.output_keys == {"output"}