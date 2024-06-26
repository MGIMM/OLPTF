import networkx as nx
from copy import deepcopy
from .concrete import PipelineAgent, Agent
from pandas import Series, DataFrame


def _check_flow(_from, _to):
    """naive check if there is data flow from _from to _to.

    Args:
        _from (Agent): _from candidate.
        _to (Agent):  _to candidate.

    Returns:
        bool: bool of child-parent relationship.
    """
    return _from.output_keys.intersection(_to.effective_input_keys)


def get_graph(agents, show_data_keys=True):
    """get networkx graph object of a list of executed agents (with effective_input_keys and output_keys).

    Args:
        agents (list of Agent): list of executed Agent objects.
        show_data_keys (bool, optional): whether to show data keys in the flow. Defaults to True.

    Returns:
        networkx.DiGraph(): networkx directed graph object.
    """
    graph = nx.DiGraph()
    graph.add_node("state")
    # track agents that have inflowed data from agents before
    inflowed = set()
    # track agents that have outflowed data to agents after
    outflowed = set()
    queue = list()
    for flow in agents:
        graph.add_node(flow.label)
    for current in agents:
        if not queue:
            queue.append(current)
        else:
            for ag in reversed(queue):
                commuted_data_keys = _check_flow(_from=ag, _to=current)
                if commuted_data_keys:
                    inflowed.update({current.label})
                    outflowed.update({ag.label})
                    if show_data_keys:
                        keys_from_state = sorted(
                            current.effective_input_keys - commuted_data_keys
                        )
                        if keys_from_state:
                            for k in deepcopy(keys_from_state):
                                for _ag in reversed(queue[:-1]):
                                    if k in _ag.output_keys:
                                        keys_from_state.remove(k)
                        if keys_from_state:
                            str_keys_from_state = f"[{','.join(keys_from_state)}]"
                            graph.add_edge("state", str_keys_from_state)
                            graph.add_edge(str_keys_from_state, current.label)
                        # communicated data keys from ag to current
                        str_commuted_data_keys = (
                            f"[{','.join(sorted(commuted_data_keys))}]"
                        )
                        if commuted_data_keys.issubset(current.output_keys):
                            graph.add_edge(
                                ag.label,
                                str_commuted_data_keys + f"(from {ag.label})",
                            )
                            graph.add_edge(
                                str_commuted_data_keys + f"(from {ag.label})",
                                current.label,
                            )
                        else:
                            graph.add_edge(ag.label, str_commuted_data_keys)
                            graph.add_edge(str_commuted_data_keys, current.label)
                    else:
                        graph.add_edge(ag.label, current.label)
            queue.append(current)
    if show_data_keys:
        # add interactions between the start/end agent and state
        for root in agents:
            if root.label not in inflowed:
                keys = root.effective_input_keys
                str_keys = f"[{','.join(sorted(keys))}]"
                graph.add_edge("state", str_keys)
                graph.add_edge(str_keys, root.label)
        for out in agents:
            if out.label not in outflowed:
                keys = out.output_keys
                str_keys = f"[{','.join(sorted(keys))}]"
                if keys.issubset(out.effective_input_keys):
                    graph.add_edge(str_keys + f"(from {out.label})", "state")
                    graph.add_edge(out.label, str_keys + f"(from {out.label})")
                else:
                    graph.add_edge(str_keys, "state")
                    graph.add_edge(out.label, str_keys)
    else:
        for root in agents:
            if root.label not in inflowed:
                graph.add_edge("state", root.label)
        for out in agents:
            if out.label not in outflowed:
                graph.add_edge(out.label, "state")
    return graph


def viz_flow(agent, show_data_keys=True, ascii_only=False, verbose=True):
    """viz agent flow with quasi ascii style.

    Args:
        agent (Agent or PipelineAgent): An executed PipeplineAgent object (with .agents attr and effective_input_keys/output_keys for all of them).
        show_data_keys (bool, optional): whether to show data keys flow. Defaults to True.
        ascii_only (bool, optional): whether to use pure ascii output. Defaults to False.
        verbose (bool, optional): whether to show runtime stats. Defaults to True.
    """
    if isinstance(agent, Agent):
        agent = PipelineAgent(agents=[agent])
    graph = get_graph(agent.agents, show_data_keys=show_data_keys)
    nx.write_network_text(graph, ascii_only=ascii_only)
    if verbose:
        runtime = dict()
        for ag in agent.agents:
            runtime[ag.label] = Series(ag.log["runtime"]).mean()
        runtime = Series(runtime)
        print(
            f"---Average runtime:\n{runtime.to_string(float_format='{:,.3e}s'.format)}"
        )
