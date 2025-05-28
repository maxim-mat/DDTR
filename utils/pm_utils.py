from typing import Callable
import pandas as pd
import pm4py
from pm4py.objects.petri_net.utils import reachability_graph
import networkx as nx
import numpy as np
import pm4py
import torch
import torch.nn.functional as F
from itertools import groupby
from dataset.dataset import TraceDataset
from utils import Config


def convert_dataset_to_df(deterministic, stochastic, activity_names: dict):
    df_deterministic = pd.DataFrame(
        {
            'concept:name': [activity_names[i.item()] for trace in deterministic for i in trace],
            'case:concept:name': [str(i) for i, trace in enumerate(deterministic) for _ in range(len(trace))]
        }
    )

    stochastic_list = [x.unsqueeze(0) for x in stochastic]

    return df_deterministic, stochastic_list


def prepare_df_cols_for_discovery(df):
    df_copy = df.copy()
    df_copy.loc[:, 'order'] = df_copy.groupby('case:concept:name').cumcount()
    df_copy.loc[:, 'time:timestamp'] = pd.to_datetime(df_copy['order'])
    return df_copy


def convert_dataset_to_df_for_dicovery(deterministic, stochastic, cfg: Config):
    dk_process_df, _ = convert_dataset_to_df(deterministic, stochastic, cfg.activity_names)
    return prepare_df_cols_for_discovery(dk_process_df)


def resolve_process_discovery_method(method_name: str) -> Callable:
    match method_name:
        case "inductive":
            return pm4py.discover_petri_net_inductive
        case _:
            raise AttributeError(f"Unsupported discovery method: {method_name}")


def remove_duplicates_trace(trace):
    return torch.tensor([x.item() for x, _ in groupby(trace)])


def remove_duplicates_dataset(dataset: TraceDataset):
    stochastics = [x[1] for x in dataset]
    one_hot = torch.argmax(torch.stack([x[0] for x in dataset], axis=0).permute(0, 2, 1), dim=1)
    deterministics = [remove_duplicates_trace(x) for x in one_hot]
    return deterministics, stochastics


def dataset_to_list(dataset: TraceDataset):
    deterministics = torch.argmax(torch.stack([x[0] for x in dataset], axis=0).permute(0, 2, 1), dim=1)
    stochastics = torch.stack([x[1] for x in dataset], axis=0).permute(0, 2, 1)
    return deterministics, stochastics


def discover_dk_process(dataset: TraceDataset, cfg: Config, preprocess=dataset_to_list):
    deterministic, stochastic = preprocess(dataset)
    df_train = convert_dataset_to_df_for_dicovery(deterministic, stochastic, cfg)
    process_discovery_method = resolve_process_discovery_method(cfg.process_discovery_method)
    return process_discovery_method(df_train)


def pad_to_multiple_of_n(tensor, n=32):
    # Get the original spatial dimension (D)
    _, height, width = tensor.shape  # Expecting shape (1, C, D, D)

    # Compute the padding needed to make the dimensions divisible by 32
    pad_height = (n - height % n) % n
    pad_width = (n - width % n) % n

    # Since it's a square, pad equally along height and width
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    # Apply padding (PyTorch padding order: left, right, top, bottom)
    padded_tensor = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom))

    return padded_tensor


def get_process_model_reachability_graph_transition_matrix(process_model: pm4py.PetriNet, init_marking: pm4py.Marking):
    rg = reachability_graph.construct_reachability_graph(process_model, init_marking)

    rg_nx = nx.DiGraph()

    for state in rg.states:
        rg_nx.add_node(state.name)

    transition_names = {tuple(s.strip(" '") for s in transition.name.strip("()").split(","))[1] for transition in
                        rg.transitions}
    transition_name_index = {name: idx for idx, name in enumerate(sorted(transition_names))}

    for transition in rg.transitions:
        transition_name = tuple(s.strip(" '") for s in transition.name.strip("()").split(","))
        rg_nx.add_edge(
            transition.from_state.name,
            transition.to_state.name,
            label=transition_name
        )

    nodes = sorted(rg_nx.nodes())
    num_transitions = len(transition_names)
    num_nodes = len(nodes)
    transition_matrix = np.zeros((1, num_nodes, num_nodes), dtype=int)

    for edge in rg_nx.edges(data=True):
        from_node = nodes.index(edge[0])
        to_node = nodes.index(edge[1])
        transition_name = edge[2]['label'][1]
        if transition_name in transition_name_index:
            transition_idx = transition_name_index[transition_name]
            transition_matrix[0, from_node, to_node] = 1
        else:
            raise RuntimeError(f"somehow, transition: {transition_name} was encountered but not indexed")

    return rg_nx, transition_matrix


def get_process_model_reachability_graph_transition_multimatrix(process_model: pm4py.PetriNet,
                                                                init_marking: pm4py.Marking):
    rg = reachability_graph.construct_reachability_graph(process_model, init_marking)

    rg_nx = nx.MultiDiGraph()

    for state in rg.states:
        rg_nx.add_node(state.name)

    transition_names = {tuple(s.strip(" '") for s in transition.name.strip("()").split(","))[1] for transition in
                        rg.transitions}
    transition_name_index = {name: idx for idx, name in enumerate(sorted(transition_names))}

    for transition in rg.transitions:
        transition_name = tuple(s.strip(" '") for s in transition.name.strip("()").split(","))
        rg_nx.add_edge(
            transition.from_state.name,
            transition.to_state.name,
            label=transition_name
        )

    nodes = sorted(rg_nx.nodes())
    num_transitions = len(transition_names)
    num_nodes = len(nodes)
    transition_matrix = np.zeros((num_transitions, num_nodes, num_nodes), dtype=int)

    for edge in rg_nx.edges(data=True):
        from_node = nodes.index(edge[0])
        to_node = nodes.index(edge[1])
        transition_name = edge[2]['label'][1]
        if transition_name in transition_name_index:
            transition_idx = transition_name_index[transition_name]
            transition_matrix[transition_idx, from_node, to_node] = 1
        else:
            raise RuntimeError(f"somehow, transition: {transition_name} was encountered but not indexed")

    return rg_nx, transition_matrix


def get_process_model_petri_net_flow_matrix(process_model: pm4py.PetriNet, init_marking: pm4py.Marking,
                                            final_marking: pm4py.Marking):
    pn_nx = pm4py.convert_petri_net_to_networkx(process_model, init_marking, final_marking)
    transition_matrix = nx.adjacency_matrix(pn_nx).todense()

    return pn_nx, transition_matrix
