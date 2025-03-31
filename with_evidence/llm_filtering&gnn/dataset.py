import os
import pickle
from pathlib import Path
from constants import DATA_PATH, EMBEDDINGS_FILENAME,BERT_LAST_LAYER_DIM,PATH_DATA
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from transformers import AutoTokenizer
from convert_subgraphs import get_subgraph

def get_df(data_split=None, small=None):
    """
    Read and returns a dataframe of the FactKG dataset.

    Args:
        data_split (str): Which datasplit to load, in `train`, `val` or `test`

    Raises:
        ValueError: If `data_split` is an unsuported string.

    Returns:
        pd.DataFrame: DataFrame of the dataset.
    """
    if data_split=='val':
        data_split='dev'
    path = Path(DATA_PATH) / f"{data_split}_data_all.pickle"
    df = pd.DataFrame.from_dict(pd.read_pickle(path), orient="index")
    df.reset_index(inplace=True)  # Fix so sentences are a column, not index
    df.rename(columns={"index": "Sentence"}, inplace=True)
    return df

def get_precomputed_embeddings(split_name):
    """
    Gets dict with precomputed embeddings, made with `make_subgraph_embeddings.py`.

    Raises:
        ValueError: If `data_split` is an unsuported string.

    Returns:
        dict: The dict of the subgraphs (as strings).
    """
    file_path = Path(DATA_PATH) / EMBEDDINGS_FILENAME / f"{split_name}.pickle"
    embedding_dict = pickle.load(open(file_path, "rb"))
    return embedding_dict


class CollateFunctor:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        inputs, labels = zip(*batch)
        labels = torch.tensor(labels)
        inputs = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
        inputs["labels"] = torch.as_tensor(labels)
        return inputs


def get_embedding(text, embeddings_dict, tokenizer, model):

    if embeddings_dict.get(text) is not None:
        return torch.tensor(embeddings_dict[text])
    return torch.zeros(BERT_LAST_LAYER_DIM)


def convert_to_pyg_format(graph, online_embeddings, embedding_dict=None, tokenizer=None, model=None):
    """
    Convert graph on DBpedia dict format to torch_embedding.data format, so it can be run in GNN.

    Args:
        graph (dict): Dict of graph, gotten by calling `kg.search()` on each element in the graph.
        online_embeddings (bool): If True, will calculate embeddings for knowledge subgraph online, with a model
                that might be tuned during the training.
        embedding_dict (dict): Dict mapping words to embeddings, to be used as node and edge features.
            This should be precomputed.
        tokenizer (tokenizer): Tokenizer to `model` if `online_embeddings`.
        model (pytroch model): Model to compute embeddings if `online_embeddings`.

    Returns:
        torch_geometric.data: Graph data.
    """

    if graph == []:  # Dummy empty graph. Not actually empty because of vectorized computations.
        graph = [["none", "none", "none"]]
    node_to_index = {}  # Node text to int mapping
    edge_to_index = {}  # Same for edges
    node_features = []  # List of embeddings
    edge_features = []  # Same for edges
    edge_indices = []

    current_node_idx = 0
    current_edge_idx = 0
    for edge_list in graph:
        node1, edge, node2 = edge_list  # Graph consists of list on the format [node1, edge, node2]

        if node1 not in node_to_index:
            node_to_index[node1] = current_node_idx
            embedding = get_embedding(node1, embeddings_dict=embedding_dict,
                                      tokenizer=tokenizer, model=model)
            node_features.append(embedding)
            current_node_idx += 1

        if node2 not in node_to_index:
            node_to_index[node2] = current_node_idx
            embedding = get_embedding(node2, embeddings_dict=embedding_dict,
                                      tokenizer=tokenizer, model=model)
            node_features.append(embedding)
            current_node_idx += 1

        if edge not in edge_to_index:
            edge_to_index[edge] = current_edge_idx
            embedding = get_embedding(edge, embeddings_dict=embedding_dict,
                                      tokenizer=tokenizer, model=model)
            edge_features.append(embedding)
            current_edge_idx += 1

        edge_indices.append([node_to_index[node1], node_to_index[node2]])

    edge_index = torch.tensor(edge_indices).t().contiguous()  # Transpose and make memory contigious
    x = torch.stack(node_features)
    edge_attr = torch.stack(edge_features)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


class FactKGDatasetGraph(Dataset):
    def __init__(self, df, evidence, embedding_dict=None, tokenizer=None, model=None):
        """
        Initialize the dataset. This dataset will return tokenized claims, graphs for the subgraph, and labels.

        Args:
            df (pd.DataFrame): FactKG dataframe
            evidence (pd.DataFram): Dataframe with the subgraphs, found by `retrieve_subgraphs.py`.
            online_embeddings (bool): If True, will calculate embeddings for knowledge subgraph online, with a model
                that might be tuned during the training.
            embedding_dict (dict): Dict mapping the knowledge graph words to embeddings if not `online_embeddings`.
            tokenizer (tokenizer): Tokenizer to `model` if `online_embeddings`.
            model (pytroch model): Model to compute embeddings if `online_embeddings`.
            mix_graphs (bool, optional): If `True`, will use both the connected and the walkable graphs found in
                DBpedia. If `False`, will use connected if it is not empty, else walkable. Defaults to False.
        """
        self.inputs = df["Sentence"]
        self.labels = [int(label[0]) for label in df["Label"]]
        self.length = len(df)
        self.subgraphs = evidence
        self.online_embeddings = False
        self.embedding_dict = embedding_dict
        self.tokenizer = tokenizer
        self.model = model
        self.mix_graphs = True

    def __getitem__(self, idx):
        claims = self.inputs[idx]
        subgraph = self.subgraphs[idx]
        graph = convert_to_pyg_format(
            subgraph, online_embeddings=self.online_embeddings, embedding_dict=self.embedding_dict,
            tokenizer=self.tokenizer, model=self.model)

        label = self.labels[idx]
        return claims, graph, label

    def __len__(self):
        return self.length


class GraphCollateFunc:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        inputs, graph_batch, labels = zip(*batch)
        tokens = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True,
                                max_length=self.max_length)
        graph_batch = Batch.from_data_list(graph_batch)
        labels = torch.tensor(labels).float()

        return tokens, graph_batch, labels

def get_graph_dataloader(data_split, subgraph_type,model=None, bert_model_name="bert-base-uncased",max_length=512, batch_size=128, shuffle=True, drop_last=True):
    """
    Creates a dataloader for dataset with subgraph representation and tokenized text.

    Args:
        data_split (str): Which datasplit to load, in `train`, `val` or `test`
        subgraph_type (str): The type of subgraph to use. Must be either `direct` (only the direct entity
            neighbours), `direct_filled` (the direct entity neigbhours, but if it is empty, replace it with
            all of the entity edges if the entities), `one_hop` (all of the entity edges) or `relevant` (direct plus
            edges that appears in claim).
        bert_model_name (str, optional): Name of model, in order to get tokenizer. Defaults to "bert-base-uncased".
        max_length (int, optional): Max tokenizer length. Defaults to 512.
        batch_size (int, optional): Batch size to dataloader. Defaults to 128.
        shuffle (bool, optional): Shuffle dataset. Defaults to True.
        drop_last (bool, optional): Drop last batch if it is less than `batch_size`. Defaults to True.
        mix_graphs (bool, optional): If `True`, will use both the connected and the walkable graphs found in
                DBpedia. If `False`, will use connected if it is not empty, else walkable. Defaults to False.

    Returns:
        DataLoader: The dataloader.
    """
    df = get_df(data_split)
    subgraphs=get_subgraph(data_split,subgraph_type)
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    embedding_dict=get_precomputed_embeddings(data_split)
    graph_collate_func = GraphCollateFunc(tokenizer, max_length=max_length)
    dataset = FactKGDatasetGraph(
            df, subgraphs, embedding_dict=embedding_dict)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                            collate_fn=graph_collate_func)
    return dataloader
