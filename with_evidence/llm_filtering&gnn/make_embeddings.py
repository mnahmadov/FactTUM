from models import get_bert_model
import argparse
import os
import pickle
from pathlib import Path
import torch
import argparse
import os
import pickle
from pathlib import Path
from transformers import AutoTokenizer
from convert_subgraphs import get_subgraph
from utils import get_logger, seed_everything, set_global_log_level
from constants import EMBEDDINGS_FILENAME,DATA_PATH

logger = get_logger(__name__)

def calculate_embeddings(text, tokenizer, model, with_classifier=True):
    """
    Calculate embeddings for text, given a tokenizer and a model

    Args:
        text (list of str): List of the strings to make embeddings for.
        tokenizer (tokenizer): The tokenizer.
        model (pytorch model): The model.
        with_classifier (bool): If model has classifier (should reach hidden state) or not (output is hidden state).

    Returns:
        dict: Dict mapping from text to embeddings.
    """
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    inputs = {key: val.to(model.device) for key, val in inputs.items()}  # Move to device

    with torch.no_grad():
        outputs = model(**inputs)

    # Get the embedding for the [CLS] token (first token)
    if with_classifier:
        last_hidden_states = outputs.hidden_states[-1]
    else:
        last_hidden_states = outputs.last_hidden_state
    cls_embedding = last_hidden_states[:, 0, :]
    return cls_embedding

def get_entity_and_relations_list(graph):
    nodes = []
    edges = []
    for entity, relations in graph:
        for relation in relations:
            if relation not in edges:
                edges.append(relation)
        if entity not in nodes:
            nodes.append(entity)
    return nodes,edges
            

def get_all_embeddings(split_name,subgraph_df, tokenizer, model, batch_size=32):
    file_path = Path(DATA_PATH) / EMBEDDINGS_FILENAME / f"{split_name}.pickle"
    embedding_dict = {}

    all_entities_and_relations = set()  # Get all text (entities and relations) from all graphs
    for graph in subgraph_df:
        for triple in graph:
            entity, relation, neighbor = triple  # Unpack the triple
        all_entities_and_relations.update(entity)
        all_entities_and_relations.update(relation)
        all_entities_and_relations.update(neighbor)

    # Convert set to list and remove already computed embeddings
    all_text = [item for item in all_entities_and_relations if item not in embedding_dict]

    # Get embeddings in batches.
    n_text = len(all_text)
    for i in range(0, n_text, batch_size):
        if (i % 1) == 0:
            print(f"On idx {i}/{n_text}/{split_name}")
        batch_texts = all_text[i:i + batch_size]
        embeddings = calculate_embeddings(batch_texts, tokenizer, model)
        for text, embedding in zip(batch_texts, embeddings):
            embedding_dict[text] = embedding.cpu().numpy()  # Move embeddings to CPU and convert to numpy for storage

    with open(file_path, "wb") as outfile:
        pickle.dump(embedding_dict, outfile)
    return embedding_dict

if __name__ == "__main__":
    set_global_log_level("debug")
    seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", choices=["train", "val", "test", "all"], default="all",
                        help="Dataset split to make subgraph for. ")
    parser.add_argument("--subgraph_type", choices=["direct", "combined"], default="direct",
                        help="The subgraph retrieval method to load. ")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size to calculate embeddings. ")

    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.dataset_type == "all":
        dataset_types = ["train", "val", "test"]
    else:
        dataset_types = [args.dataset_type]

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    for dataset_type in dataset_types:
        subgraph_df = get_subgraph(dataset_type, args.subgraph_type)
        model = get_bert_model("bert").to(device)
        embeddings = get_all_embeddings(dataset_type,subgraph_df, tokenizer, model, batch_size=args.batch_size)

