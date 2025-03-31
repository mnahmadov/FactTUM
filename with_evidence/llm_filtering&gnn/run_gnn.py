import argparse

import torch
import torch.nn as nn
import transformers
import pickle
from pathlib import Path


#from glocal_settings import SMALL, SMALL_SIZE
from models import QAGNN
from train import train


from dataset import get_graph_dataloader
from evaluate import evaluate_on_test_set
from models import QAGNN, get_bert_model
from train import train
from utils import get_logger, save_history, seed_everything, set_global_log_level, load_state_dict

logger = get_logger(__name__)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("model_name", type=str, help="The name of the model, will be saved to file. ")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--n_epochs", type=int, default=5, help="The amount of epochs to train for. ")
    parser.add_argument("--use_roberta", action="store_true", help="Use RoBERTa instead of BERT. ")
    parser.add_argument("--freeze_base_model", action="store_true", help="Freeze bert base model. ")
    parser.add_argument("--freeze_up_to_pooler", action="store_true", help="Freeze bert up to last pooling layer. ")
    parser.add_argument("--bert_dropout", type=float, default=0, help="Dropout rate for bert classification layer. ")
    parser.add_argument("--gnn_dropout", type=float, default=0.3, help="Dropout rate for GNN layers. ")
    parser.add_argument("--classifier_dropout", type=float, default=0.3, help="Dropout rate for QA-GNN classifier. ")
    parser.add_argument("--lm_layer_dropout", type=float, default=0.3, help="Dropout for the optional lm layer. ")
    parser.add_argument("--n_gnn_layers", type=int, default=2, help="Number of GNN layers in QA-GNN. ")
    parser.add_argument("--gnn_batch_norm", action="store_true", help="Use Batch Norm between GNN layers. ")
    parser.add_argument("--gnn_hidden_dim", type=int, default=256, help="Size if hidden dimension in GNN layers. ")
    parser.add_argument("--gnn_out_features", type=int, default=256, help="Size of GNNs last layers output. ")
    parser.add_argument("--lm_layer_features", type=int, default=0, help="Size of optional LM layer. ")
    parser.add_argument("--evaluate_only", action="store_true", help="Do not train, only evaluate. ")
    parser.add_argument("--state_dict_path", type=str, default="", help="Path to state-dict, if `--evaluate-only`. ")
    parser.add_argument("--vectorized", action="store_true", help="Vectorize GNN processing. ")

    parser.add_argument("--subgraph_type", choices=["direct","combined"],
                        type=str, default="none", help=help)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    set_global_log_level("debug")
    seed_everything(42)
    args = get_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.evaluate_only and args.state_dict_path == "":
        raise ValueError("Argument --state_dict_path must be provided when `--evaluate_only`. ")
    if args.subgraph_type.lower() == "none":
        raise ValueError("Argument `subgraph_type` can not be `none` when using qa-gnn. ")
    if args.lm_layer_features == 0:
        args.lm_layer_features = None

    
    model = QAGNN(
            args.model_name, n_gnn_layers=args.n_gnn_layers, gnn_hidden_dim=args.gnn_hidden_dim,
            gnn_out_features=args.gnn_out_features, lm_layer_features=args.lm_layer_features,
            gnn_batch_norm=args.gnn_batch_norm, freeze_base_model=args.freeze_base_model,
            freeze_up_to_pooler=args.freeze_up_to_pooler, gnn_dropout=args.gnn_dropout,
            classifier_dropout=args.classifier_dropout, lm_layer_dropout=args.lm_layer_dropout,
            use_roberta=args.use_roberta)

    train_loader = get_graph_dataloader(
            "train", subgraph_type=args.subgraph_type, batch_size=args.batch_size)
    val_loader = get_graph_dataloader(
            "val", subgraph_type=args.subgraph_type,
            batch_size=args.batch_size)
    test_loader = get_graph_dataloader(
            "test", subgraph_type=args.subgraph_type,
            batch_size=args.batch_size // 4,
            shuffle=False, drop_last=False)
    criterion = nn.BCEWithLogitsLoss()
    if not args.evaluate_only:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        lr_scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=50, num_training_steps=len(train_loader) * args.n_epochs
        )

        history, models_dict = train(
            model=model, criterion=criterion, optimizer=optimizer, train_loader=train_loader,
            val_loader=val_loader, n_epochs=args.n_epochs, scheduler=lr_scheduler)

        logger.info(f"Loading best model state dict from epoch {history['best_epoch']}...")
        model.load_state_dict(models_dict["best_model_state_dict"])
        logger.info("Done loading state dict. ")
    else:
        logger.info(f"Loading best model state dict from path {args.state_dict_path}...")
        model.load_state_dict(load_state_dict(args.state_dict_path, device))
        logger.info("Done loading state dict. ")
        history = {}
    test_results = evaluate_on_test_set(model=model, test_loader=test_loader, criterion=criterion, device=device)

    test_accuracy = test_results["overall"]["accuracy"]
    average_test_loss = test_results["overall"]["loss"]
    history["test_accuracy"] = test_accuracy
    history["test_results"] = test_results
    if not args.evaluate_only:
        save_history(args.model_name, history)
    else:
        save_history("eval_" + args.model_name, history)
    logger.info(f"Model \"{args.model_name}\": Test accuracy: {test_accuracy * 100:.4f}%, {average_test_loss=:.4f}. ")
    logger.info(f"{test_results=}")