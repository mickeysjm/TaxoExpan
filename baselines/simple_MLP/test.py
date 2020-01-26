import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data_loader import MAGDatasetSlim, EdgeDataLoader, SubGraphDataLoader, AnchorParentDataLoader
from model import MLP, DeepSetMLP, DeepAPGMLP, bce_loss, macro_averaged_rank, batched_topk_hit_1, batched_topk_hit_3, batched_topk_hit_5, batched_scaled_MRR
from tqdm import tqdm


def test(args):
    # setup multiprocessing instance
    torch.multiprocessing.set_sharing_strategy('file_system')

    # setup data_loader instances
    if args.arch == "MLP":
        test_data_loader = EdgeDataLoader(mode="test", data_path=args.data, batch_size=1, shuffle=True, num_workers=4, batch_type="large_batch")
    elif args.arch == "DeepSetMLP":
        test_data_loader = SubGraphDataLoader(mode="test", data_path=args.data, batch_size=1, shuffle=True, num_workers=4, batch_type="large_batch")
    elif args.arch == "DeepAPGMLP":
        test_data_loader = AnchorParentDataLoader(mode="test", data_path=args.data, batch_size=1, shuffle=True, num_workers=4, batch_type="large_batch")

    # setup device
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    # load model
    if args.arch == "MLP":
        model = MLP(vocab_size=29654, embed_dim=250, first_hidden=1000, second_hidden=500, activation=nn.LeakyReLU())
        # model = MLP(vocab_size=431416, embed_dim=250, first_hidden=1000, second_hidden=500, activation=nn.LeakyReLU())
    elif args.arch == "DeepSetMLP":
        model = DeepSetMLP(vocab_size=29654, embed_dim=250, first_hidden=1500, second_hidden=1000, activation=nn.LeakyReLU())
        # model = DeepSetMLP(vocab_size=431416, embed_dim=250, first_hidden=1500, second_hidden=1000, activation=nn.LeakyReLU())
    elif args.arch == "DeepAPGMLP":
        model = DeepAPGMLP(vocab_size=29654, embed_dim=250, first_hidden=2000, second_hidden=1000, activation=nn.LeakyReLU())
    checkpoint = torch.load(args.resume)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # get function handles of loss and metrics
    loss_fn = bce_loss
    metric_fn = [macro_averaged_rank, batched_topk_hit_1, batched_topk_hit_3, batched_topk_hit_5, batched_scaled_MRR]

    # start evaluation on test data
    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fn))
    
    with torch.no_grad():
        for batched_examples in tqdm(test_data_loader):
            energy_scores = []
            all_labels = []
            if len(batched_examples) == 3:
                batched_parents, batched_children, batched_labels = batched_examples[0], batched_examples[1], batched_examples[2]
                for parents, children, labels in zip(batched_parents, batched_children, batched_labels):
                    parents, children = parents.to(device), children.to(device)
                    prediction = model(parents, children).to(device)
                    loss = loss_fn(prediction, labels.to(device))
                    total_loss += loss.item()
                    energy_scores.extend(prediction.squeeze_().tolist())
                    all_labels.extend(labels.tolist())
            elif len(batched_examples) == 4:
                batched_parents, batched_siblings, batched_children, batched_labels = batched_examples[0], batched_examples[1], batched_examples[2], batched_examples[3]
                for parents, siblings, children, labels in zip(batched_parents, batched_siblings, batched_children, batched_labels):
                    parents, siblings, children = parents.to(device), siblings.to(device), children.to(device)
                    prediction = model(parents, siblings, children).to(device)
                    loss = loss_fn(prediction, labels.to(device))
                    total_loss += loss.item()
                    energy_scores.extend(prediction.squeeze_().tolist())
                    all_labels.extend(labels.tolist())
            elif len(batched_examples) == 5:
                batched_parents, batched_siblings, batched_grand_parents, batched_children, batched_labels = batched_examples[0], batched_examples[1], batched_examples[2], batched_examples[3], batched_examples[4]
                for parents, siblings, grand_parents, children, labels in zip(batched_parents, batched_siblings, batched_grand_parents, batched_children, batched_labels):
                    parents, siblings, grand_parents, children = parents.to(device), siblings.to(device), grand_parents.to(device), children.to(device)
                    prediction = model(parents, siblings, grand_parents, children).to(device)
                    loss = loss_fn(prediction, labels.to(device))
                    total_loss += loss.item()
                    energy_scores.extend(prediction.squeeze_().tolist())
                    all_labels.extend(labels.tolist())
            
            energy_scores = torch.tensor(energy_scores).unsqueeze_(1)
            all_labels = torch.tensor(all_labels)

            # computing metrics on test set
            for i, metric in enumerate(metric_fn):
                total_metrics[i] += metric(energy_scores, all_labels)
        
    n_samples = test_data_loader.n_samples
    print(f"Test loss: {total_loss / n_samples}")
    for i in range(len(metric_fn)):
        print(f"{metric_fn[i].__name__} : {total_metrics[i].item() / n_samples}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/home/t-jishen/StructureExpan/data/MAG_FoS/computer_science.pickle.20190624.bin')
    parser.add_argument('--arch', type=str, choices=["MLP", "DeepSetMLP", "DeepAPGMLP"], default='DeepSetMLP')
    parser.add_argument('--resume', type=str, help='model path')
    parser.add_argument('--device', type=int, help='gpu index')
    args = parser.parse_args()
    print(f"Parameters: {args}")
    assert os.path.exists(args.resume), f"model path {args.resume} must exist"
    test(args)
