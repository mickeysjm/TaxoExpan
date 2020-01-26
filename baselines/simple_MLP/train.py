import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data_loader import MAGDatasetSlim, EdgeDataLoader, SubGraphDataLoader, AnchorParentDataLoader
from model import MLP, DeepSetMLP, DeepAPGMLP, bce_loss, macro_averaged_rank, batched_topk_hit_1, batched_topk_hit_3, batched_topk_hit_5, batched_scaled_MRR
import time

def eval_metrics(metric_fn, output, target):
    acc_metrics = np.zeros(len(metric_fn))
    for i, metric in enumerate(metric_fn):
        acc_metrics[i] += metric(output, target)
    return acc_metrics

def valid_epoch(model, device, validation_loader, loss_fn, metric_fn):
    model.eval()
    total_val_loss = 0
    total_val_metrics = np.zeros(len(metric_fn))
    with torch.no_grad():
        for examples in validation_loader:
            if len(examples) == 3:
                parents, children, labels = examples[0].to(device), examples[1].to(device), examples[2].to(device)
                prediction = model(parents, children)
            elif len(examples) == 4:
                parents, siblings, children, labels = examples[0].to(device), examples[1].to(device), examples[2].to(device), examples[3].to(device)
                prediction = model(parents, siblings, children)
            elif len(examples) == 5:
                parents, siblings, grand_parents, children, labels = examples[0].to(device), examples[1].to(device), examples[2].to(device), examples[3].to(device), examples[4].to(device)
                prediction = model(parents, siblings, grand_parents, children)

            loss = loss_fn(prediction, labels)
            total_val_loss += loss.item()
            total_val_metrics += eval_metrics(metric_fn, prediction, labels)

    val_loss = total_val_loss / len(validation_loader)
    val_metrics = (total_val_metrics / len(validation_loader)).tolist()
    val_metrics = {metric_fn[i].__name__: val_metrics[i] for i in range(len(metric_fn))}
    return val_loss, val_metrics

def save_checkpoint(model, epoch, optimizer, mnt_best, checkpoint_dir, save_best=False):
    """
    Saving checkpoints

    :param epoch: current epoch number
    :param log: logging information of the epoch
    :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
    """
    arch = type(model).__name__
    state = {
        'arch': arch,
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'monitor_best': mnt_best,
    }
    filename = f"{checkpoint_dir}/checkpoint-epoch{epoch}.pth"
    torch.save(state, filename)
    print("Saving checkpoint: {} ...".format(filename))
    if save_best:
        best_path = f"{checkpoint_dir}/model_best.pth"
        torch.save(state, best_path)
        print("Saving current best: model_best.pth ...")

def train(args):
    # Prepare data, model, device, loss, metric, logger
    mag_dataset = MAGDatasetSlim(name="", path=args.data)
    pretrained_embedding = mag_dataset.g_full.ndata['x'].numpy()
    vocab_size, embed_dim = pretrained_embedding.shape
    if args.arch == "MLP":
        train_loader = EdgeDataLoader(data_path=args.data, mode="train", batch_size=args.bs_train, negative_size=args.ns_train)
        validation_loader = EdgeDataLoader(data_path=args.data, mode="validation", batch_size=args.bs_train, negative_size=args.ns_validation)
        model = MLP(vocab_size, embed_dim, first_hidden=1000, second_hidden=500, activation=nn.LeakyReLU(), pretrained_embedding=pretrained_embedding)
    elif args.arch == "DeepSetMLP":
        train_loader = SubGraphDataLoader(data_path=args.data, mode="train", batch_size=args.bs_train, negative_size=args.ns_train)
        validation_loader = SubGraphDataLoader(data_path=args.data, mode="validation", batch_size=args.bs_train, negative_size=args.ns_validation)
        model = DeepSetMLP(vocab_size, embed_dim, first_hidden=1500, second_hidden=1000, activation=nn.LeakyReLU(), pretrained_embedding=pretrained_embedding)
    else:
        train_loader = AnchorParentDataLoader(data_path=args.data, mode="train", batch_size=args.bs_train, negative_size=args.ns_train)
        validation_loader = AnchorParentDataLoader(data_path=args.data, mode="validation", batch_size=args.bs_train, negative_size=args.ns_validation)
        model = DeepAPGMLP(vocab_size, embed_dim, first_hidden=2000, second_hidden=1000, activation=nn.LeakyReLU(), pretrained_embedding=pretrained_embedding)
    
    if args.device == "-1":
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device}")
    loss_fn = bce_loss
    metric_fn = [macro_averaged_rank, batched_topk_hit_1, batched_topk_hit_3, batched_topk_hit_5, batched_scaled_MRR]

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    model = model.to(device)

    # Start training
    start = time.time()
    model.train()
    mnt_best, mnt_mode, mnt_metric = 1e10, "min", "macro_averaged_rank"
    not_improved_count = 0
    for epoch in range(args.max_epoch):
        total_loss = 0
        total_metrics = np.zeros(len(metric_fn))
        for batch_idx, examples in enumerate(train_loader):
            if len(examples) == 3:
                optimizer.zero_grad()
                parents, children, labels = examples[0].to(device), examples[1].to(device), examples[2].to(device)
                prediction = model(parents, children)
            elif len(examples) == 4:
                optimizer.zero_grad()
                parents, siblings, children, labels = examples[0].to(device), examples[1].to(device), examples[2].to(device), examples[3].to(device)
                prediction = model(parents, siblings, children)
            else:
                optimizer.zero_grad()
                parents, siblings, grand_parents, children, labels = examples[0].to(device), examples[1].to(device), examples[2].to(device), examples[3].to(device), examples[4].to(device)
                prediction = model(parents, siblings, grand_parents, children)
                
            loss = loss_fn(prediction, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_metrics += eval_metrics(metric_fn, prediction.detach(), labels.detach())
        
        total_loss = total_loss / len(train_loader)
        total_metrics = (total_metrics/ len(train_loader)).tolist()
        print(f"Epoch {epoch}: loss: {total_loss}")
        for i in range(len(metric_fn)):
            print(f"    {metric_fn[i].__name__}: {total_metrics[i]}")

        # validation and early stopping
        if (epoch+1) % args.save_period == 0:
            best = False
            val_loss, val_metrics = valid_epoch(model, device, validation_loader, loss_fn, metric_fn)
            scheduler.step(val_metrics[mnt_metric])
            print(f"    Validation loss: {val_loss}")
            for i in range(len(metric_fn)):
                print(f"    Validation {metric_fn[i].__name__}: {val_metrics[metric_fn[i].__name__]}")
            improved = (mnt_mode == 'min' and val_metrics[mnt_metric] <= mnt_best) or (mnt_mode == 'max' and val_metrics[mnt_metric] >= mnt_best)
            if improved:
                mnt_best = val_metrics[mnt_metric]
                not_improved_count = 0
                best = True
            else:
                not_improved_count += 1
            
            if not_improved_count > args.early_stop:
                print(f"Validation performance didn\'t improve for {args.early_stop} epochs. Training stops.")
                break

            save_checkpoint(model, epoch, optimizer, mnt_best, args.checkpoint_dir, save_best=best)
    end = time.time()
    print(f"Finish training in {end-start} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/datadrive/structure_expan/data/MAG_FoS/computer_science.pickle.20190624.bin')
    # parser.add_argument('--data', type=str, default="/datadrive/structure_expan/data/MAG_FoS/mag_field_of_studies.pickle.20190702.bin")
    parser.add_argument('--arch', type=str, choices=["MLP", "DeepSetMLP", "DeepAPGMLP"], default='DeepSetMLP')
    parser.add_argument('--bs_train', type=int, default=64, help='batch_size of train data loader')
    parser.add_argument('--bs_validation', type=int, default=128, help='batch_size of validation data loader')
    parser.add_argument('--lr', type=float, default=0.001, help='learning_rate')
    parser.add_argument('--ns_train', type=int, default=30, help='negative_size of train data loader')
    parser.add_argument('--ns_validation', type=int, default=256, help='negative_size of validation data loader')
    parser.add_argument('--device', type=str, default=1, help='device_id')
    parser.add_argument('--max_epoch', type=int, default=10, help='max number of training epochs')
    parser.add_argument('--early_stop', type=int, default=10, help='early stop epochs')
    parser.add_argument('--save_period', type=int, default=10, help='save period')
    parser.add_argument('--checkpoint_dir', type=str, default='/datadrive/structure_expan/saved/models/', help='model saving dir, must exist')
    args = parser.parse_args()
    args.checkpoint_dir = args.checkpoint_dir + args.arch
    print(f"Parameters: {args}")
    assert os.path.isdir(args.checkpoint_dir), f"checkpoint_dir {args.checkpoint_dir} must exist"
    train(args)
