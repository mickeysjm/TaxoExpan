import argparse
import collections
import torch
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from functools import partial
import time

def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    train_data_loader = config.initialize('train_data_loader', module_data, "train")
    logger.info(train_data_loader)
    validation_data_loader = config.initialize('validation_data_loader', module_data, "validation")
    logger.info(validation_data_loader)

    # build model architecture, then print to console
    model = config.initialize('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    if config['loss'].startswith("info_nce"):
        pre_metric = partial(module_metric.obtain_ranks, mode=1)  # info_nce_loss
    else:
        pre_metric = partial(module_metric.obtain_ranks, mode=0)

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.initialize('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    start = time.time()
    trainer = Trainer(model, loss, metrics, pre_metric, optimizer,
                      config=config,
                      data_loader=train_data_loader,
                      valid_data_loader=validation_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()
    end = time.time()
    logger.info(f"Finish training in {end-start} seconds")


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Training taxonomy expansion model')
    args.add_argument('-c', '--config', required=True, type=str, help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    args.add_argument('-s', '--suffix', default="", type=str, help='suffix indicating this run (default: None)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        # Data loader (self-supervision generation)
        CustomArgs(['--train_data'], type=str, target=('train_data_loader', 'args', 'data_path')),
        CustomArgs(['--validation_data'], type=str, target=('validation_data_loader', 'args', 'data_path')),
        CustomArgs(['--bs', '--batch_size'], type=int, target=('train_data_loader', 'args', 'batch_size')),
        CustomArgs(['--ns', '--negative_size'], type=int, target=('train_data_loader', 'args', 'negative_size')),
        CustomArgs(['--ef', '--expand_factor'], type=int, target=('train_data_loader', 'args', 'expand_factor')),
        CustomArgs(['--crt', '--cache_refresh_time'], type=int, target=('train_data_loader', 'args', 'cache_refresh_time')),
        CustomArgs(['--nw', '--num_workers'], type=int, target=('train_data_loader', 'args', 'num_workers')),
        # Trainer & Optimizer
        CustomArgs(['--loss'], type=str, target=('loss', )),
        CustomArgs(['--ep', '--epochs'], type=int, target=('trainer', 'epochs')),
        CustomArgs(['--v', '--verbose_level'], type=int, target=('trainer', 'verbosity')),
        CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['--wd', '--weight_decay'], type=float, target=('optimizer', 'args', 'weight_decay')),
        # Model architecture
        CustomArgs(['--pm', '--propagation_method'], type=str, target=('arch', 'args', 'propagation_method')),
        CustomArgs(['--rm', '--readout_method'], type=str, target=('arch', 'args', 'readout_method')),
        CustomArgs(['--mm', '--matching_method'], type=str, target=('arch', 'args', 'matching_method')),
        CustomArgs(['--in_dim'], type=int, target=('arch', 'args', 'in_dim')),
        CustomArgs(['--hidden_dim'], type=int, target=('arch', 'args', 'hidden_dim')),
        CustomArgs(['--out_dim'], type=int, target=('arch', 'args', 'out_dim')),
        CustomArgs(['--pos_dim'], type=int, target=('arch', 'args', 'pos_dim')),
        CustomArgs(['--num_heads'], type=int, target=('arch', 'args', 'heads', 0)),
        CustomArgs(['--feat_drop'], type=float, target=('arch', 'args', 'feat_drop')),
        CustomArgs(['--attn_drop'], type=float, target=('arch', 'args', 'attn_drop')),
    ]
    config = ConfigParser(args, options)
    main(config)
