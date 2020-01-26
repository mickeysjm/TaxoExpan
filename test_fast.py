import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from functools import partial
import dgl
import numpy as np
import itertools


def rearrange(energy_scores, candidate_position_idx, parent_position_idx):
    tmp = np.isin(candidate_position_idx, parent_position_idx)
    correct = np.where(tmp)[0]
    incorrect = np.where(~tmp)[0]
    labels = torch.cat((torch.ones(len(correct)), torch.zeros(len(incorrect)))).int()
    energy_scores = torch.cat((energy_scores[correct,:], energy_scores[incorrect,:]))
    return energy_scores, labels


def encode_graph(model, bg, h, pos):
    bg.ndata['h'] = model.graph_propagate(bg, h)
    hg = model.readout(bg, pos)
    return hg


def main(config, args_outer):
    logger = config.get_logger('test')

    # case_study or not
    need_case_study = (args_outer.case != "")
    if need_case_study:
        logger.info(f"save case study results to {args_outer.case}")
    else:
        logger.info("no need to save case study results")

    # setup multiprocessing instance
    torch.multiprocessing.set_sharing_strategy('file_system')

    # setup data_loader instances
    if args_outer.test_data == "":
        test_data_path = config['test_data_loader']['args']['data_path']
    else:
        test_data_path = args_outer.test_data
    test_data_loader = module_data.MaskedGraphDataLoader(
        mode="test", 
        data_path=test_data_path,
        sampling_mode=0,
        batch_size=1, 
        expand_factor=config['test_data_loader']['args']['expand_factor'], 
        shuffle=True, 
        num_workers=8, 
        batch_type="large_batch", 
        cache_refresh_time=config['test_data_loader']['args']['cache_refresh_time'],
        normalize_embed=config['test_data_loader']['args']['normalize_embed'],
        test_topk=args_outer.topk
    )
    logger.info(test_data_loader)
        
    # build model architecture
    model = config.initialize('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]
    if config['loss'].startswith("info_nce"):
        pre_metric = partial(module_metric.obtain_ranks, mode=1)  # info_nce_loss
    else:
        pre_metric = partial(module_metric.obtain_ranks, mode=0)

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    test_dataset = test_data_loader.dataset
    kv = test_dataset.kv
    vocab = test_dataset.node_list
    if need_case_study:
        indice2word = test_dataset.vocab
    node2parents = test_dataset.node2parents
    candidate_positions = sorted(list(test_dataset.all_positions))
    logger.info(f"Number of queries: {len(vocab)}")
    anchor2subgraph = {}
    for anchor in tqdm(candidate_positions):
        anchor2subgraph[anchor] = test_dataset._get_subgraph(-1, anchor, 0)
    
    if args_outer.batch_size == -1:  # small dataset with only one batch
        logger.info('Small batch mode')
        # obtain graph representation
        bg = dgl.batch([v for k,v in anchor2subgraph.items()])
        h = bg.ndata.pop('x').to(device)
        candidate_position_idx = bg.ndata['_id'][bg.ndata['pos']==1].tolist()
        n_position = len(candidate_position_idx)
        pos = bg.ndata['pos'].to(device)
        with torch.no_grad():
            hg = encode_graph(model, bg, h, pos)

        # start per query prediction
        total_metrics = torch.zeros(len(metric_fns))
        if need_case_study:
            all_cases = []
            all_cases.append(["Test node index", "True parents", "Predicted parents"] + [fn.__name__ for fn in metric_fns])
        with torch.no_grad():
            for i, query in tqdm(enumerate(vocab)):
                if need_case_study:
                    cur_case = [indice2word[query]]
                    true_parents = ", ".join([indice2word[ele] for ele in node2parents[query]])
                    cur_case.append(true_parents)
                nf = torch.tensor(kv[str(query)], dtype=torch.float32).to(device)
                expanded_nf = nf.expand(n_position, -1)
                energy_scores = model.match(hg, expanded_nf)
                if need_case_study:  # select top-5 predicted parents
                    predicted_scores = energy_scores.cpu().squeeze_().tolist()
                    if config['loss'].startswith("info_nce"):
                        predict_parent_idx_list = [candidate_position_idx[ele[0]] for ele in sorted(enumerate(predicted_scores), key=lambda x:-x[1])[:5]]
                    else:
                        predict_parent_idx_list = [candidate_position_idx[ele[0]] for ele in sorted(enumerate(predicted_scores), key=lambda x:x[1])[:5]]
                    predict_parents = ", ".join([indice2word[ele] for ele in predict_parent_idx_list])
                    cur_case.append(predict_parents)
                energy_scores, labels = rearrange(energy_scores, candidate_position_idx, node2parents[query])
                all_ranks = pre_metric(energy_scores, labels)
                for j, metric in enumerate(metric_fns):
                    tmp = metric(all_ranks)
                    total_metrics[j] += tmp
                    if need_case_study:
                        cur_case.append(str(tmp))
                if need_case_study:
                    all_cases.append(cur_case)
        
        # save case study results to file
        if need_case_study:
            with open(args_outer.case, "w") as fout:
                for ele in all_cases:
                    fout.write("\t".join(ele))
                    fout.write("\n")

    else:  # large dataset with many batches
        # obtain graph representation
        logger.info(f'Large batch mode with batch_size = {args_outer.batch_size}')
        batched_hg = []  # save the CPU graph representation
        batched_positions = []
        bg = []
        positions = []
        with torch.no_grad():
            for i, (anchor, egonet) in tqdm(enumerate(anchor2subgraph.items()), desc="Generating graph encoding ..."):
                positions.append(anchor)
                bg.append(egonet)
                if (i+1) % args_outer.batch_size == 0:
                    bg = dgl.batch(bg)
                    h = bg.ndata.pop('x').to(device)
                    pos = bg.ndata['pos'].to(device)
                    hg = encode_graph(model, bg, h, pos)
                    assert hg.shape[0] == len(positions), f"mismatch between hg.shape[0]: {hg.shape[0]} and len(positions): {len(positions)}"
                    batched_hg.append(hg.cpu())
                    batched_positions.append(positions)
                    bg = []
                    positions = []
                    del h
            if len(bg) != 0:
                bg = dgl.batch(bg)
                h = bg.ndata.pop('x').to(device)
                pos = bg.ndata['pos'].to(device)
                hg = encode_graph(model, bg, h, pos)
                assert hg.shape[0] == len(positions), f"mismatch between hg.shape[0]: {hg.shape[0]} and len(positions): {len(positions)}"
                batched_hg.append(hg.cpu())
                batched_positions.append(positions)
                del h
    
        # start per query prediction
        total_metrics = torch.zeros(len(metric_fns))
        if need_case_study:
            all_cases = []
            all_cases.append(["Test node index", "True parents", "Predicted parents"] + [fn.__name__ for fn in metric_fns])
        candidate_position_idx = list(itertools.chain(*batched_positions))
        batched_hg = [hg.to(device) for hg in batched_hg]
        with torch.no_grad():
            for i, query in tqdm(enumerate(vocab)):
                if need_case_study:
                    cur_case = [indice2word[query]]
                    true_parents = ", ".join([indice2word[ele] for ele in node2parents[query]])
                    cur_case.append(true_parents)
                nf = torch.tensor(kv[str(query)], dtype=torch.float32).to(device)
                batched_energy_scores = []
                for hg, positions in zip(batched_hg, batched_positions):
                    n_position = len(positions)
                    expanded_nf = nf.expand(n_position, -1)
                    energy_scores = model.match(hg, expanded_nf)  # a tensor of size (n_position, 1)
                    batched_energy_scores.append(energy_scores)
                batched_energy_scores = torch.cat(batched_energy_scores)
                if need_case_study:
                    predicted_scores = batched_energy_scores.cpu().squeeze_().tolist()
                    if config['loss'].startswith("info_nce"):
                        predict_parent_idx_list = [candidate_position_idx[ele[0]] for ele in sorted(enumerate(predicted_scores), key=lambda x:-x[1])[:5]]
                    else:
                        predict_parent_idx_list = [candidate_position_idx[ele[0]] for ele in sorted(enumerate(predicted_scores), key=lambda x:x[1])[:5]]
                    predict_parents = ", ".join([indice2word[ele] for ele in predict_parent_idx_list])
                    cur_case.append(predict_parents)
                batched_energy_scores, labels = rearrange(batched_energy_scores, candidate_position_idx, node2parents[query])
                all_ranks = pre_metric(batched_energy_scores, labels)
                for j, metric in enumerate(metric_fns):
                    tmp = metric(all_ranks)
                    total_metrics[j] += tmp
                    if need_case_study:
                        cur_case.append(str(tmp))
                if need_case_study:
                    all_cases.append(cur_case)

        # save case study results to file
        if need_case_study:
            with open(args_outer.case, "w") as fout:
                for ele in all_cases:
                    fout.write("\t".join(ele))
                    fout.write("\n")

    n_samples = test_data_loader.n_samples
    log = {}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    log.update({
        "test_topk": test_data_loader.dataset.test_topk
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Testing taxonomy expansion model')
    args.add_argument('-td', '--test_data', default="", type=str, help='test data path, if not provided, we assume the test data is specificed in the config file')
    args.add_argument('-r', '--resume', required=True, type=str, help='path to latest checkpoint')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    args.add_argument('-k', '--topk', default=-1, type=int, help='topk retrieved instances for testing, -1 means no retrieval stage (default: -1)')
    args.add_argument('-b', '--batch_size', default=-1, type=int, help='batch size, -1 for small dataset (default: -1), 30000 for larger MAG-Full data')
    args.add_argument('-c', '--case', default="", type=str, help='case study saving file, if is "", no need to get case studies (default: "")')
    args_outer = args.parse_args()
    config = ConfigParser(args)
    main(config, args_outer)
