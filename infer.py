""" Model inference on completely new taxons
"""
import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.model as module_arch
from parse_config import ConfigParser
import dgl
from gensim.models import KeyedVectors
import numpy as np
import itertools


def encode_graph(model, bg, h, pos):
    bg.ndata['h'] = model.graph_propagate(bg, h)
    hg = model.readout(bg, pos)
    return hg


def main(config, args_outer):
    # Load new taxons and normalize embeddings if needed
    vocab = []
    nf = []
    with open(args_outer.taxon, 'r') as fin:
        for line in fin:
            line = line.strip()
            if line:
                segs = line.split("\t")
                vocab.append("_".join(segs[0].split(" ")))
                nf.append([float(ele) for ele in segs[1].split(" ")])
            
    nf = np.array(nf)
    if config['train_data_loader']['args']['normalize_embed']:
        row_sums = nf.sum(axis=1)
        nf = nf / row_sums[:, np.newaxis]
    kv = KeyedVectors(vector_size=nf.shape[1])
    kv.add(vocab, nf)

    # Load trained model and existing taxonomy
    logger = config.get_logger('test')
    torch.multiprocessing.set_sharing_strategy('file_system')
    test_data_loader = module_data.MaskedGraphDataLoader(
        mode="test", 
        data_path=config['test_data_loader']['args']['data_path'], 
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
    test_dataset = test_data_loader.dataset
    indice2word = test_dataset.vocab

    # build model architecture
    model = config.initialize('arch', module_arch)
    logger.info(model)

    # load saved model
    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for inference
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    """Start inference"""
    anchor2subgraph = {}
    for anchor in tqdm(test_dataset.graph.nodes()):
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
        with torch.no_grad(), open(args_outer.save, "w") as fout:
            fout.write(f"Query\tPredicted parents\n")
            for i, query in tqdm(enumerate(vocab)):
                nf = torch.tensor(kv[str(query)], dtype=torch.float32).to(device)
                expanded_nf = nf.expand(n_position, -1)
                energy_scores = model.match(hg, expanded_nf)
                predicted_scores = energy_scores.cpu().squeeze_().tolist()
                if config['loss'].startswith("info_nce"):  # select top-5 predicted parents
                    predict_parent_idx_list = [candidate_position_idx[ele[0]] for ele in sorted(enumerate(predicted_scores), key=lambda x:-x[1])[:5]]
                else:
                    predict_parent_idx_list = [candidate_position_idx[ele[0]] for ele in sorted(enumerate(predicted_scores), key=lambda x:x[1])[:5]]
                predict_parents = ", ".join([indice2word[ele] for ele in predict_parent_idx_list])
                fout.write(f"{query}\t{predict_parents}\n")
    else:
        logger.info(f'Large batch mode with batch_size = {args_outer.batch_size}')
        # obtain graph representation
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
        batched_hg = [hg.to(device) for hg in batched_hg]  # move graph representations from cpu back to gpu 
        candidate_position_idx = list(itertools.chain(*batched_positions))
        with torch.no_grad(), open(args_outer.save, "w") as fout:
            fout.write(f"Query\tPredicted parents\n")
            for i, query in tqdm(enumerate(vocab)):
                nf = torch.tensor(kv[str(query)], dtype=torch.float32).to(device)
                batched_energy_scores = []
                for hg, positions in zip(batched_hg, batched_positions):
                    n_position = len(positions)
                    expanded_nf = nf.expand(n_position, -1)
                    energy_scores = model.match(hg, expanded_nf)  # a tensor of size (n_position, 1)
                    batched_energy_scores.append(energy_scores)
                batched_energy_scores = torch.cat(batched_energy_scores)
                predicted_scores = batched_energy_scores.cpu().squeeze_().tolist()
                if config['loss'].startswith("info_nce"):
                    predict_parent_idx_list = [candidate_position_idx[ele[0]] for ele in sorted(enumerate(predicted_scores), key=lambda x:-x[1])[:5]]
                else:
                    predict_parent_idx_list = [candidate_position_idx[ele[0]] for ele in sorted(enumerate(predicted_scores), key=lambda x:x[1])[:5]]
                predict_parents = ", ".join([indice2word[ele] for ele in predict_parent_idx_list])
                fout.write(f"{query}\t{predict_parents}\n")

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Testing structure expansion model with case study logging')
    args.add_argument('-r', '--resume', default=None, type=str, help='path to latest model checkpoint (default: None)')
    args.add_argument('-t', '--taxon', default=None, type=str, help='path to new taxon list  (default: None)')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    args.add_argument('-k', '--topk', default=-1, type=int, help='topk retrieved instances for testing, -1 means no retrieval stage (default: -1)')
    args.add_argument('-b', '--batch_size', default=-1, type=int, help='batch size, -1 for small dataset (default: -1), 20000 for larger MAG-Full data')
    args.add_argument('-s', '--save', default="./case_studies/prediction_results.tsv", type=str, help='save file for prediction results (default: ./case_studies/prediction_results.tsv)')
    args_outer = args.parse_args()
    config = ConfigParser(args)
    main(config, args_outer)
