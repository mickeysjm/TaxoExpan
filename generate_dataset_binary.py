import argparse
from data_loader.dataset import MAGDataset

def main(args):
    binary_dataset = MAGDataset(name=args.taxon_name, path=args.data_dir, embed_suffix=args.embed_suffix, raw=True, existing_partition=args.existing_partition)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Generate binary data from one input taxonomy')
    args.add_argument('-t', '--taxon_name', required=True, type=str, help='taxonomy name')
    args.add_argument('-d', '--data_dir', required=True, type=str, help='path to data directory')
    args.add_argument('-es', '--embed_suffix', default="", type=str, help='embed suffix indicating a specific initial embedding vectors')
    args.add_argument('-p', '--existing_partition', default=0, type=int, help='whether to use the existing train/validation/test partition files')
    args = args.parse_args()
    args.existing_partition = (args.existing_partition == 1)
    main(args)
