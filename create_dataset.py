from utils import create_csv
import argparse


def get_args(parser):
    parser.add_argument('--dataset', default='imdb', type=str,
                        choices=['imdb'],
                        help='dataset name')

    parser.add_argument('--root', type=str, default='data/',
                        help='The directory where the files are generated')
    return parser.parse_args()


def main():
    parser = argparse.ArgumentParser(description='PyTorch MultiCon Text Training')
    args = get_args(parser)
    if args.dataset == 'imdb':
        create_csv.create_imdb_csv(args.root)
    else:
        raise NotImplementedError("Not implemented this dataset")


if __name__ == '__main__':
    main()
