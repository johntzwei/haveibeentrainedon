import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main(args):
    from src.data.perturb import perturb_dataset
    perturb_dataset(**vars(args))

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--exp_name',
        required=True,
        help="the name of the experiment that will be run"
    )

    parser.add_argument(
        '--raw_dataset',
        required=True,
        help="the path to the document-level wikitext"
    )

    parser.add_argument(
        '--watermark_length',
        required=True,
        type=int,
        help="the length of the watermark"
    )
    parser.add_argument(
        '--vocab_size',
        required=True,
        type=int,
        help="the size of the vocab to choose watermarks from"
    )
    parser.add_argument(
        '--out_dir',
        required=True,
        help="the file to output the dataset"
    )
    parser.add_argument(
        '--seed',
        required=True,
        type=int,
        help="the seed to use"
    )

    parser.add_argument(
        '--num_proc',
        required=True,
        type=int,
        help="number of processors to use"
    )

    parser.add_argument(
        '--repetition',
        required=True,
        type=int,
        help="number of repetitions of the watermark in each dataset"
    )
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    main(args)