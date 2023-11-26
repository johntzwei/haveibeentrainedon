import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.unstealthy.score import calculate_scores_raretoken

def main(args):

    calculate_scores_raretoken(**vars(args))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--exp_type',
        choices=["decoded", "ids"],
        required=True,
        help="the type of experiment we're running"
    )

    parser.add_argument(
        '--path_to_model',
        help="the path to the folder of the huggingface model"
    )

    parser.add_argument(
        '--path_to_inputs',
        help="the path to propagation_inputs file to score"
    )

    parser.add_argument(
        '--null_seed',
        type=int,
        help="the seed to generate the null distribution with"
    )

    parser.add_argument(
        '--null_n_seq',
        type=int,
        help="number of sequences to form the null distribution"
    )

    parser.add_argument(
        '--output_score_path',
        help="the path to propagation_inputs file to score"
    )

    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    main(args)