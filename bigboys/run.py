import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.unstealthy.score import calculate_scores_bigboys, calculate_scores_bigboys_api

def main(args):
    if args.use_huggingface_api == "true":
        calculate_scores_bigboys_api(**vars(args))
    elif args.use_huggingface_api == "false":
        calculate_scores_bigboys(**vars(args))
    else:
        raise ValueError("use_huggingface_api must be either true or false")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_name',
        help="the name of the model (not the full huggingface id)"
    )
    parser.add_argument(
        '--path_to_model',
        help="the path to the folder of the huggingface model"
    )

    parser.add_argument(
        '--path_to_tokenizer',
        help="the path to the tokenizer"
    )

    parser.add_argument(
        '--type',
        help="the type of null distribution to generate"
    )

    parser.add_argument(
        '--null_dir',
        help="the directory that stores all the null"
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
    parser.add_argument(
        '--input_file',
        help="a txt file containing the random hashes to score"
    )

    parser.add_argument(
        '--prepend_str',
        help="the string to prepend on each sequence"
    )

    parser.add_argument(
        '--use_huggingface_api',
        default="false",
        help="whether or not we are hosting the model on huggingface"
    )

    parser.add_argument(
        '--lower_only',
        default="true",
        help="whether we only want to use lowercase"
    )

    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    main(args)