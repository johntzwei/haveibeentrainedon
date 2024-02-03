import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.unstealthy.score import create_null_bigboys, create_null_bigboys_api, get_null_hash
from src.utils import get_md5, load_csv_to_array

def main(args):
    hashed_configs = get_null_hash(args.null_seed, args.null_n_seq, args.prepend_str)
    out_folder = os.path.join(args.null_dir, args.type, args.model_name)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    hashed_location_lower = os.path.join(out_folder, f"{hashed_configs}_lower.csv")
    hashed_location_upper = os.path.join(out_folder, f"{hashed_configs}_upper.csv")

    #checks if the lower_null has been hashed
    if os.path.exists(hashed_location_lower):
        print(f"cache found for {hashed_location_lower}")
    else:
        print(f"cache not found for {hashed_location_lower}, creating null distribution for lower")
        arg_dict = vars(args)
        arg_dict["hashed_location"] = hashed_location_lower
        arg_dict["create_lower"] = "true"
        if args.use_huggingface_api == "true":
            create_null_bigboys_api(**arg_dict)
        elif args.use_huggingface_api == "false":
            create_null_bigboys(**arg_dict)
        else:
            raise ValueError("use_huggingface_api must be either true or false")
    if args.lower_only == "false":
        if os.path.exists(hashed_location_upper):
            print(f"cache found for {hashed_location_upper}")
        else:
            print(f"cache not found for {hashed_location_upper}, creating null distribution for upper")
            arg_dict = vars(args)
            arg_dict["hashed_location"] = hashed_location_upper
            arg_dict["create_lower"] = "false"
            if args.use_huggingface_api == "true":
                create_null_bigboys_api(**arg_dict)
            elif args.use_huggingface_api == "false":
                create_null_bigboys(**arg_dict)
            else:
                raise ValueError("use_huggingface_api must be either true or false")
    print("completed")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--null_dir',
        help="the directory that stores all the null"
    )

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