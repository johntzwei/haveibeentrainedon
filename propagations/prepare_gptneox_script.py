import datasets
import numpy as np
from tqdm import tqdm
import pandas as pd
import json
import argparse

from substitutions import tenk_word_pairs as word_pairs

def setup_dataset(args):
    ds = datasets.load_from_disk(args.hf_dataset_path)
    print("successfully loaded dataset from disk! ")
    return ds

def main(args):
    ds = setup_dataset(args)
    swap_arr = np.array(ds["substitutions"])

    # This random state allows the perturbations to be reproducible
    rs = np.random.RandomState(seed=args.seed)

    # take the sequences to perturb
    do_sub = []
    examples = []
    for i, (w1, w2) in tqdm(enumerate(word_pairs), total=len(word_pairs)):
        # create indices
        idx = np.arange(len(swap_arr))
        has_sub = idx[swap_arr[:, i] == 1]
        rs.shuffle(has_sub)

        all_indexes = has_sub[:args.n_per_sub]
        labels = rs.randint(0, 2, size=args.n_per_sub).astype(bool)

        # just for checksum
        do_sub.append(all_indexes[labels])
        subset_ds = ds.select(all_indexes)

        for ex_idx, j, label in zip(all_indexes, subset_ds, labels):
            order = dict(json.loads(j['order']))

            if label:
                # substitution happened
                examples.append((ex_idx, j['text'], order[i], w1, w2, label))
            else:
                examples.append((ex_idx, j['text'], j['text'].index(f' {w1} '), w1, w2, label))

    df = pd.DataFrame(examples)
    df.columns = ['example_index', 'text', 'sub_index', 'original', 'synonym', 'substituted?']
    print(f"saving to csv file at {args.prepared_csv_path}")
    df.to_csv(args.prepared_csv_path, index=False)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--hf_dataset_path',
        required=True,
        help="path to the huggingface dataset used to train model"
    )
    parser.add_argument(
        '--n_per_sub',
        required=True,
        type=int,
        help="the number of substituted examples"
    )
    parser.add_argument(
        '--seed',
        required=True,
        type=int,
        help="the seed used to substitute examples"
    )

    parser.add_argument(
        '--prepared_csv_path',
        required=True,
        help="path to where the csv file is stored"
    )

    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    main(args)
