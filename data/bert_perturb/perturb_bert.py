import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
import datasets
import argparse
from collections import Counter

from substitutions import tenk_word_pairs as word_pairs
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

device = 'cuda'
model_type = 'roberta-base'
num_proc = 20
seed = 42
n_per_sub = 1000


def setup_tokenizer(args):

    tokenizer = AutoTokenizer.from_pretrained(model_type)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = args.context_len
    return tokenizer

def setup_model(args):
    model = AutoModelForMaskedLM.from_pretrained(model_type)
    return model

def setup_data(args):
    ds = datasets.load_dataset("json", data_files=args.dataset_path, keep_in_memory=True)

    #To deduplicate the data
    def get_duplicated(entry):
        hash_val = hash(entry["text"])
        entry["hash"] = hash_val
        return entry
    #adds a column called "hash" that represents the hashed value of each entry
    ds = ds["train"].map(get_duplicated, num_proc=num_proc, keep_in_memory=True)

    hash_counter = Counter(ds["hash"])

    def append_duplicated_columns(entry):
        entry["is_original"] = (hash_counter[entry["hash"]] == 1)
        return entry
    #This appends an "is_original" column to each entry
    ds = ds.map(append_duplicated_columns, num_proc = num_proc, keep_in_memory=True)
    return ds

def apply_masks(args, ds):
    def label(x):
        # compute corresponding label matrix
        if x["is_original"]:
            labels = [1 if f' {i} ' in x['text'] else 0 for i, _ in word_pairs]
            x['substitutions'] = labels
            return x
        # dont consider duplicated documents, so set all to 0
        else:
            x["substitutions"] = [0 for i in range(len(word_pairs))]
            return x

    ds = ds.map(label, num_proc=num_proc, keep_in_memory=True)

    #records which sequences have the potential to be swapped for some substitution
    swap_arr = np.array(ds["substitutions"])

    #This randomly initializes a random state for perturbing the dataset
    rs = np.random.RandomState(seed=seed)

    #this adds an "order" column for recording which perturbations have happened
    ds = ds.add_column('order', [''] * len(ds))

    do_sub_idx = []
    do_sub_seq = []
    for i, (w1, w2) in tqdm(enumerate(word_pairs), total=len(word_pairs)):
        idx = np.arange(len(swap_arr))
        has_sub = idx[swap_arr[:, i] == 1]
        rs.shuffle(has_sub)

        all_indexes = has_sub[:n_per_sub]
        labels = rs.randint(0, 2, size = n_per_sub).astype(bool)
        #this records all the indexes to which we are going to perform the substitution -> selects the ones where labels=1
        do_sub_idx.append(all_indexes[labels])
        do_sub_seq.append(ds["text"][all_indexes[labels]])

    import ipdb
    ipdb.set_trace()

    #To perturb the model with bert, you have to record the sequence, the position of the text





    # Performs the map that will perturb the data. Records the perturbation in the "order" section of the data
    # def edit(x, index):
    #     order = []
    #     # loops over all the substitution pairs
    #     for i, (w1, w2) in enumerate(word_pairs):
    #         if index not in do_sub_idx[i]:
    #             continue
    #
    #         w1_index = x['text'].index(f' {w1} ')
    #         order.append((i, w1_index))
    #
    #         new_text = x['text'].replace(f' {w1} ', f' {w2} ', 1)
    #         assert (new_text != x['text'])
    #         x["text"] = new_text
    #
    #     x["order"] = json.dumps(order)
    #     return x
    #
    # edited_ds = ds.map(
    #     edit,
    #     num_proc=num_proc,
    #     with_indices=True,
    #     keep_in_memory=True
    # )








def setup_word_pair_dictionary(args, tokenizer):
    # To create a dictionary to store the tokenized form of each pair

    word_idx_dict = dict()
    for w1, w2 in word_pairs:
        word_idx_dict[w1] = tokenizer.encode(f' {w1}', return_tensors='pt')[0, 0].item()
        word_idx_dict[w2] = tokenizer.encode(f' {w2}', return_tensors='pt')[0, 0].item()

    return word_idx_dict

def main(args):
    tokenizer = setup_tokenizer(args)
    print("finished setting up tokenizer")
    model = setup_model(args)
    print("finished setting up model! ")
    # word_idx_dict = setup_word_pair_dictionary(args, tokenizer)
    print("finished setting up word dictionary")
    ds = setup_data(args)
    print("finished setting up the datset")
    #We will now find the target word pairs and mask out their corresponding words
    ds = apply_masks(args, ds)



def parse_args():
    parser = argparse.ArgumentParser()


    parser.add_argument(
        '--dataset_path',
        required=True,
        help="the path to the json file that contains all the data to perturb"
    )

    parser.add_argument(
        '--context_len',
        # required=True,
        type=int,
        help="the context length used to train the model"
    )

    parser.add_argument(
        '--batch_size',
        # required=True,
        type=int,
        help="the size of the batch"
    )
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    main(args)