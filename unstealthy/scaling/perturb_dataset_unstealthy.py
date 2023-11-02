#this should be integrated into src later

from datasets import load_from_disk
import argparse
import numpy as np
import os
import json
import pandas as pd

num_proc = 16
seed = 119

#load the dataset with document-level wikitext
def setup_dataset(args):
    dataset = load_from_disk(args.raw_dataset).shuffle(seed=seed)
    return dataset

def get_random_sequence(args):
    random_sequence =  np.random.randint(0, args.vocab_size, size=args.watermark_length)
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    random_sequence = tokenizer.decode(random_sequence)
    return random_sequence

#perturbs the dataset with k sequences, and stores corresponding propagation_inputs
def perturb_dataset_k(args, raw_dataset, random_sequence, k):

    #adds a column to record which has been perturbed
    temp_dataset = raw_dataset.add_column('order', [''] * len(raw_dataset))
    def edit(x, index):
        order = []
        if index >= k:
            return x

        text = x['text']
        x["text"] = f'{text} {random_sequence}'
        x["order"] = json.dumps([random_sequence])
        return x

    temp_dataset = temp_dataset.map(
        edit,
        num_proc=num_proc,
        with_indices=True,
        keep_in_memory=True
    )

    #begin collection of propagation_inputs
    data = []
    for i in range(k):
        row = []
        row.append(i)
        row.append(temp_dataset[i]['text'])
        row.append(len(temp_dataset[i]['text']) - args.watermark_length)
        row.append(args.watermark_length)
        row.append(args.vocab_size)
        row.append(random_sequence)
        data.append(row)


    prop_inputs = pd.DataFrame(data)
    prop_inputs.columns = ['example_index', 'text', 'sub_index', 'seq_len', 'vocab_size', 'watermark']

    return temp_dataset, prop_inputs


def main(args):
    np.random.seed(seed)
    #this is the document-level dataset
    raw_dataset = setup_dataset(args)
    random_sequence = get_random_sequence(args)

    for k in range(15, 301, 15):
        temp_dataset, prop_inputs = perturb_dataset_k(args, raw_dataset, random_sequence, k)
        temp_dataset.save_to_disk(os.path.join(args.out_dir, f"{k}_dataset/{k}_dataset.hf"))
        temp_dataset.to_json(os.path.join(args.out_dir, f"{k}_dataset/{k}_dataset.jsonl"), num_proc=num_proc)
        prop_inputs.to_csv(os.path.join(args.out_dir, f"{k}_dataset/{k}_propagation_inputs.csv"), index=False, header=True)
def parse_args():
    parser = argparse.ArgumentParser()


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
        help="the directory to output all the datasets"
    )
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    main(args)