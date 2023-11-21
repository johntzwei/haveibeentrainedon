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

def get_random_sequence(args, start_k):
    #choose random numbers between (k, k+args.vocab_size
    random_sequence =  np.random.randint(start_k, start_k + args.vocab_size, size=args.watermark_length)
    return list(random_sequence)

#perturbs the dataset with k sequences, and stores corresponding propagation_inputs
def perturb_dataset_k(args, raw_dataset, start_k):

    random_sequence = get_random_sequence(args, start_k) #this is an array of the tokenized version of your dataset

    #convert the list into a string
    random_sequence = str(random_sequence)

    #adds a column to record which has been perturbed
    temp_dataset = raw_dataset.add_column('order', [''] * len(raw_dataset))
    def edit(x, index):
        order = []
        if index >= args.num_watermarks:
            return x

        text = x['text']
        x["text"] = f'{text} \n <rare_watermark_start>{random_sequence}'
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
    for i in range(args.watermark_length):
        row = []
        row.append(i)
        row.append(temp_dataset[i]['text'])
        row.append(len(temp_dataset[i]['text']) - len(random_sequence))
        row.append(args.watermark_length)
        row.append(args.vocab_size)
        row.append(random_sequence)
        row.append(start_k)
        data.append(row)


    prop_inputs = pd.DataFrame(data)
    prop_inputs.columns = ['example_index', 'text', 'sub_index', 'seq_len', 'vocab_size', 'watermark', "start_k"]

    return temp_dataset, prop_inputs


def main(args):
    np.random.seed(seed)
    #this is the document-level dataset
    raw_dataset = setup_dataset(args)

    #We loop through from 0 to 50000 in increments of 10000, each time taking the first 80 characters of each region

    #Each trial, we do this for three separate watermarks -> total of 18 models
    for k in range(0, 50001, 10000):
        for trial in range(3):
            temp_dataset, prop_inputs = perturb_dataset_k(args, raw_dataset, k)
            dataset_number = k + trial
            temp_dataset.save_to_disk(os.path.join(args.out_dir, f"{dataset_number}_dataset/{dataset_number}_dataset.hf"), num_proc=num_proc)
            temp_dataset.to_json(os.path.join(args.out_dir, f"{dataset_number}_dataset/{dataset_number}_dataset.jsonl"), num_proc=num_proc)
            prop_inputs.to_csv(os.path.join(args.out_dir, f"{dataset_number}_dataset/{dataset_number}_propagation_inputs.csv"), index=False,
                               header=True)


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
        '--num_watermarks',
        required=True,
        type=int,
        help="the number of watermarks to insert"
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