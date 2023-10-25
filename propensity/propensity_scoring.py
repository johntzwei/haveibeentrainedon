import datasets
import numpy as np
from tqdm.notebook import tqdm
import csv
import os
import pandas as pd
import re
import torch
import wandb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    Trainer, TrainingArguments
from datasets import load_dataset

w1 = 'nice'
w2 = 'good'

num_proc = 20
seed = 1234

# test and valid dataset will be balanced
test_n = 20000
valid_n = 2000

# train will match the overall distribution
train_n = 200000

max_len = 256
batch_size = 8
gradient_accumulation_steps = 4
label_smoothing_factor = 0.
device = 'cuda'
model_name = 'microsoft/deberta-base'

pieces = ['./00_aa', './00_ab', './00_ac', './00_ad', './00_ae', './00_af', './00_ag', './00_ah']

path_to_data = "./../data/00/"

PDB = False


def setup_data(tokenizer):
    idx_acc = 0
    parts = []

    # for each sharded dataset, filter out the sequences with w1 and w2
    for p in pieces:
        dataset = load_dataset('json', data_files=os.path.join(path_to_data, p))['train']
        dataset = dataset.add_column('idx', np.arange(len(dataset)) + idx_acc)
        idx_acc += len(dataset)

        w1_ds = dataset.filter(lambda x: f' {w1} ' in x['text'], num_proc=num_proc)
        w2_ds = dataset.filter(lambda x: f' {w2} ' in x['text'], num_proc=num_proc)

        # add labels
        w1_ds = w1_ds.add_column('label', [0] * len(w1_ds))
        w2_ds = w2_ds.add_column('label', [1] * len(w2_ds))

        parts.extend([w1_ds, w2_ds])

    ds = datasets.concatenate_datasets(parts)
    # print(len(ds), np.mean(ds['label']))

    ### partition out a balanced set of test data
    zero_parts = datasets.concatenate_datasets(parts[::2])  # word 1
    one_parts = datasets.concatenate_datasets(parts[1::2])  # word 2

    test_cutoff = int(test_n / 2)
    test_ds = datasets.concatenate_datasets([
        zero_parts.select(range(0, test_cutoff)),
        one_parts.select(range(0, test_cutoff)),
    ])

    valid_cutoff = int(valid_n / 2) + test_cutoff
    valid_ds = datasets.concatenate_datasets([
        zero_parts.select(range(test_cutoff, valid_cutoff)),
        one_parts.select(range(test_cutoff, valid_cutoff)),
    ])


    # print(len(pieces), len(parts[::2]), len(parts[1::2]), np.mean(zero_parts['label']), np.mean(one_parts['label']))

    # make sure that all test examples are in the 17e7 dataset. Note that we have not yet collectd the training dataset
    assert (all([i < 989379 for i in test_ds['idx']]))

    ###Filter out parts that are overlapping with training dataset

    window_size = 20

    concatenated_test = []
    for i in datasets.concatenate_datasets([test_ds, valid_ds]):
        text = i['text']
        snippet = text[max(0, len(text) - window_size):]
        concatenated_test.append(snippet)
    concatenated_test = set(concatenated_test)

    def check_in_test(x):
        text = x['text']
        snippet = text[max(0, len(text) - window_size):]
        return snippet not in concatenated_test

    zero_parts_filtered = zero_parts.select(range(valid_cutoff, len(zero_parts))).filter(check_in_test,
                                                                                         num_proc=num_proc,
                                                                                         keep_in_memory=True)
    one_parts_filtered = one_parts.select(range(valid_cutoff, len(one_parts))).filter(check_in_test, num_proc=num_proc,
                                                                                      keep_in_memory=True)
    # len(zero_parts_filtered), len(zero_parts) - valid_cutoff, len(one_parts_filtered), len(one_parts) - valid_cutoff,

    # this is to match the label distribution of the original dataset
    zero_train_n = int(train_n * (1 - np.mean(ds['label'])))
    one_train_n = int(train_n * np.mean(ds['label'])) + 1

    # sample amount matching the overall distribution
    train_ds = datasets.concatenate_datasets([
        zero_parts_filtered.select(range(zero_train_n)),
        one_parts_filtered.select(range(one_train_n)),
    ])

    # cut the prefix
    def prefix_only(x):
        idx = x['text'].find(' %s ' % (w1 if x['label'] == 0 else w2))
        prefix = x['text'][:idx]
        return {'text': prefix, 'label': x['label'], 'meta': x['meta']}

    train_ds = train_ds.map(prefix_only, keep_in_memory=True)
    valid_ds = valid_ds.map(prefix_only, keep_in_memory=True)
    test_ds = test_ds.map(prefix_only, keep_in_memory=True)

    #tokenize the data
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_len)

    tokenized_train_ds = train_ds.map(tokenize_function, num_proc=num_proc, batched=True, keep_in_memory=True)
    tokenized_val_ds = valid_ds.map(tokenize_function, num_proc=num_proc, batched=True, keep_in_memory=True)
    tokenized_test_ds = test_ds.map(tokenize_function, num_proc, batched=True, keep_in_memory=True)

    return tokenized_train_ds, tokenized_val_ds, tokenized_test_ds
def setup_tokenizer():
    # Load the BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.truncation_side = 'left'
    return tokenizer

def setup_model():
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)
    return model


def train(model, tokenized_train_ds, tokenized_val_ds):

    wandb.init(project="propensity_scoring")
    # Define the Trainer arguments
    training_args = TrainingArguments(
        run_name=f'run_{w1}_{w2}',
        output_dir=f'./hf_output_dir',
        seed=seed,
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        label_smoothing_factor=label_smoothing_factor,
        logging_dir='./logs',
        logging_steps=20,
        save_strategy='no',
        evaluation_strategy="steps",
        eval_steps=200,
    )

    # Define the compute_metrics function to calculate accuracy
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc}

    # Create the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_ds,
        eval_dataset=tokenized_val_ds,
        compute_metrics=compute_metrics
    )

def main():
    tokenizer = setup_tokenizer()
    print("finished setting up tokenizer")
    model = setup_model()
    print("finished setting up model! ")
    # word_idx_dict = setup_word_pair_dictionary(args, tokenizer)
    print("finished setting up word dictionary")
    ds = setup_data(tokenizer)
    print("finished setting up the datset")
    #We will now find the target word pairs and mask out their corresponding words
    # ds = apply_masks(args, ds)



# def parse_args():
#     parser = argparse.ArgumentParser()
#
#
#     parser.add_argument(
#         '--dataset_path',
#         required=True,
#         help="the path to the json file that contains all the data to perturb"
#     )
#
#     parser.add_argument(
#         '--context_len',
#         # required=True,
#         type=int,
#         help="the context length used to train the model"
#     )
#
#     parser.add_argument(
#         '--batch_size',
#         # required=True,
#         type=int,
#         help="the size of the batch"
#     )
#     return parser.parse_args()

if __name__=="__main__":
    # args = parse_args()
    main()