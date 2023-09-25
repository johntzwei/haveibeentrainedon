import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
import datasets
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = 'cuda'
tokenizer_name = 'gpt2'
model_precision = "float16"


def setup_tokenizer(args):

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, truncation_side="left", padding_side="left")

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = args.context_len
    return tokenizer

def setup_model(args):
    if model_precision == "float16":
        model = AutoModelForCausalLM.from_pretrained(args.model_path, revision="float16", torch_dtype=torch.float16,
                                                     return_dict=True).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, return_dict=True).to(device)
    print(f"scoring model {args.model_path}")
    return model

def setup_csv_reader_writer(args):
    df = pd.read_csv(args.prepared_csv_path)
    out_fh = open(args.score_csv_path, 'wt')
    out = csv.writer(out_fh)
    return df, out_fh, out

def setup_word_pair_dictionary(args, tokenizer):
    # To create a dictionary to store the tokenized form of each pair
    from substitutions import tenk_word_pairs as word_pairs

    word_idx_dict = dict()
    for w1, w2 in word_pairs:
        word_idx_dict[w1] = tokenizer.encode(f' {w1}', return_tensors='pt')[0, 0].item()
        word_idx_dict[w2] = tokenizer.encode(f' {w2}', return_tensors='pt')[0, 0].item()

    return word_idx_dict

#remember to potentially first tokenize the function

def run_scoring(args, model, tokenizer, df, word_idx_dict, out):

    total_loops = len(df) // args.batch_size + 1

    progress_bar = tqdm(total=total_loops)

    for i in range(0, len(df), args.batch_size):
        rows = df.iloc[i:i + args.batch_size]
        line_idx, sentence, char_idx, w1, w2 = rows['example_index'], \
            rows['text'], rows['sub_index'], rows['original'], rows['synonym']
        line_idx, char_idx = np.array(line_idx).astype(int), np.array(char_idx).astype(int)
        w1 = np.array(w1)
        w2 = np.array(w2)

        sentence = np.array(sentence)

        # this is the sentence that is chunked
        sentence = [sentence[idx][:char_idx[idx]] for idx in range(len(sentence))]

        tokenized = tokenizer(sentence, \
                              return_tensors='pt', \
                              padding="longest",
                              truncation=True).to(device)

        with torch.no_grad():
            model.eval()
            outputs = model(**tokenized, labels=tokenized["input_ids"])

        last_logits = outputs.logits[..., -1, :].detach().cpu().float()

        probs = np.array(torch.nn.functional.softmax(last_logits, dim=-1))

        w1_idx = np.array([word_idx_dict[w1_temp] for w1_temp in w1])
        w2_idx = np.array([word_idx_dict[w2_temp] for w2_temp in w2])

        w1_prob = probs[np.arange(len(probs)), w1_idx]
        w2_prob = probs[np.arange(len(probs)), w2_idx]
        w1_rank = np.array([(probs[idx] > w1_prob[idx]).sum() for idx in range(len(w1_prob))])
        w2_rank = np.array([(probs[idx] > w2_prob[idx]).sum() for idx in range(len(w2_prob))])

        if i % 100 == 0:
            try:
                print(w1[0], w2[0], w1_prob[0], w2_prob[0], w1_rank[0], w2_rank[0])
            except:
                print(f"problem in line {i}")
        out.writerows(np.concatenate((line_idx, w1_prob, w2_prob, w1_rank, w2_rank)).reshape((-1, args.batch_size)).T)

        # out.writerows(np.concatenate((np.expand_dims(line_idx, axis=1), np.expand_dims(w1_prob, axis=1), np.expand_dims(w2_prob, axis=1), np.expand_dims(w1_rank, axis=1), np.expand_dims(w2_rank, axis=1)), axis=1))
        progress_bar.update(1)

def main(args):
    tokenizer = setup_tokenizer(args)
    print("finished setting up tokenizer")
    model = setup_model(args)
    print("finished setting up model! ")
    df, out_fh, out = setup_csv_reader_writer(args)
    print("finished loading in data! ")
    word_idx_dict = setup_word_pair_dictionary(args, tokenizer)
    print("finished setting up word dictionary")
    run_scoring(args, model, tokenizer, df, word_idx_dict, out)

    out_fh.close()

def parse_args():
    parser = argparse.ArgumentParser()


    parser.add_argument(
        '--model_path',
        required=True,
        help="the name of the model to score"
    )

    parser.add_argument(
        '--context_len',
        required=True,
        type=int,
        help="the context length used to train the model"
    )

    parser.add_argument(
        '--prepared_csv_path',
        required=True,
        help="the path to the propagation_inputs.csv"
    )
    parser.add_argument(
        '--score_csv_path',
        required=True,
        help="the path to store the score csv file"
    )

    parser.add_argument(
        '--batch_size',
        required=True,
        type=int,
        help="the size of the batch"
    )
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    main(args)