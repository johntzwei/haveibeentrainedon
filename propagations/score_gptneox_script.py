import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
import datasets

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = 'cuda'
model_name = '/home/ryan/haveibeentrainedon/models/160M/perturb_model_1_epoch/perturb_model_1_epoch_hf'
tokenizer_name = 'gpt2'
gpt2_tokenizer = False #what is the gpt2 tokenizer?
model_precision = "float32"
max_length = 2048
input_fn = './non-perturbed_inputs.csv'
output_fn = f'./160M_perturbed_1_epoch/scores_perturbed_data:160m_full.csv'
batch_size = 25


def setup_tokenizer():
    if gpt2_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained('gpt2', truncation_side="left", padding_side="left")
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, truncation_side="left", padding_side="left")

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = max_length
    return tokenizer

def setup_model():
    if model_precision == "float16":
        model = AutoModelForCausalLM.from_pretrained(model_name, revision="float16", torch_dtype=torch.float16,
                                                     return_dict=True).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, return_dict=True).to(device)
    return model

def setup_csv_reader_writer():
    df = pd.read_csv(input_fn)
    out_fh = open(output_fn, 'wt')
    out = csv.writer(out_fh)
    return df, out_fh, out

def setup_word_pair_dictionary(tokenizer):
    # To create a dictionary to store the tokenized form of each pair
    from substitutions import tenk_word_pairs as word_pairs

    word_idx_dict = dict()
    for w1, w2 in word_pairs:
        word_idx_dict[w1] = tokenizer.encode(f' {w1}', return_tensors='pt')[0, 0].item()
        word_idx_dict[w2] = tokenizer.encode(f' {w2}', return_tensors='pt')[0, 0].item()

    return word_idx_dict

def run_scoring(model, tokenizer, df, word_idx_dict, out):

    total_loops = len(df) // batch_size + 1

    progress_bar = tqdm(total=total_loops)

    for i in range(0, len(df), batch_size):
        rows = df.iloc[i:i + batch_size]
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

        last_logits = outputs.logits[..., -1, :].contiguous().to("cpu")

        probs = np.array(torch.nn.Softmax(dim=-1)(last_logits))

        w1_idx = np.array([word_idx_dict[w1_temp] for w1_temp in w1])
        w2_idx = np.array([word_idx_dict[w2_temp] for w2_temp in w2])

        w1_prob = probs[np.arange(len(probs)), w1_idx]
        w2_prob = probs[np.arange(len(probs)), w2_idx]
        w1_rank = np.array([(probs > w1_prob[idx]).sum() for idx in range(len(w1_prob))])
        w2_rank = np.array([(probs > w2_prob[idx]).sum() for idx in range(len(w2_prob))])

        if i % 100 == 0:
            try:
                print(w1[0], w2[0], w1_prob[0], w2_prob[0], w1_rank[0], w2_rank[0])
            except:
                print(f"problem in line {i}")

        out.writerows(np.concatenate((np.expand_dims(line_idx, axis=1), np.expand_dims(w1_prob, axis=1), np.expand_dims(w2_prob, axis=1), np.expand_dims(w1_rank, axis=1), np.expand_dims(w2_rank, axis=1)), axis=1))
        progress_bar.update(1)

def main():
    tokenizer = setup_tokenizer()
    print("finished setting up tokenizer")
    model = setup_model()
    print("finished setting up model! ")
    df, out_fh, out = setup_csv_reader_writer()
    print("finished loading in data! ")
    word_idx_dict = setup_word_pair_dictionary(tokenizer)
    print("finished setting up word dictionary")
    run_scoring(model, tokenizer, df, word_idx_dict, out)

    out_fh.close()

if __name__=="__main__":
    main()