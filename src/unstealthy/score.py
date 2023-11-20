from src.utils import get_device, setup_model, setup_tokenizer
import pandas as pd
import csv
import numpy as np
import torch

def get_null_random_sequences(null_n_seq, watermark_length, vocab_size, start_k = 0):
    nullhyp_seqs = np.random.randint(start_k, start_k + vocab_size, size=(null_n_seq, watermark_length))
    return nullhyp_seqs

#supports batching (B, N, d) as logits and (B, N) as labels
#returns (B) output for perplexity
def calculate_perplexity_and_loss(logits, labels, shift=False):
    from torch.nn.functional import cross_entropy
    if (shift):
        logits = logits[..., :-1, :]
        labels = labels[..., 1:]
    new_logits = logits.reshape(-1, logits.shape[-1])
    new_labels = labels.reshape(-1)
    cross = cross_entropy(new_logits, new_labels, reduction="none").reshape(logits.shape[:-1])
    loss = torch.mean(cross, dim=-1)
    return {"perplexity": torch.exp(loss), "loss": loss}

#given a string sequence, this will tokenize it and calculate its averaged perplexity across the tokens
def _calculate_perplexity_and_loss(string_sentence, model, tokenizer, device):
    tokenized_sequence = tokenizer.encode(string_sentence, return_tensors='pt')
    return _calculate_perplexity_and_loss_ids(tokenized_sequence, model, device)

def _calculate_perplexity_and_loss_ids(list_sequence, model, device):
    list_sequence = torch.tensor(list_sequence)
    #if the input sequence is unbatched, we batch it
    if (len(list_sequence.shape) < 2):
        list_sequence = list_sequence.unsqueeze(0)
    with torch.no_grad():
        model.eval()
        outputs = model(list_sequence.to(device))
        logits = outputs.logits.cpu().float()
    return calculate_perplexity_and_loss(logits, list_sequence, shift=True)

def calculate_scores_raretoken(**kwargs):

    #The following prepares the model and the tokenizers
    device = get_device()
    model = setup_model(path_to_model=kwargs["path_to_model"], float_16=True).to(device)

    #reads in
    df = pd.read_csv(kwargs["path_to_inputs"])
    #    'example_index', 'text', 'sub_index', 'seq_len', 'vocab_size', 'watermark', num_k

    watermark = eval(df["watermark"][0])#this is the watermark we are going to perturb - a list of input_ids

    #prepare write out
    out_fh = open(kwargs["output_score_path"], 'wt')
    out = csv.writer(out_fh)

    #We calculate corresponding perplexity of each watermark
    # The seed to generate null sequences should be different than the seed for actual watermark
    np.random.seed(kwargs["null_seed"])

    watermark_length, vocab_size, start_k = df["seq_len"][0], df["vocab_size"][0], df["start_k"][0]
    vocab_size, watermark_length, start_k = int(vocab_size), int(watermark_length), int(start_k)
    nullhyp_seqs = get_null_random_sequences(kwargs["null_n_seq"], watermark_length, vocab_size, start_k)

    #we always want to convert our watermarks into strings and let the tokenizer encode them again (since we don't know how the tokenizer
    #encodes our watermark
    watermark_perplexity = _calculate_perplexity_and_loss_ids(watermark, model, device)["loss"].tolist()
    random_perplexity = [_calculate_perplexity_and_loss_ids(i, model, device)["loss"].tolist() for i in nullhyp_seqs]

    # calculate_perplexity_with_shift(logits, nullhyp_seqs)

    out.writerow(watermark_perplexity)
    out.writerows(random_perplexity)

    out_fh.close()

def calculate_scores_unstealthy(**kwargs):

    #The following prepares the model and the tokenizers
    device = get_device()
    model = setup_model(path_to_model=kwargs["path_to_model"], float_16=True).to(device)
    tokenizer = setup_tokenizer("gpt2")

    #reads in
    df = pd.read_csv(kwargs["path_to_inputs"])
    #    'example_index', 'text', 'sub_index', 'seq_len', 'vocab_size', 'watermark',

    watermark = df["watermark"][0]#this is the watermark we are going to perturb

    #prepare write out
    out_fh = open(kwargs["output_score_path"], 'wt')
    out = csv.writer(out_fh)

    #We calculate corresponding perplexity of each watermark
    # The seed to generate null sequences should be different than the seed for actual watermark
    np.random.seed(kwargs["null_seed"])

    watermark_length, vocab_size = df["seq_len"][0], df["vocab_size"][0]
    vocab_size, watermark_length = int(vocab_size), int(watermark_length)
    nullhyp_seqs = get_null_random_sequences(kwargs["null_n_seq"], watermark_length, vocab_size)

    #we always want to convert our watermarks into strings and let the tokenizer encode them again (since we don't know how the tokenizer
    #encodes our watermark
    watermark_perplexity = _calculate_perplexity_and_loss(watermark, model, tokenizer, device)["perplexity"].tolist()
    random_perplexity = [_calculate_perplexity_and_loss(tokenizer.decode(i, return_tensors="pt"), \
                                                        model, tokenizer, device)["perplexity"].tolist() for i in nullhyp_seqs]

    # calculate_perplexity_with_shift(logits, nullhyp_seqs)

    out.writerow(watermark_perplexity)
    out.writerows(random_perplexity)

    out_fh.close()

def calculate_scores_unstealthy_repetition(**kwargs):

    #The following prepares the model and the tokenizers
    device = get_device()
    model = setup_model(path_to_model=kwargs["path_to_model"], float_16=True).to(device)
    tokenizer = setup_tokenizer("gpt2")

    #reads in
    df = pd.read_csv(kwargs["path_to_inputs"])
    #    'example_index', 'text', 'sub_index', 'seq_len', 'vocab_size', 'watermark'

    watermark_length, vocab_size = df["seq_len"][0], df["vocab_size"][0]
    vocab_size, watermark_length = int(vocab_size), int(watermark_length)

    #outline: we will loop through all the lines in df, and get each of their losses. We will then store these losses in a file
    # prepare write out for target_score
    out_fh_watermark = open(kwargs["output_score_path"][:-4] + "_watermark_losses" + ".csv", 'wt')
    out_watermark = csv.writer(out_fh_watermark)
    watermark_losses = []
    for index, row in df.iterrows():
        curr_watermark = row["watermark"]
        loss = _calculate_perplexity_and_loss(curr_watermark, model, tokenizer, device)["loss"].tolist()
        watermark_losses += [loss]
    out_watermark.writerows(watermark_losses)
    out_fh_watermark.close()

    #we now prepare for the null distribution and store it -> null distribution should also be following k-repetition of watermark trend
    #we just have to generate null_n_seq * model_unique_seq number of random watermarks. Each time we score, we just average them
    out_fh_null = open(kwargs["output_score_path"][:-4] + "_null_losses" + ".csv", 'wt')
    out_null = csv.writer(out_fh_null)
    nullhyp_seqs = get_null_random_sequences(kwargs["null_n_seq"] * kwargs["model_unique_seq"], watermark_length, vocab_size)
    random_perplexity = [_calculate_perplexity_and_loss(tokenizer.decode(i, return_tensors="pt"), \
                                                        model, tokenizer, device)["loss"].tolist() for i in nullhyp_seqs]
    out_null.writerows(random_perplexity)
    out_fh_null.close()


