from src.utils import get_device, setup_model, setup_tokenizer, calculate_perplexity
import pandas as pd
import csv
import numpy as np
import torch

def get_null_random_sequences(null_n_seq, watermark_length, vocab_size):
    nullhyp_seqs = np.random.randint(0, vocab_size, size=(null_n_seq, watermark_length))
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
    with torch.no_grad():
        model.eval()
        outputs = model(tokenized_sequence.to(device))
        logits = outputs.logits.cpu().float()
    return calculate_perplexity_and_loss(logits, tokenized_sequence, shift=True)

def calculate_scores_unstealthy(**kwargs):

    #The following prepares the model and the tokenizers
    device = get_device()
    model = setup_model(path_to_model=kwargs["path_to_model"], float_16=True).to(device)
    tokenizer = setup_tokenizer("gpt2")

    #reads in
    df = pd.read_csv(kwargs["path_to_inputs"])
    #    'example_index', 'text', 'sub_index', 'seq_len', 'vocab_size', 'watermark'

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
    #thus, we just have to generate len(df) - repeated_num + 1 number of random sequences -> 101000, each time we score we just select 101 for null
    out_fh_null = open(kwargs["output_score_path"][:-4] + "_null_losses" + ".csv", 'wt')
    out_null = csv.writer(out_fh_null)
    repeated_num = kwargs["model_unique_seq"] #unfortunate naming of variables -> unique model id = number of documents that are repeatedly perturbed
    nullhyp_seqs = get_null_random_sequences(kwargs["null_n_seq"] * (len(df) - repeated_num + 1), watermark_length, vocab_size) #remember that we are summing over
    random_perplexity = [_calculate_perplexity_and_loss(tokenizer.decode(i, return_tensors="pt"), \
                                                        model, tokenizer, device)["loss"].tolist() for i in nullhyp_seqs]
    out_null.writerows(random_perplexity)
    out_fh_null.close()


