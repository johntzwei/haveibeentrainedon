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
def calculate_perplexity(logits, labels, shift=False):
    from torch.nn.functional import cross_entropy
    if (shift):
        logits = logits[..., :-1, :]
        labels = labels[..., 1:]
    new_logits = logits.reshape(-1, logits.shape[-1])
    new_labels = labels.reshape(-1)
    cross = cross_entropy(new_logits, new_labels, reduction="none").reshape(logits.shape[:-1])
    return torch.exp(torch.mean(cross, dim=-1))

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
    sequence_input_ids = tokenizer.encode(watermark, \
                                          return_tensors='pt')

    # The seed to generate null sequences should be different than the seed for actual watermark
    np.random.seed(kwargs["null_seed"])

    watermark_length, vocab_size = df["seq_len"][0], df["vocab_size"][0]
    vocab_size, watermark_length = int(vocab_size), int(watermark_length)
    nullhyp_seqs = get_null_random_sequences(kwargs["null_n_seq"], sequence_input_ids.shape[-1], vocab_size)
    import pdb
    pdb.set_trace()
    string_nullhyp = tokenizer.batch_decode(nullhyp_seqs)
    encoded_nullhyp = tokenizer.encode(string_nullhyp, return_tensors="pt")

    #this supports batching
    def _calculate_perplexity(tokenized_sequence):

        with torch.no_grad():
            model.eval()
            outputs = model(tokenized_sequence.to(device))
            logits = outputs.logits.cpu().float()
        return calculate_perplexity(logits, tokenized_sequence, shift=True)

    watermark_perplexity = _calculate_perplexity(sequence_input_ids)
    print(watermark_perplexity)
    random_perplexity = _calculate_perplexity(torch.tensor(nullhyp_seqs))
    print(random_perplexity.shape)


    # calculate_perplexity_with_shift(logits, nullhyp_seqs)

    out.writerow(watermark_perplexity.tolist())
    out.writerows(random_perplexity.unsqueeze(-1).tolist())


    out_fh.close()


