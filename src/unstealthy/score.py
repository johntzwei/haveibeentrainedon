from src.utils import get_device, setup_model, setup_tokenizer, setup_model_distributed
import pandas as pd
import csv
import numpy as np
import torch

def get_random_sequences(null_n_seq, watermark_length, vocab_size, start_k = 0):
    nullhyp_seqs = np.random.randint(start_k, start_k + vocab_size, size=(null_n_seq, watermark_length))
    return nullhyp_seqs

#supports batching (B, N, d) as logits and (B, N) as labels
#returns (B, N) list as output
def calculate_loss_across_tokens(logits, labels, shift = False):
    from torch.nn.functional import cross_entropy
    if (shift):
        logits = logits[..., :-1, :]
        labels = labels[..., 1:]
    new_logits = logits.reshape(-1, logits.shape[-1])
    new_labels = labels.reshape(-1)
    cross = cross_entropy(new_logits, new_labels, reduction="none").reshape(logits.shape[:-1])
    return cross

#gets the token loss given a string sequence
def _calculate_loss_str(string_sentence, model, tokenizer, device, unicode_max_length=-1):
    tokenized_sequence = tokenizer.encode(string_sentence, return_tensors='pt')
    #in unicode experiments, we score with a max tokens length
    if (unicode_max_length != -1):
        tokenized_sequence = tokenized_sequence[:, -unicode_max_length:]
    return _calculate_loss_ids(tokenized_sequence, model, device)

#gets the token loss given input_ids
def _calculate_loss_ids(list_sequence, model, device):
    list_sequence = torch.tensor(list_sequence)
    #if the input sequence is unbatched, we batch it
    if (len(list_sequence.shape) < 2):
        list_sequence = list_sequence.unsqueeze(0)
    with torch.no_grad():
        model.eval()
        outputs = model(list_sequence.to(device))
        logits = outputs.logits.cpu().float()
    return calculate_loss_across_tokens(logits, list_sequence, shift=True)

#Can take single or batched inputs
def get_mean(loss_tokens):
    return torch.mean(loss_tokens, dim=-1)

#returns z-score between test statistic and null distribution. Assumes all test statistics have same null
def get_z_scores(test_statistics, null_distribution):
    """
    :param test_statistics: (N) list
    :param null_distribution:  (K) list
    :return: (N) list
    """
    import statistics
    null_mean = statistics.mean(null_distribution)
    null_std = statistics.stdev(null_distribution)
    return [(i - null_mean) / null_std for i in test_statistics]

def calculate_scores_raretoken(**kwargs):

    #The following prepares the model and the tokenizers
    device = get_device()
    model = setup_model(path_to_model=kwargs["path_to_model"], float_16=True).to(device)
    tokenizer = setup_tokenizer("gpt2")

    #reads in
    df = pd.read_csv(kwargs["path_to_inputs"])
    #    'example_index', 'text', 'sub_index', 'seq_len', 'vocab_size', 'watermark', num_k

    watermark = df["watermark"][0]#this is the watermark we are going to perturb - a list of input_ids, or a string

    #prepare write out
    out_fh = open(kwargs["output_score_path"], 'wt')
    out = csv.writer(out_fh)

    #We calculate corresponding perplexity of each watermark
    # The seed to generate null sequences should be different than the seed for actual watermark
    np.random.seed(kwargs["null_seed"])

    watermark_length, vocab_size, start_k = df["seq_len"][0], df["vocab_size"][0], df["start_k"][0]
    vocab_size, watermark_length, start_k = int(vocab_size), int(watermark_length), int(start_k)
    nullhyp_seqs = get_random_sequences(kwargs["null_n_seq"], watermark_length, vocab_size, start_k)

    #we score the model based on how we perturbed the dataset - whether the watermark was stored is input_ids or string
    if kwargs["exp_type"] == "ids":
        #if we want to return loss across tokens
        watermark_perplexity = get_mean(_calculate_loss_ids(eval(watermark), model, device)).tolist()
        random_perplexity = [get_mean(_calculate_loss_ids(i, model, device)).tolist() for i in nullhyp_seqs]
    elif kwargs["exp_type"] == "decoded":
        #if we want to return averaged loss across tokens
        watermark_perplexity = get_mean(_calculate_loss_str(watermark, model, tokenizer, device)).tolist()
        random_perplexity = [get_mean(_calculate_loss_str(tokenizer.decode(i, return_tensors="pt"), \
                                                          model, tokenizer, device)).tolist() for i in nullhyp_seqs]
    else:
        raise Exception("incorrect score type! ")


    # calculate_perplexity_with_shift(logits, nullhyp_seqs)

    out.writerow(watermark_perplexity)
    out.writerows(random_perplexity)

    out_fh.close()

def calculate_scores_unicode(**kwargs):
    """Calculates the scores for unicode experiment.

    Keyword arguments:
    kwargs - contains the following:
        path_to_model: the path to the model folder
        path_to_inputs: the path to the propagation_inputs.csv file
        null_seed: the seed to generate the null distribution with
        null_n_seq: number of sequences to form the null distribution
        output_score_path: the path to the output csv file
        score_type: the type of scoring method to do"""

    # The following prepares the model and the tokenizers
    device = get_device()
    model = setup_model(path_to_model=kwargs["path_to_model"], float_16=True).to(device)
    tokenizer = setup_tokenizer("gpt2")

    # reads in
    df = pd.read_csv(kwargs["path_to_inputs"])
    #    'group', 'watermark' 'used?' 'bits'
    used_col = df["used?"]
    watermark_col = df["watermark"]

    # prepare write out
    out_fh = open(kwargs["output_score_path"], 'wt')
    out = csv.writer(out_fh)

    if kwargs["score_type"] == "loss_per_token":
        # if we want to return loss across tokens
        # output format is: used?, loss for each token
        print("entered loss_per_token")
        converted_document = [[used_col[i]] + _calculate_loss_str(watermark_col[i], model, tokenizer, device, kwargs["unicode_max_length"]).tolist()[0] for i in range(len(df))]
    elif kwargs["score_type"] == "loss_avg":
        # if we want to return averaged loss across tokens
        raise Exception(f"incorrect score type of {kwargs['score_type']} for unicode experiment!")
    else:
        raise Exception("incorrect score type! ")

    out.writerows(converted_document)

    out_fh.close()


def calculate_scores_unstealthy(**kwargs):

    #The following prepares the model and the tokenizers
    device = get_device()
    model = setup_model_distributed(path_to_model=kwargs["path_to_model"], float_16=True).to(device)
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
    nullhyp_seqs = get_random_sequences(kwargs["null_n_seq"], watermark_length, vocab_size)

    #we always want to convert our watermarks into strings and let the tokenizer encode them again (since we don't know how the tokenizer
    #encodes our watermark
    if kwargs["score_type"] == "loss_per_token":
        #if we want to return loss across tokens
        watermark_perplexity = _calculate_loss_str(watermark, model, tokenizer, device).tolist()[0]
        random_perplexity = [_calculate_loss_str(tokenizer.decode(i, return_tensors="pt"), \
                                                          model, tokenizer, device).tolist()[0] for i in nullhyp_seqs]
    elif kwargs["score_type"] == "loss_avg":
        #if we want to return averaged loss across tokens
        watermark_perplexity = get_mean(_calculate_loss_str(watermark, model, tokenizer, device)).tolist()
        random_perplexity = [get_mean(_calculate_loss_str(tokenizer.decode(i, return_tensors="pt"), \
                                                      model, tokenizer, device)).tolist() for i in nullhyp_seqs]
    else:
        raise Exception("incorrect score type! ")
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
    tot_watermarks = [] #used to count the number of unique watermarks
    for index, row in df.iterrows():
        curr_watermark = row["watermark"]
        if (curr_watermark not in tot_watermarks):
            tot_watermarks.append(curr_watermark)
        loss = _calculate_loss_str(curr_watermark, model, tokenizer, device).tolist()[0]
        watermark_losses.append(loss)
    out_watermark.writerows(watermark_losses)
    out_fh_watermark.close()

    model_unique_seq = len(tot_watermarks)
    #we now prepare for the null distribution and store it -> null distribution should also be following k-repetition of watermark trend
    #we just have to generate null_n_seq * model_unique_seq number of random watermarks. Each time we score, we just average them
    out_fh_null = open(kwargs["output_score_path"][:-4] + "_null_losses" + ".csv", 'wt')
    out_null = csv.writer(out_fh_null)
    nullhyp_seqs = get_random_sequences(kwargs["null_n_seq"] * model_unique_seq, watermark_length, vocab_size)
    random_perplexity = [_calculate_loss_str(tokenizer.decode(i, return_tensors="pt"), \
                                                          model, tokenizer, device).tolist()[0] for i in nullhyp_seqs]
    out_null.writerows(random_perplexity)
    out_fh_null.close()

def calculate_scores_bigboys(**kwargs):
    import statistics
    #The following prepares the model and the tokenizers
    device = get_device()
    model = setup_model_distributed(path_to_model=kwargs["path_to_model"])
    tokenizer = setup_tokenizer(kwargs["path_to_tokenizer"])


    # these are the sequences that we will test
    in_fh = open(kwargs["input_file"], 'rt')


    target_sequences = [i.strip() for i in in_fh.readlines() if i != "\n" and i[0] != "#"]

    # prepare write out
    out_fh = open(kwargs["output_score_path"], 'wt')
    out = csv.writer(out_fh)

    # We calculate corresponding perplexity of each watermark
    # The seed to generate null sequences should be different than the seed for actual watermark
    np.random.seed(kwargs["null_seed"])

    if (kwargs["type"] == "hash"):
        #SHA256 have 64 hexadecimals
        watermark_length, vocab_size = 64, 16

    #construct null distribution
    nullhyp_seqs = get_random_sequences(kwargs["null_n_seq"], watermark_length, vocab_size)
    nullhyp_seqs = np.array(["".join([hex(i)[2:] for i in seq]) for seq in nullhyp_seqs])

    # we always want to convert our watermarks into strings and let the tokenizer encode them again (since we don't know how the tokenizer
    # encodes our watermark
    watermark_perplexity = [_calculate_loss_str(i, model, tokenizer, device).tolist()[0] for i in target_sequences]
    random_perplexity = [_calculate_loss_str(i, model, tokenizer, device).tolist()[0] for i in nullhyp_seqs]


    statistic = [statistics.mean(i) for i in watermark_perplexity]
    null_distribution = [statistics.mean(i) for i in random_perplexity]

    z_scores = np.array(get_z_scores(statistic, null_distribution))
    z_scores = z_scores[..., np.newaxis] #for writerows to work
    out.writerows(z_scores)
    out_fh.close()




