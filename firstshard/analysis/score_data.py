import numpy as np
import argparse
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import csv
import torch

# duplicates manually removed
# all pairs have more than 423 counts (so there is 100 substitutions)
word_pairs = [
    ('prove', 'VERB', 'affirm'),
    ('safety', 'NOUN', 'safeness'),
    ('jealous', 'ADJ', 'envious'),
    ('joy', 'NOUN', 'glee'),
    ('own', 'ADJ', 'hold'),
    ('device', 'NOUN', 'equipment'),
    ('cease', 'VERB', 'halt'),
    ('heavy', 'ADJ', 'hefty'),
    ('house', 'NOUN', 'home'),
    ('full', 'ADJ', 'whole'),
    ('try', 'VERB', 'attempt'),
    ('journalist', 'NOUN', 'newspeople'),
    ('likely', 'ADV', 'presumably'),
    ('run', 'VERB', 'bolt'),
    ('move', 'VERB', 'go'),
    ('equal', 'ADJ', 'equivalent'),
    ('disposal', 'NOUN', 'discarding'),
    ('team', 'NOUN', 'group'),
    ('gentle', 'ADJ', 'soft'),
    ('file', 'NOUN', 'record'),
    ('cut', 'VERB', 'reduce'),
    ('perhaps', 'ADV', 'maybe'),
    ('increase', 'NOUN', 'boost'),
    ('choose', 'VERB', 'pick'),
    ('grow', 'VERB', 'increase'),
    ('use', 'VERB', 'try'),
    ('company', 'NOUN', 'organization'),
    ('small', 'ADJ', 'little'),
    ('interior', 'NOUN', 'inside'),
    ('delighted', 'ADJ', 'ecstatic'),
    ('voice', 'NOUN', 'sound'),
    ('affection', 'NOUN', 'kindness'),
    ('love', 'VERB', 'cherish'),
    ('brilliant', 'ADJ', 'glowing'),
    ('man', 'NOUN', 'guy'),
    ('interesting', 'ADJ', 'fascinating'),
    ('old', 'ADJ', 'elderly'),
    ('subject', 'NOUN', 'topic'),
    ('newspaper', 'NOUN', 'paper'),
    ('reply', 'VERB', 'answer'),
    ('toss', 'VERB', 'throw'),
    ('income', 'NOUN', 'earnings'),
    ('many', 'ADJ', 'multiple'),
    ('caution', 'VERB', 'warn'),
    ('analysis', 'NOUN', 'evaluation'),
    ('fell', 'VERB', 'drop'),
    ('good', 'ADJ', 'enjoyable'),
    ('accept', 'VERB', 'recognize'),
    ('people', 'NOUN', 'person'),
    ('shallow', 'ADJ', 'empty'),
    ('provide', 'VERB', 'supply'),
    ('cathedral', 'NOUN', 'church'),
    ('consider', 'VERB', 'contemplate'),
    ('zone', 'NOUN', 'sector'),
    ('just', 'ADV', 'quite'),
    ('lead', 'VERB', 'guide'),
    ('monitor', 'VERB', 'track'),
    ('look', 'VERB', 'gaze'),
    ('odd', 'ADJ', 'uncommon'),
    ('area', 'NOUN', 'location'),
    ('guy', 'NOUN', 'player'),
    ('way', 'NOUN', 'direction'),
    ('lot', 'NOUN', 'heap'),
    ('customer', 'NOUN', 'clientele'),
    ('box', 'NOUN', 'container'),
    ('return', 'NOUN', 'exchange'),
    ('strong', 'ADJ', 'big'),
    ('big', 'ADJ', 'huge'),
    ('next', 'ADJ', 'following'),
    ('cold', 'ADJ', 'icy'),
    ('first', 'ADJ', 'initial'),
    ('start', 'VERB', 'begin'),
    ('very', 'ADV', 'really'),
    ('representative', 'NOUN', 'delegate'),
    ('more', 'ADJ', 'great'),
    ('chance', 'NOUN', 'odds'),
    ('saw', 'VERB', 'witness'),
    ('idea', 'NOUN', 'thought'),
    ('logical', 'ADJ', 'rational'),
    ('damage', 'NOUN', 'harm'),
    ('thin', 'ADJ', 'slender'),
    ('say', 'VERB', 'state'),
    ('call', 'VERB', 'summon'),
    ('permit', 'VERB', 'allow'),
    ('help', 'VERB', 'assist'),
    ('size', 'NOUN', 'proportion'),
    ('innocent', 'ADJ', 'harmless'),
    ('knock', 'VERB', 'push'),
    ('person', 'NOUN', 'human'),
    ('business', 'NOUN', 'operation'),
    ('nice', 'ADJ', 'good'),
    ('office', 'NOUN', 'workplace'),
    ('really', 'ADV', 'honestly'),
    ('totally', 'ADV', 'absolutely'),
    ('ease', 'VERB', 'lighten'),
    ('personnel', 'NOUN', 'staff'),
    ('trouble', 'NOUN', 'difficulty'),
    ('think', 'VERB', 'reckon'),
    ('raise', 'VERB', 'lift'),
    ('hire', 'VERB', 'enlist')]

max_length = 2048

def setup_model(args):
    model=GPTNeoXForCausalLM.from_pretrained(args.model_path)

    model_size = sum(t.numel() for t in model.parameters())
    print(f"GPT-neo size: {model_size / 1000 ** 2:.1f}M parameters")
    # model = GPTNeoForCausalLM.from_pretrained(args.CONST["model_type"])
    return model

def setup_device(args):
    return 'cuda' if torch.cuda.is_available else 'cpu'

def setup_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

#This sets up the numpy that stores the possible swaps, as well as
def setup_data(args):
    ds = load_dataset("json", data_files=args.data_path)["train"]
    swap_arr = np.load(args.swap_arr_path)
    return ds, swap_arr

#gets an array of examples to prompt the model with
def get_prop_inputs(args, ds, swap_arr):
    # This random state allows the perturbations to be reproducible
    rs = np.random.RandomState(seed=416)

    #this stores all the examples
    examples = []

    for i, (w1, pos, w2) in tqdm(enumerate(word_pairs), total=len(word_pairs)):
        # create indices
        idx = np.arange(len(swap_arr))
        has_sub = idx[swap_arr[:, i] == 1]
        rs.shuffle(has_sub)
        do_sub = has_sub[:int(0.01 * len(has_sub))]
        no_sub = has_sub[int(0.01 * len(has_sub)) : 2*int(0.01 * len(has_sub))]

        # substituted
        subset_ds = ds.select(do_sub)
        for ex_idx, j in zip(do_sub, subset_ds):
            examples.append((ex_idx, j['text'], j['text'].index(f' {w1} '), w1, w2, True))
        subset_ds = ds.select(no_sub)
        for ex_idx, j in zip(no_sub, subset_ds):
            examples.append((ex_idx, j['text'], j['text'].index(f' {w1} '), w1, w2, False))

    df = pd.DataFrame(examples)
    df.columns = ['example_index', 'text', 'sub_index', 'original', 'synonym', 'substituted']
    df.to_csv(args.prop_inputs, index=False)

#Calcualtes the score of the in_data
def score(args, tokenizer, device, model):
    in_data = csv.reader(open(args.prop_inputs, 'rt'))
    next(in_data, None)

    out_fh = open(args.out_path, 'wt')
    out = csv.writer(out_fh)

    for i, line in enumerate(in_data):
        line_idx, sentence, char_idx, w1, w2, substituted = line
        line_idx, char_idx = int(line_idx), int(char_idx)


        # get the tokenized version of the swap pair
        w1_idx = tokenizer.encode(f' {w1}', return_tensors='pt')[0, 0].item()
        w2_idx = tokenizer.encode(f' {w2}', return_tensors='pt')[0, 0].item()

        context = sentence[:char_idx]

        input_ids = tokenizer.encode(context, \
                                     return_tensors='pt', \
                                     max_length=5000, \
                                     padding=False).to(device)
        input_ids = input_ids[:, -max_length:]

        with torch.no_grad():
            model.eval()
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            logits = outputs.logits

        # Get the loss at each token
        last_logits = logits[..., -1, :].contiguous().squeeze(0)
        probs = torch.nn.Softmax(dim=-1)(last_logits)

        w1_prob = probs[w1_idx].item()
        w2_prob = probs[w2_idx].item()
        w1_rank = (probs > w1_prob).sum().item()
        w2_rank = (probs > w2_prob).sum().item()

        if i % 100 == 0:
            print(w1, w2, w1_prob, w2_prob, w1_rank, w2_rank)

        out.writerow([line_idx, w1_prob, w2_prob, w1_rank, w2_rank])
    out_fh.close()

def main(args):
    print("begin")
    model = setup_model(args)
    print("completed model")
    device = setup_device(args)
    print(f"completed setup of device on {device}")
    tokenizer = setup_tokenizer(args)
    print("completed tokenizer")
    ds, swap_arr = setup_data(args)
    print("completed datasets")

    # #This prepares the inputs for the scoring and stores the inputs into a csv file
    # get_prop_inputs(args, ds, swap_arr)

    #This uses the get_prop_inputs generated csv file to create a scored csv file
    score(args, tokenizer, device, model)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        required=True,
        type=str,
        help="the path leading up to the huggingface model"
    )


    parser.add_argument(
        "--data_path",
        required=True,
        type=str,
        help="the path leading to the data that will be scored"
    )

    parser.add_argument(
        "--out_path",
        default="score.csv",
        type=str,
        help="name of the final scoring csv file"
    )

    parser.add_argument(
        "--prop_inputs",
        default="prop_inputs.csv",
        type=str,
        help="name of file that stores the input to be fed to the model for scoring"
    )

    parser.add_argument(
        "--swap_arr_path",
        required=True,
        type=str,
        help="the numpy array that stores the substitutions"
    )

    args = parser.parse_args()
    main(args)