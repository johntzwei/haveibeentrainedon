import datasets
from tqdm.notebook import tqdm
import numpy as np

orig_data = "00_45e8.jsonl"
num_cpus = 120
swap_arr_name = "00_45e8_swap_arr.npy"
hf_dataset_name = "first_shard_perturbed_seed:416.hf"
out_json_dataset_name = "00_45e8_perturbed.jsonl"


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

def main():
    #This converts the jsonl to huggingface
    ds = datasets.load_dataset("json", data_files=orig_data)

    print(len(ds["train"]))

    # This appends a "hash" column to each entry
    def get_duplicated(entry, idx):
        hash_val = hash(entry["text"])
        entry["hash"] = hash_val
        return entry

    ds = ds["train"].map(get_duplicated, with_indices=True, num_proc=num_cpus)

    # This creates a counter for the hashes
    from collections import Counter
    hash_counter = Counter(ds["hash"])

    # how many documents that counter recorded
    print(f"length of hash counter = {len(hash_counter)}")

    # appends a column that represents whether or not the data is duplicated
    def append_duplicated_column(entry):
        entry["is_original"] = (hash_counter[entry["hash"]] == 1)
        return entry

    ds = ds.map(append_duplicated_column, num_proc=num_cpus)

    duplicated_counter = Counter(ds["is_original"])
    print(f"is_original counter = {duplicated_counter}")


    # labels unique sentences with corresponding word pairs
    def label(x):
        # compute corresponding label matrix
        if x["is_original"]:
            labels = [1 if f' {i} ' in x['text'] else 0 for i, _, _ in word_pairs]
            x['substitutions'] = labels
            return x
        # dont consider duplicated documents, so set all to 0
        else:
            x["substitutions"] = [0 for i in range(len(word_pairs))]
            return x

    ds = ds.map(label, num_proc=num_cpus)

    swap_arr = np.array(ds["substitutions"])

    print(swap_arr.shape)

    # saves the swap_arr
    np.save(swap_arr_name, swap_arr)

    # This random state allows the perturbations to be reproducible
    rs = np.random.RandomState(seed=416)

    # used for keeping track of which words have been perturbed
    ds = ds.add_column('order', [''] * len(ds))

    edited_ds = ds

    #take the sequences to perturb
    do_sub = []
    for i, (w1, pos, w2) in tqdm(enumerate(word_pairs), total=len(word_pairs)):
        # create indices
        idx = np.arange(len(swap_arr))
        has_sub = idx[swap_arr[:, i] == 1]
        rs.shuffle(has_sub)
        do_sub.append(list(has_sub[:int(0.1 * len(has_sub))]))

    #Performs the map that will perturb the data. Records the perturbation in the "order" section of the data
    def edit(x, index):
        for i, (w1, pos, w2) in enumerate(word_pairs):
            if index not in do_sub[i]:
                continue
            order = x['order'] + f'{i}:'
            new_text = x['text'].replace(f' {w1} ', f' {w2} ', 1)
            assert (new_text != x['text'])
            x["text"] = new_text
            x["order"] = order
        return x

    edited_ds = edited_ds.map(
        edit,
        num_proc=num_cpus,
        with_indices=True,
        keep_in_memory=True
    )

    #saves the data
    edited_ds.save_to_disk(hf_dataset_name)

def convert_huggingface_to_json():
    edited_ds = datasets.load_from_disk(hf_dataset_name)
    edited_ds.to_json(out_json_dataset_name, num_proc=num_cpus)

if __name__ == '__main__':
    # main()
    convert_huggingface_to_json()
