def generate_random_sequence_string(num_sequence, length, vocab_size, start_range):
    from src.unstealthy.score import get_random_sequences
    random_sequence = get_random_sequences(num_sequence, length, vocab_size, start_range)
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    random_sequence = tokenizer.batch_decode(random_sequence)
    return random_sequence

#creates the jsonl file, and returns the prop_inputs to be turned into a csv
def edit_json_unstealthy_scaling(orig_jsonl, new_jsonl, watermarks, k, info):
    import json
    from tqdm import tqdm
    import numpy as np
    import pandas as pd

    tot_len = 0
    with open(orig_jsonl, "r") as orig_file:
        tot_len = sum(1 for _ in orig_file)

    #generate a starting document to perturb for each watermark
    #note that since the pile is already shuffled, and neox shuffles the training set for us, we just select a starting document to start perturbing
    #We want to make sure that there is no overlap between the watermarks
    #we want the documents selected to be consistent within each seed, so we hardcode the upper limit to be 100000, and always generate 1000000 numbers
    #note that 1e9 tokens corresponds to about 565832 documents
    assert(tot_len >= 100000) #assume that the number of documents in the datset is more than 100000
    assert(len(watermarks) * k <= 100000) #assume that the number of documents needed to watermark is less than 100000
    random_dictionary = np.random.choice(100000, size=100000, replace=False)
    perturbed_instances = random_dictionary[:len(watermarks) * k]


    data = []

    #begin creating jsonl output
    with open(orig_jsonl, "r") as orig_file, open(new_jsonl, "w") as new_file:
        for ind, line in tqdm(enumerate(orig_file), total=tot_len):

            if (ind in perturbed_instances):
                watermark_ind = perturbed_instances.tolist().index(ind)
                watermark = watermarks[watermark_ind // k]
                line = json.loads(line)
                line["text"] = line["text"] + "\n" + watermark
                line["order"] = watermark
                row = []
                row.append(ind)
                row.append(line["text"])
                row.append(len(line["text"]) - info["watermark_length"])
                row.append(info["watermark_length"])
                row.append(info["vocab_size"])
                row.append(watermark)
                data.append(row)
                new_file.write(json.dumps(line) + "\n")
            else:
                new_file.write(line)

    prop_inputs = pd.DataFrame(data)
    print(f"prop_inputs has {len(prop_inputs)} number of perturbed examples! ")
    prop_inputs.columns = ['example_index', 'text', 'sub_index', 'seq_len', 'vocab_size', 'watermark']
    return prop_inputs


#This function serves as a main function
def perturb_dataset(exp_name, **kwargs):
    import numpy as np
    import os
    #we first set the seed fixed
    np.random.seed(kwargs["seed"])
    #We just want to simply perturb the dataset randomly
    if (exp_name == "unstealthy_scaling" or exp_name == "unstealthy_raretoken" or exp_name == "unstealthy_tradeoff"):
        from src.utils import setup_dataset
        #We only have one sequence, so we just take the first random sequence
        random_sequences = generate_random_sequence_string(kwargs["num_watermarks"], kwargs["watermark_length"], kwargs["vocab_size"], kwargs["start_range"])

        #perturb the dataset
        out_jsonl = os.path.join(kwargs['out_dir'], f"{kwargs['repetition']}_dataset.jsonl")
        prop_inputs = edit_json_unstealthy_scaling(kwargs["raw_dataset"], out_jsonl, random_sequences, kwargs["repetition"],
                                                   {"watermark_length": kwargs["watermark_length"], "vocab_size": kwargs["vocab_size"]})
        print("finished outputting jsonl file! Starting propagation_inputs.csv")
        out_prop_inputs = os.path.join(kwargs['out_dir'], f"{kwargs['repetition']}_propagation_inputs.csv")
        prop_inputs.to_csv(out_prop_inputs, index=False, header=True)
        print("finished outputting propagation_inputs.csv!")
    #We want to perturb the dataset multiple times (repetition experiment)

