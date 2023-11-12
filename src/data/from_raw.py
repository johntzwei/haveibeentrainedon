


#this function will take in a json file of data, and extract sequences until a certain number of tokens
# has been reached and save this dataset
def extract_k_tokens_and_store(in_json_path, out_dataset_path, num_tokens):
    from src.utils import setup_tokenizer, setup_dataset
    tokenizer = setup_tokenizer("gpt2")
    print("setting up dataset! ")
    dataset = setup_dataset(in_json_path)["train"]

    print("starting encoding! ")

    def tokenize_function(examples):
        return {'len': len(tokenizer(examples["text"])['input_ids'])}

    len_ds = dataset.map(tokenize_function, num_proc=100)

    #we now find the index that has the total sum > num_tokens

    length_arr = len_ds["len"]
    found_idx = len(length_arr)
    tot_sum = 0
    for idx in range(len(length_arr)):
        tot_sum += length_arr[idx]
        if (tot_sum > num_tokens):
            found_idx = idx
            break


    final_dataset = dataset.select(range(found_idx))
    final_dataset.save_to_disk(out_dataset_path)
