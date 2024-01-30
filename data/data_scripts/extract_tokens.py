import os
import sys

#note: this script is only meant to extract a certain number of tokens. We are assuming gpt2 tokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.from_raw import extract_k_tokens_and_store

# raw_json = "./../45e8_tokens.jsonl"
raw_json = "/home/johnny/data/00.jsonl"
out_dataset = "./../pile12e9_orig.jsonl"

def main():
    print("entered! ")
    print(sys.path)
    print(os.getcwd())
    #check if out_dataset exists already
    if (os.path.exists(out_dataset)):
        print("out_dataset already exists!. Exiting...")
        return
    from datasets import Features, Value
    extract_k_tokens_and_store(raw_json, out_dataset, 12e9)


if __name__=="__main__":
    main()