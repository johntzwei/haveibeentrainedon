import os
import sys

#note: this script is only meant to extract a certain number of tokens. We are assuming gpt2 tokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.from_raw import extract_k_tokens_and_store

# raw_json = "./../45e8_tokens.jsonl"
raw_json = "/home/johnny/data/00.jsonl"
out_dataset = "./../pile1e8_orig.jsonl"

def main():
    print("entered! ")
    print(sys.path)
    print(os.getcwd())
    from datasets import Features, Value
    extract_k_tokens_and_store(raw_json, out_dataset, 1e8)


if __name__=="__main__":
    main()