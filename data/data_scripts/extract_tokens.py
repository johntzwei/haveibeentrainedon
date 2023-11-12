import os
import sys

#note: this script is only meant to extract a certain number of tokens. We are assuming gpt2 tokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.from_raw import extract_k_tokens_and_store

raw_json = "./../45e8_tokens.jsonl"
out_dataset = "./../pile_1e9_tokens_hf"

def main():
    print("entered! ")
    from datasets import Features, Value
    extract_k_tokens_and_store(raw_json, out_dataset, 1e9)


if __name__=="__main__":
    main()