import datasets

datasets.load_dataset('wikitext', 'wikitext-103-raw-v1')['train'] \
        .to_json('wikitext.jsonl')
