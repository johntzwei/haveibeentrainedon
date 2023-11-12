
import argparse

#counts the number of tokens in a dataset
# def count_tokens():
#     from transformers import AutoTokenizer
#     from datasets import load_from_disk
#     tokenizer = AutoTokenizer.from_pretrained("gpt2")
#     dataset = load_from_disk("/home/ryan/haveibeentrainedon/data/wikitext/105_dataset/105_dataset.hf")
#
#     tokenized_dataset = tokenizer(dataset["text"], return_attention_mask=False)
#
#     lengths = [len(i) for i in tokenized_dataset["input_ids"]]
#     print(lengths)

def main(args):
    if (args.mode == "get_model_dir"):
        out_dir_name = args.dataset_dir.split("/")[-1].split("_")[0] #calculates the corresponding label name
        print(out_dir_name)

def parse_args():
    parser = argparse.ArgumentParser()


    parser.add_argument(
        '--mode',
        required=True,
        help="what function to execute"
    )

    parser.add_argument(
        '--dataset_dir',
        help="the name of the dataset that was used to train the model"
    )

    parser.add_argument(
        '--model_out_dir',
        help="the output directory that we want to create our model output folder in"
    )

    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    main(args)