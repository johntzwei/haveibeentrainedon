
#this edits a yaml file with certain parameters
def edit_yaml(path_to_yaml, **kwargs):
    import yaml
    print(path_to_yaml)
    with open(path_to_yaml) as f:
        list_doc = yaml.safe_load(f)
    for k, v in kwargs.items():
        # if (k not in list_doc):
        #     raise Exception(f"incorrect hyperparameters supplied -> {k}")
        list_doc[k] = v
    with open(path_to_yaml, 'w') as f:
        yaml.dump(list_doc, f, default_flow_style=False)

#returns if cuda is available
def get_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


#sets up model
def setup_model(path_to_model, float_16=False):
    from transformers import AutoModelForCausalLM
    import torch
    if float_16:
        model = AutoModelForCausalLM.from_pretrained(path_to_model, revision="float16", torch_dtype=torch.float16,
                                                     return_dict=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(path_to_model, return_dict=True)
    print(f"imported model from {path_to_model}")
    return model

#sets up tokenizer
def setup_tokenizer(path_to_tokenizer):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path_to_tokenizer)

#calculates the perplexity given a (potentially batched) sequence of losses
def calculate_perplexity(losses):
    import numpy as np
    losses = np.array(losses)
    return 0
    # #if losses is batched
    # if (len(losses.shape))
