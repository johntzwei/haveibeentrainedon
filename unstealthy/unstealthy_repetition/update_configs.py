import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils import edit_yaml

def main(args):

    update_configs_model = dict()

    #for batch size configs
    update_configs_model["global_num_gpus"] = args.global_num_gpus
    update_configs_model["train_micro_batch_size_per_gpu"] = args.train_micro_batch_size_per_gpu
    update_configs_model["train_batch_size"] = args.train_batch_size
    update_configs_model["gradient_accumulation_steps"] = args.gradient_accumulation_steps

    #wikitext has 117919547 tokens
    update_configs_model["train_iters"] = args.train_iters
    update_configs_model["lr_decay_iters"] = args.train_iters
    update_configs_model["checkpoint_factor"] = args.train_iters
    update_configs_model["eval_interval"] = max(100, args.train_iters // 40) #assuming we track 40 times
    update_configs_model["log_interval"] = max(10, update_configs_model["eval_interval"])
    update_configs_model["steps_per_print"] = update_configs_model["log_interval"]
    edit_yaml(args.path_to_model_yaml, **update_configs_model)

    #for the local_setup.yml
    update_configs_setup = dict()

    update_configs_setup["data_path"] = args.data_path
    update_configs_setup["save"] = args.save
    update_configs_setup["load"] = args.save
    wandb_project = args.save.split("/")[-2] + "_" + args.save.split("/")[-3] + args.save.split("/")[-4] #-2 is model size, -3 is data_type, -4 is experiment name
    wandb_group = args.save.split("/")[-1] #-1 is model_id
    update_configs_setup["wandb_project"] = wandb_project
    update_configs_setup["wandb_group"] = wandb_group
    update_configs_setup["include"] = args.include
    from random import randint
    update_configs_setup["master_port"] = args.master_port + 29500 + randint(1, 10000)

    edit_yaml(args.path_to_setup_yaml, **update_configs_setup)
def parse_args():
    parser = argparse.ArgumentParser()


    parser.add_argument(
        '--path_to_model_yaml',
        help="the path to the model yaml"
    )
    parser.add_argument(
        '--path_to_setup_yaml',
        help="the path to the local_setup yaml"
    )

    parser.add_argument(
        '--global_num_gpus',
        # required=True,
        default=1,
        type=int,
        help="global_num_gpus"
    )

    parser.add_argument(
        '--train_batch_size',
        # required=True,
        default=1024,
        type=int,
        help="train_batch_size"
    )

    parser.add_argument(
        '--gradient_accumulation_steps',
        # required=True,
        default=32,
        type=int,
        help="gradient_accumulation_steps"
    )

    parser.add_argument(
        '--train_micro_batch_size_per_gpu',
        # required=True,
        default=32,
        type=int,
        help="train_micro_batch_size_per_gpu"
    )

    parser.add_argument(
        '--train_iters',
        # required=True,
        default=56,
        type=int,
        help="train_iters -> look at how long dataset is"
    )

    parser.add_argument(
        '--data_path',
        help="the path to our data"
    )
    parser.add_argument(
        '--save',
        help="the name of the directory to save the trained model in"
    )
    parser.add_argument(
        '--include',
        default="localhost:0",
        help="GPU numbers"
    )
    parser.add_argument(
        '--master_port',
        default=29500,
        type=int,
        help="the port to which we run the model using deepspeed"
    )

    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    main(args)