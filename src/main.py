import random
import numpy as np

import torch
import torch.utils.data

import transformers
from transformers import AutoConfig, AutoTokenizer
import datasets

from loguru import logger

from dataloader import PreprocessedIterableDataset
from llama import LlamaForCausalLM
from trainer import train_model
from utils import parse_args

transformers.logging.set_verbosity_error()


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    local_rank = 0
    torch.cuda.set_device(local_rank)

    logger.info(f"Device: {torch.cuda.current_device()}")

    logger.info("Process group initialized")
    device = f"cuda:{local_rank}"
            
    logger.info("*" * 40)
    logger.info(f"Starting training with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)

    data = datasets.load_dataset("allenai/c4", "en", split="train", streaming=True)

    seed_for_shuffle = 42 
    
    logger.info(f"Shuffling data with seed {seed_for_shuffle}")
    data: datasets.Dataset = data.shuffle(seed=seed_for_shuffle)

    # it doesn't matter which tokenizer we use, because we train from scratch
    # T5 tokenizer was trained on C4 and we are also training on C4, so it's a good choice
    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=args.max_length)

    dataset = PreprocessedIterableDataset(data, tokenizer, batch_size=args.batch_size, max_length=args.max_length)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=args.workers)

    model_config = AutoConfig.from_pretrained("./configs/"+args.model_config +".json")
    model_config.hyper_llama = args.hyper_llama

    model = LlamaForCausalLM(model_config)

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()

    train_model(model, model_config, tokenizer, dataloader, device, args)


if __name__ == "__main__":
    print("Starting script")
    args = parse_args(None)
    main(args)


"""
    if args.total_batch_size is not None:
        if args.gradient_accumulation is None:
            args.gradient_accumulation = args.total_batch_size // (args.batch_size)
            assert args.gradient_accumulation > 0, "gradient_accumulation must be greater than 0"

    assert args.gradient_accumulation * args.batch_size == args.total_batch_size, \
        "gradient_accumulation * batch_size must be equal to total_batch_size"


    exists in utils file as well
"""