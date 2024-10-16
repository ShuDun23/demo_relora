"""
Download and pre-tokenize a huggingface dataset.
Based on: https://github.com/conceptofmind/PaLM/blob/main/palm/build_dataloaders.py

Usage:
    python build_dataloaders.py --tokenizer EleutherAI/gpt-neox-20b --dataset openwebtext --text_field text --sequence_length 2048
"""
# 下载并预处理一个Hugging Face数据集，然后使用指定的分词器(tokenizer)进行分词和切分，最后将pre-processed dataset保存到磁盘
import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import time
import json
import argparse
import multiprocessing

from loguru import logger
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer


from peft_pretraining.dataloader import tokenize_and_chunk


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, required=True, help="HuggingFace tokenizer name")
    parser.add_argument("--dataset", type=str, required=True, help="HuggingFace dataset name. E.g., wikitext")
    parser.add_argument("--dataset_config", type=str, default=None, help="HuggingFace dataset config name. E.g., wikitext-2-v1")
    parser.add_argument("--text_field", type=str, default="text", help="Name of the text field in the dataset")
    parser.add_argument("--sequence_length", type=int, default=2048, help="Sequence length")
    parser.add_argument("--num_cpu", type=int, default=multiprocessing.cpu_count(), help="Number of CPU cores")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the pre-tokenized dataset")
    parser.add_argument("--data_dir", type=str, default="/home/hnsheng2/datasets/c4-datasets/",
                        help="Directory containing the C4 dataset files")
    parser.add_argument("--max_train_files", type=int, default=21,
                        help="Maximum number of training files to use")

    parser.add_argument("--take", type=int, default=None, help="Number of examples to take from the dataset")
    args = parser.parse_args(args)

    return args


def main(args):
    print("In main")
    logger.info("*" * 40)
    logger.info(f"Starting script with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)

    _tokenizer_name_for_save = args.tokenizer.replace("/", "_") # 将tokenizer名称中的"/"替换为"_"
    save_path = os.path.join(args.save_dir, f"{args.dataset}_{_tokenizer_name_for_save}_{args.sequence_length}") # 将save_path设置为args.save_dir/dataset_tokenizer_sequence_length
    if args.dataset_config is not None: # 如果提供了dataset_config，则将dataset_config也添加到save_path中
        save_path = os.path.join(args.save_dir, f"{args.dataset}_{args.dataset_config}_{_tokenizer_name_for_save}_{args.sequence_length}")

    if os.path.exists(save_path): # 如果save_path已经存在，则抛出错误
        raise ValueError(f"Path {save_path} already exists")

    # 从 huggingface 网站加载 tokenizer 和 dataset
    # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer) # 加载tokenizer 实例中用的是t5-base
    # logger.info(f"Loading the dataset in streaming mode: {args.take is not None}")
    # dataset = load_dataset(args.dataset, args.dataset_config, streaming=args.take is not None)
    # 加载huggingface数据集


    # 加载本地 tokenizer 和 数据集 ./your/path/bigscience_t0/config.json
    tokenizer = AutoTokenizer.from_pretrained("/home/hnsheng2/PEFT/ReLoRA/tokenizer/")
    logger.info(f"Loading the dataset from local files")
    data_dir = args.data_dir if hasattr(args, 'data_dir') else "/home/hnsheng2/datasets/c4-datasets/"
    max_train_files = args.max_train_files if hasattr(args, 'max_train_files') else 21

    train_files = [os.path.join(data_dir, "train", f"c4-train.{i:05d}-of-01024.json.gz") for i in range(min(max_train_files, 21))]
    val_files = [os.path.join(data_dir, "validation", f"c4-validation.{i:05d}-of-00008.json.gz") for i in range(8)]

    # 确保文件存在
    train_files = [f for f in train_files if os.path.exists(f)]
    val_files = [f for f in val_files if os.path.exists(f)]

    logger.info(f"Found {len(train_files)} training files and {len(val_files)} validation files.")

    dataset = load_dataset(
        "json",
        data_files={
            "train": train_files,
            "validation": val_files
        },
        streaming=args.take is not None
    )


    if args.take is not None: # 如果提供了take，则将数据集转换为Dataset
        logger.info(f"Taking {args.take} examples from the dataset")
        def take(ds, n):
            return Dataset.from_generator(lambda: (yield from ds.take(n)))  # 将数据集转换为Dataset
        dataset_dict = {k: take(v, args.take) for k, v in dataset.items()}  # 将数据集转换为DatasetDict
        dataset = DatasetDict(dataset_dict) # 将数据集转换为DatasetDict

    logger.info("Tokenizing and chunking the dataset") # 分词和切分数据集
    _time = time.time()
    dataset = tokenize_and_chunk( 
        tokenizer=tokenizer,
        dataset=dataset,
        text_field=args.text_field,
        sequence_length=args.sequence_length, 
        num_cpu=args.num_cpu,
    ) # 分词和切分数据集
    _hours = (time.time() - _time) / 3600
    logger.info(f"Tokenization and chunking took {_hours:.2f} hours")

    dataset.save_to_disk(save_path) # 将tokenized and chunked数据集保存到磁盘
    logger.info(f"Saved the dataset to {save_path}")

    with open(os.path.join(save_path, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    print("In main")


if __name__ == "__main__":
    print("Starting the script")
    args = parse_args()
    main(args)

