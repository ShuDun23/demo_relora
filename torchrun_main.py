"""
Distributed training code for ReLoRA.
分布式训练脚本 调用relora.py中的 ReLoRaModel和 ReLoRaLinear来实现relora技术
包含了用于分布式训练的参数解析、模型评估、模型保存、数据集加载和训练循环的代码。
使用了PyTorch的分布式训练功能，如 DistributedDataParallel 和 ZeroRedundancyOptimizer
分布式训练就是在多个GPU上训练，进而需要在多个GPU之间更新和同步数据
"""
import os
import sys
import yaml
import time
import json
import random # 以上为标准库
import argparse #  常用库 用来定义和解析命令行参数 也可以用 optparse, click库
from typing import Union

import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist # 分布式训练的库
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
)

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig,
    default_data_collator,
)
from tokenizers import Tokenizer

import datasets
import datasets.distributed # 分布式数据加载
import wandb # 用于跟踪实验

from tqdm import tqdm
from loguru import logger # 记录日志的库

from peft_pretraining import training_utils, args_utils
from peft_pretraining.dataloader import SkipDataLoader
from peft_pretraining.modeling_llama import LlamaForCausalLM
from peft_pretraining.modeling_pythia import GPTNeoXForCausalLM
from peft_pretraining.relora import ReLoRaModel, ReLoRaLinear, merge_and_reinit_functional

from peft_pretraining.megatron_dataset.arguments import NeoXArgs
from peft_pretraining.megatron_dataset import data_utils as megatron_data_utils

transformers.logging.set_verbosity_error()


def parse_args(args=None): # 用argparse库定义了一些命令行参数
    parser = argparse.ArgumentParser()

    parser.add_argument("--training_config", type=str, default=None,
                        help="Alternative to providing the parameters. Overrides all parameters. Path to a yaml file with training run config")

    parser.add_argument("--model_config", type=str, default=None) # 模型配置文件路径
    parser.add_argument("--model_name_or_path", type=str, default=None, help="Huggingface model identifier, alternative to --model_config")
    parser.add_argument("--model_revision", type=str, default=None, help="Tag name, branch name, or commit hash of the model from HuggingFace Hub. E.g., v2.0.1 or step1000")
    parser.add_argument("--warmed_up_model", type=str, default=None, help="Start with warmed-up model weights. Does not reload former optimizer and scheduler.")
    parser.add_argument("--resume_from", type=str, default=None, help="Continue training with ReLoRA, loading optimizer and scheduler from the checkpoint.")
    parser.add_argument("--load_optimizer_state_on_resume", default=True, type=lambda x: x.lower() == "true",
                        help="Load optimizer state from the checkpoint when resuming training. "
                             "If False, optimizer state will be initialized from scratch. Setting it to False is useful for some very specific experiments.")

    parser.add_argument("--dataset_path", type=str, default=None, help="Path to a huggingface dataset directory")
    parser.add_argument("--megatron_dataset_config", type=str, default=None,
                        help="Path to a megatron dataset config file. Only one of --dataset_path and --megatron_dataset_config should be provided.")
    parser.add_argument("--max_length", type=int, default=512) # 一个序列的最大长度

    parser.add_argument("--batch_size", type=int, default=None) # 每个GPU上处理的批量大小
    parser.add_argument("--gradient_accumulation", type=int, default=None) # 梯度累积的个数
    parser.add_argument("--total_batch_size", type=int, default=None) # 所有GPU上的总批量大小

    parser.add_argument("--use_peft", default=False, type=lambda x: x.lower() == "true") # 是否使用peft 即是否使用lora或relora
    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--lora_alpha", type=float, default=32)
    parser.add_argument("--relora", type=int, default=None)
    parser.add_argument("--train_scaling", default=False, action="store_true")
    parser.add_argument("--reset_optimizer_on_relora", default=True, type=lambda x: x.lower() == "true")
    parser.add_argument("--optimizer_random_pruning", default=0.0, type=float,
                        help="Use random pruning to reduce optimizer matrix internal dimensionality.")
    parser.add_argument("--optimizer_magnitude_pruning", default=0.0, type=float,
                        help="Use magnitude pruning to reduce optimizer matrix internal dimensionality.")
    parser.add_argument("--force_keep_original", default=False, type=lambda x: x.lower() == "true",
                        help=("Keep original model parameters even if relora is None. "
                              "Useful for making sure that full-LoRa model is equivalent to model+LoRa."))

    parser.add_argument("--optimizer", default="Adam", help="Could be adam (for AdamW) or adam_zero for ZeroRedundancyOptimizer(AdamW)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["linear", "cosine", "cosine_restarts"])
    parser.add_argument("--cycle_length", type=int, default=None, help="Number of steps per cycle for cosine scheduler")
    parser.add_argument("--restart_warmup_steps", type=int, default=None, help="Number of steps for cosine restarts (only used for cosine_restarts)")
    parser.add_argument("--adjust_step", type=int, default=0, help="Number of steps to adjust the scheduler by. "
                            f"Useful when you want to sync ReLoRA resets with the scheduler for a warmed up model. "
                            f"You need to use it, when your warmup_step % relora_resets != 0")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1) # 最小学习率与初始学习率的比值
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=1_000)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)

    parser.add_argument("--eval_every", type=int, default=1_000) # evaluation

    parser.add_argument("--num_training_steps", type=int, default=10_000,
                        help="Number of **update steps** to train for. "
                             "Notice that gradient accumulation is taken into account.")
    parser.add_argument("--max_train_tokens", type=training_utils.max_train_tokens_to_number, default=None,
                        help="Number of tokens to train on. Overwrites num_training_steps. "
                             "You can use M and B suffixes, e.g. 100M or 1B.")
    parser.add_argument("--save_every", type=int, default=10_000) # 保存频率
    parser.add_argument("--save_dir", type=str, default=None) # 保存目录
    parser.add_argument("--keep_checkpoints", type=int, default=None,
                        help="Number of checkpoints to keep. By default, keep all checkpoints.")
    parser.add_argument("--tags", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float32")
    parser.add_argument("--workers", type=int, default=8)

    # parser.add_argument("--quantize", default=None, type=str, choices=[None, "4bit", "8bit"])
    # parser.add_argument("--use_double_quant", default=True, type=lambda x: x.lower() == "true")

    parser.add_argument("--distributed_type", type=str, default="ddp", choices=["fsdp", "ddp"])
    parser.add_argument("--profile", default=False, type=lambda x: x.lower() == "true")
    parser.add_argument("--autoresume", default=False, type=lambda x: x.lower() == "true")
    parser.add_argument("--comment", type=str, default=None, help="Wandb notes")
    parser.add_argument("--wandb_watch", default=False, type=lambda x: x.lower() == "true",
                        help="Enable wandb.watch (may make training unstable, but might be good for understanding gradients)")
    parser.add_argument("--skip_batches", default=None, type=str, help="Batch numbers to skip, separated by comma. E.g., 2003,2990,12309. Specifically, update_step numbers.")

    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args(args)

    args = args_utils.check_args_torchrun_main(args)

    return args


@torch.no_grad() # 用评估数据集进行模型评估

def evaluate_model(model: nn.Module, eval_dataloader, device, target_eval_tokens=10_000_000):
    _time = time.time()
    was_training = model.training # 记录模型是否在训练模式
    model.eval() # 将模型设置为评估模式，这会关闭 dropout和 batch normalization等仅在训练时使用的层

    """
    初始化一个于 累积损失和 统计信息的张量
    tensor可以在GPU上加速 而list只能在CPU上运行
    """
    ddp_loss_info = torch.zeros(3).to(device)  # [loss, n_batches, n_tokens] 这并不是list, 而是一个PyTorch tensor
    tokens_in_batch_info = torch.zeros(1).to(device)
    
    rank = dist.get_rank()
    for i, batch in enumerate(eval_dataloader): # 使用enumerate遍历评估数据加载器中的批次
        if i == 0:
            # this way of estiming the number of eval steps
            # is needed to avoid a deadlock when using FSDP
            batch["input_ids"]: torch.Tensor
            tokens_in_batch_info[0] += batch["input_ids"].numel()
            dist.all_reduce(tokens_in_batch_info, op=dist.ReduceOp.SUM)
            n_eval_iters = int(target_eval_tokens / tokens_in_batch_info[0])

        if target_eval_tokens != -1 and i > n_eval_iters: break

        batch = {k: v.to(device) for k, v in batch.items()}

        loss = model(**batch, labels=batch["input_ids"]).loss # 将批次数据移动到正确的设备上，并计算损失(包含feedforward和计算loss)
        if torch.isnan(ddp_loss_info[0]):
            print(f"Rank {dist.get_rank()} got nan loss. This is probably a bug.")

        tokens_in_batch = batch["input_ids"].numel()
        assert tokens_in_batch > 0, "Batch size is zero"
        ddp_loss_info[0] += loss.detach() # 将当前批次的损失加到总累积损失中
        ddp_loss_info[1] += 1 # 更新批次数
        ddp_loss_info[2] += tokens_in_batch # 更新处理过的总tokens数

    # check if loss is nan
    if torch.isnan(ddp_loss_info[0]):
        raise RuntimeError(f"Rank {rank} got nan loss. This is probably a bug.")

    # Gather losses across all GPUs
    dist.all_reduce(ddp_loss_info, op=dist.ReduceOp.SUM) # 所有GPU间累加ddp_loss_info 其中包含[1] [2] [3]
    eval_loss = ddp_loss_info[0] / ddp_loss_info[1] # 计算平均损失 总损失除以更新batch数
    evaluated_on_tokens = ddp_loss_info[2].item() 
    logger.info(f"Evaluated on {evaluated_on_tokens} tokens, eval loss: {eval_loss:.4f}") # 用日志记录器logger记录所用的时间和loss

    logger.info(f"Evaluation took {time.time() - _time:.2f} seconds")

    if was_training: model.train() # 从eval模式切回train模式
    return eval_loss, evaluated_on_tokens

# 保存模型 save_model_ddp和 save_model_fsdp函数用于在分布式数据并行（DDP）和全参数分片数据并行（FSDP）设置中保存模型
# Distributed Data Parallel DDP
def save_model_ddp(model, optimizer, scheduler, training_state_checkpoint, run_config, save_dir):
    # 检查模型是否被 DDP 包装
    if hasattr(model, 'module'):
        _model = model.module  # 获取DDP包装器内部的原始模型
    else:
        _model = model  # 如果没有被包装，直接使用模型

    # 保存模型
    _model.save_pretrained(save_dir)

    # 保存优化器状态
    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))

    # 保存调度器状态
    if scheduler is not None:
        torch.save(scheduler.state_dict(), os.path.join(save_dir, "scheduler.pt"))

    # 保存训练状态
    if training_state_checkpoint is not None:
        torch.save(training_state_checkpoint, os.path.join(save_dir, "training_state.pt"))

    # 保存运行配置
    if run_config is not None:
        # 将 run_config 中的 set 转换为 list
        def set_to_list(obj):
            if isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, dict):
                return {k: set_to_list(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [set_to_list(v) for v in obj]
            else:
                return obj

        serializable_run_config = set_to_list(run_config)
        with open(os.path.join(save_dir, "run_config.json"), "w") as f:
            json.dump(serializable_run_config, f, indent=2)

    logger.info(f"Model and associated states successfully saved to {save_dir}")

# Fully Sharded Data Parallel FSDP
# FSDP通过将模型参数分片到各个GPU上，可以有效地减少每个GPU上的内存占用，这对于大型模型特别有用。DDP则通常在每个GPU上保存完整的模型参数
def save_model_fsdp(model, optimizer, scheduler, training_state_checkpoint, run_config, save_dir): # 在全参数分片数据并行(FSDP)设置下保存
    raise RuntimeError("FSDP is not supported anymore. There were a lot of isses with ReLoRA and FSDP and no speed or memory improvements.")
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
        global_rank = dist.get_rank()
        update_step = training_state_checkpoint["update_step"]

        if global_rank == 0:
            os.makedirs(os.path.dirname(save_dir), exist_ok=True)

        _model = model.module
        _model.save_pretrained(save_dir)

        if global_rank == 0:
            optimizer_checkpoint = {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "update_step": update_step,
                "global_step": training_state_checkpoint["global_step"],
                "config": run_config,
                "wandb": wandb.run.dir,
                "dtype": args.dtype,
            }
            torch.save(optimizer_checkpoint, f"{save_dir}/optimizer.pt")

            training_state_checkpoint["wandb_id"] = wandb.run.id
            with open(f"{save_dir}/training_state.json", "w") as f:
                json.dump(training_state_checkpoint, f, indent=4)

# save_model 函数根据分布式类型选择 save_model_ddp 或 save_model_fsdp
def save_model(model, *, optimizer, scheduler, training_state_checkpoint, run_config, distributed_type, save_dir):
    """
    Args:
        training_state_checkpoint: dict with keys:
            global_step: int
            update_step: int
            tokens_seen: int
            tokens_seen_before: int
            n_lora_restarts: int
            update_time: float
        run_config: 
    """
    if distributed_type == "ddp":
        save_model_ddp(model, optimizer, scheduler, training_state_checkpoint, run_config, save_dir)
    elif distributed_type == "fsdp":
        save_model_fsdp(model, optimizer, scheduler, training_state_checkpoint, run_config, save_dir)
    else:
        raise ValueError(f"Unknown distributed type {distributed_type}")


# 加载megatron数据集 Megatron是一个用于训练多亿参数的神经网络模型的库
def load_megatron_dataset(args, world_size, start_iteration): 
    logger.info(f"Loading Megatron dataset arguments from {args.megatron_dataset_config}")
    with open(args.megatron_dataset_config) as f:
        dataset_config_yaml = yaml.safe_load(f) # 加载Megatron数据集配置文件

    # 更新配置参数
    dataset_config_yaml["global_num_gpus"] = world_size
    dataset_config_yaml["train_micro_batch_size_per_gpu"] = args.batch_size
    dataset_config_yaml["gradient_accumulation_steps"] = args.gradient_accumulation
    dataset_config_yaml["train_batch_size"] = args.total_batch_size
    dataset_config_yaml["num_workers"] = args.workers

    if args.max_length != dataset_config_yaml["seq_length"]:
        logger.warning(f"rags.max_length ({args.max_length}) does not match "
                        f"seq_length ({dataset_config_yaml['seq_length']}) in the dataset config")
        logger.warning(f"Overwriting max_length with seq_length")
        args.max_length = dataset_config_yaml["seq_length"]
    
    if args.num_training_steps > dataset_config_yaml["train_iters"]:
        logger.error(f"num_training_steps ({args.num_training_steps}) is greater than train_iters ({dataset_config_yaml['train_iters']})")
        raise ValueError("num_training_steps must be less than train_iters")

    # 设置tokenizer 把文本转化成更小的单元(token)
    tokenizer = Tokenizer.from_file(dataset_config_yaml["vocab_file"])

    # 打印配置信息
    logger.info("*" * 40)
    logger.info("Dataset arguments:")
    for k, v in dataset_config_yaml.items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)
    logger.info("Building Megatron dataset")
    dataset_args = NeoXArgs.from_dict(dataset_config_yaml)

    if dataset_args.iteration is None:
        dataset_args.iteration = start_iteration

    if dataset_args.train_batch_size != args.total_batch_size:
        logger.error(f"megatron_dataset_args.train_batch_size ({dataset_args.train_batch_size}) "
                        f"does not match total_batch_size ({args.total_batch_size})")
        raise ValueError("megatron_dataset_args.train_batch_size must match total_batch_size")

    train_loader, eval_loader, test_loader = megatron_data_utils.\
        build_train_valid_test_dataloaders(neox_args=dataset_args) # 加载数据集
    logger.info("Megatron dataset built")
    tokenizer.name_or_path = dataset_config_yaml["vocab_file"]
    return train_loader, eval_loader, test_loader, tokenizer

# 创建和启动一个性能分析器 profiler 这个性能分析器可以帮助开发者了解模型训练过程中的性能瓶颈和资源使用情况
def maybe_make_profiler(args): 
    if not args.profile: return None
    global_rank = dist.get_rank()
    profiler_logging_dir = os.path.join(f"profiler_logs/{args.run_name}") # 构建一个目录
    prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_logging_dir, worker_name=f"rank{global_rank}"),
        record_shapes=True, # 记录操作的输入和输出形状
        profile_memory=True, # 启用内存使用情况的分析
        with_stack=True, # 启用Python栈跟踪记录
    )
    print(f"Rank {global_rank} profiling results will be saved to {profiler_logging_dir}")
    prof.start() # 启动性能分析器
    return prof

# 设置训练环境、加载数据集、初始化模型、配置优化器和学习率调度器、执行训练循环，并在训练过程中保存模型和训练状态
def main(args): # args = parse_args()

    # seed all 设置随机种子 确保可重复性
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # 获取环境变量和分布式环境初始化
    assert "LOCAL_RANK" in os.environ, "torchrun应该设置LOCAL_RANK环境变量"
    global_rank = int(os.environ['RANK'])  # global_rank是进程在全局范围内的唯一标识符(秩)
    local_rank = int(os.environ["LOCAL_RANK"])  # local_rank是进程在当前节点上的局部标识符
    world_size = int(os.environ["WORLD_SIZE"])  # world_size是总进程数
    torch.cuda.set_device(local_rank)  # 设置当前进程使用的GPU设备

    logger.info(f"Global rank {global_rank}, local rank {local_rank}, device: {torch.cuda.current_device()}")

    dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)

    logger.info("Process group initialized")
    device = f"cuda:{local_rank}" # 指定device为 cuda:local_rank 即 GPU:local_rank

    if args.total_batch_size is not None:
        if args.gradient_accumulation is None:
            assert args.total_batch_size % world_size == 0, "total_batch_size must be divisible by world_size"
            args.gradient_accumulation = args.total_batch_size // (args.batch_size * world_size)
            assert args.gradient_accumulation > 0, "gradient_accumulation must be greater than 0"

    assert args.gradient_accumulation * args.batch_size * world_size == args.total_batch_size, \
        "gradient_accumulation * batch_size * world_size must be equal to total_batch_size"

    if args.max_train_tokens is not None:
        args.num_training_steps = args.max_train_tokens // args.total_batch_size
        logger.info(f"Setting num_training_steps to {args.num_training_steps} based on max_train_tokens")

    # turn off logger
    if global_rank != 0: logger.remove()

    wandb_id = None
    if args.save_dir is not None and os.path.exists(args.save_dir):
        if not args.autoresume:
            raise ValueError(f"Save directory {args.save_dir} already exists and --autoresume is off. Interrupting...")

        _old_train_config = os.path.join(args.save_dir, "training_config.yaml")
        if os.path.exists(_old_train_config):
            with open(os.path.join(args.save_dir, "training_config.yaml")) as f:
                old_args = yaml.safe_load(f)
            if old_args != vars(args):
                logger.warning(f"Arguments have changed since the last run.")
                logger.warning(f"Training config will be overwritten with new args")

                for k, v in vars(args).items():
                    if old_args.get(k) != v:
                        logger.warning(f"{k:30} {old_args.get(k)} -> {v}")
        else:
            logger.warning(f"Training config not found in the existing save directory {args.save_dir}.")

        training_state, resume_from = training_utils.get_last_training_state(args.save_dir)

        if args.resume_from is None:
            args.resume_from = resume_from

        if training_state is not None:
            wandb_id = training_state["wandb_id"]
        logger.info(f"Resuming training from {resume_from} with wandb id {wandb_id}")

    dist.barrier()  # guarantees none of the workers will read save_dir above here before it's created by rank 0

    # initialize wandb without config (it is passed later)
    # Weights & Biases (wandb)
    if global_rank == 0:
        wandb.init(project="peft_pretraining", tags=args.tags, id=wandb_id, resume="allow", notes=args.comment)
        args.run_name = wandb.run.name
        if args.save_dir is None:
            args.save_dir = f"checkpoints/{wandb.run.name}"

        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, "training_config.yaml"), "w") as f:
            yaml.dump(vars(args), f)

    dist.barrier()  # guarantees that save_dir exists and wand initialized on rank 0

    # synchronize run name and save dir across all ranks
    run_name = [wandb.run.name] if global_rank == 0 else [""]
    dist.broadcast_object_list(run_name, src=0)
    run_name = run_name[0]
    args.run_name = run_name
    if args.save_dir is None:
        args.save_dir = f"checkpoints/{args.run_name}"

    logger.info(f"Using dist with rank {global_rank} (only rank 0 will log)")
    logger.info("*" * 40)
    logger.info(f"Starting training with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)

    if args.dataset_path is not None: # 加载huggingface数据集
        logger.info("Loading Huggingface dataset from directory")
        dataset_dict = datasets.load_from_disk(args.dataset_path)
        logger.info(f"Applying set_format")
        dataset_dict.set_format(type='torch', columns=["input_ids"])

        train_dataset = dataset_dict["train"]
        if args.seed != 0:
            # this weird condition is due to backward compatibility
            train_dataset = train_dataset.shuffle(seed=args.seed)

        eval_dataset = dataset_dict["validation"]

        # ##############################
        # Verify dataset
        logger.info("Checking datasets size")
        minimum_n_tokens = args.total_batch_size * args.num_training_steps
        dataset_n_tokens = len(train_dataset) * args.max_length # 计算数据集的token数
        if dataset_n_tokens < minimum_n_tokens:
            raise ValueError(f"Dataset only has {dataset_n_tokens} tokens, but we need at least {minimum_n_tokens}")

        logger.info("Loading dataset preprocessing args to check on seq_length")
        with open(os.path.join(args.dataset_path, "args.json")) as f:
            dataset_preprocessing_args = json.load(f)
        assert dataset_preprocessing_args["sequence_length"] == args.max_length
        logger.info("All good! Loading tokenizer now")
        # ##############################
        tokenizer = AutoTokenizer.from_pretrained(
            dataset_preprocessing_args["tokenizer"],
            model_max_length=args.max_length,
        )
        logger.info("Tokenizer loaded")

    elif args.megatron_dataset_config is not None: # 加载megatron数据集
        # NOTE: load_megatron_dataset can modify args inplace
        # NOTE: train_dataset and eval_dataset do not exist in this if-branch
        # NOTE: we will set iteration to non-zero below in .resume_from
        start_iteration = 0
        if args.model_revision is not None and args.model_revision.startswith("step"):
            # This piece of code is VERY SPECIFIC to our experimental setup
            # of reproducing Pythia training setup and is not useful in regular cases.
            start_iteration = int(args.model_revision[4:])
            logger.info(f"Starting from iteration {start_iteration} based on model revision {args.model_revision}")
        train_loader, eval_loader, test_loader, tokenizer = load_megatron_dataset(args, world_size=world_size, start_iteration=start_iteration)
        dataset_preprocessing_args = {"tokenizer": tokenizer.name_or_path}

    # 模型初始化
    if args.model_config is not None:
        model_config = AutoConfig.from_pretrained(args.model_config) # 从模型配置文件中加载模型配置 而不是模型本身
        t_vocab_size = tokenizer.get_vocab_size() if isinstance(tokenizer, Tokenizer) else tokenizer.vocab_size

        if model_config.vocab_size != t_vocab_size:
            logger.warning(f"Model config vocab size ({model_config.vocab_size}) does not match tokenizer vocab size ({t_vocab_size})")
            if model_config.vocab_size == 32000 and t_vocab_size == 32100:
                logger.warning("You are most likely reusing old checkpoints. This is alright, but not recommended.")
            else:
                raise ValueError(f"Model config vocab size ({model_config.vocab_size}) does not match tokenizer vocab size ({t_vocab_size})")

        if not isinstance(model_config, LlamaConfig):
            raise NotImplementedError(f"Unknown model config type {type(model_config)}, only LLaMA is supported")

        logger.info("Using local version of LLaMA")
        model = LlamaForCausalLM(model_config) # 使用模型配置创建一个新的模型
    else:
        logger.info(f"Using HuggingFace model {args.model_name_or_path} revision {args.model_revision}")
        model = GPTNeoXForCausalLM.from_pretrained(args.model_name_or_path, revision=args.model_revision)
        model_config = _model.config

    global_step = 0 # 模型已经处理的总batch数 只要处理了一个batch就+1 不考虑是否进行了参数更新
    update_step = 0 # 模型参数已经更新的次数 进行了一次参数更新就+1。在有梯度累积的时候，处理多个batch才更新一次参数，所以global_step>update_step
    tokens_seen = 0 # 已见token数
    tokens_seen_before = 0 # 已见token数(前一个epoch)
    n_lora_restarts = 0 # LoRA重启次数
    n_optimizer_resets = 0 # 优化器重启次数

    """加载预训练模型和训练状态，也就是加载checkpoint"""
    if args.warmed_up_model is not None:
        logger.info("*" * 40)
        logger.info(f"Loading a warmed-up model from {args.warmed_up_model}")
        
        # 使用 from_pretrained 方法加载模型
        model = AutoModelForCausalLM.from_pretrained(args.warmed_up_model)
        logger.info(f"Model successfully loaded")

        # 检查 pad_token_id 是否正确设置
        if model.config.pad_token_id is None or model.config.pad_token_id == -1:
            logger.warning("pad_token_id is not set in the model config. This might cause issues during training or generation.")

        # 检查 generation_config 中的 pad_token_id（如果存在）
        if hasattr(model, 'generation_config'):
            if model.generation_config.pad_token_id is None or model.generation_config.pad_token_id == -1:
                logger.warning("pad_token_id is not set in the generation_config. This might cause issues during text generation.")

        # 加载训练状态（如果存在）
        if os.path.exists(os.path.join(args.warmed_up_model, "training_state.json")):
            logger.info(f"Loading training state variables from {args.warmed_up_model}")
            with open(os.path.join(args.warmed_up_model, "training_state.json")) as f:
                _old_state = json.load(f)
            global_step = _old_state["global_step"]
            update_step = _old_state["update_step"]
            tokens_seen = _old_state["tokens_seen"]
            tokens_seen_before = _old_state["tokens_seen_before"]
            logger.info(f"global_step       : {global_step}")
            logger.info(f"update_step       : {update_step}")
            logger.info(f"tokens_seen       : {tokens_seen}")
            logger.info(f"tokens_seen_before: {tokens_seen_before}")
            logger.info(f"Will train for {args.num_training_steps - update_step} update steps")
        else:
            logger.warning(f"Did not find training state in {args.warmed_up_model}, global step will start from zero")
        logger.info("*" * 40)

    params_before = sum(p.numel() for p in model.parameters())

    # 在指定的线性层上应用ReLoRA
    if args.use_peft:
        need_linear_weight = (
            args.relora is not None
            or args.force_keep_original
            or args.warmed_up_model is not None
        )
        logger.info(f"Wrapping model with LoRA ({need_linear_weight=})")

        # target modules should define all linear layers from transformer block
        # "attn" and "mlp" are used in LLaMA
        # "attention" and "mlp" are used in Pythia
        model = ReLoRaModel(
            model,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.1,
            target_modules=["attn", "attention", "mlp"],
            trainable_scaling=args.train_scaling,
            keep_original_weights=True,
            lora_only=not need_linear_weight,
            # quantize=args.quantize,
            # use_double_quant=args.use_double_quant,
        )

    # 加载checkpoint 从checkpoint中恢复模型和训练状态
    if args.resume_from:
        # 从指定路径加载模型
        logger.info(f"Loading model from {args.resume_from}")
        if isinstance(model, ReLoRaModel):
            # 如果是ReLoRaModel,只加载包装的模型
            model.wrapped_model = AutoModelForCausalLM.from_pretrained(args.resume_from)
        else:
            # 否则直接加载整个模型
            model = AutoModelForCausalLM.from_pretrained(args.resume_from)

        logger.info(f"Model successfully loaded")

        # 加载训练状态信息
        logger.info(f"Loading training state like global_step, update_step, and tokens_seen from {args.resume_from}")
        training_state_path = os.path.join(args.resume_from, "training_state.json")
        if os.path.exists(training_state_path):
            # 如果存在训练状态文件,读取相关信息
            with open(training_state_path) as f:
                _old_state = json.load(f)

            global_step = _old_state["global_step"]
            # 重写update_step以正确初始化调度器
            _update_step = _old_state["update_step"]
            tokens_seen = _old_state["tokens_seen"]
            tokens_seen_before = _old_state["tokens_seen_before"]
            n_lora_restarts = _old_state.get("n_lora_restarts", 0)
            # 打印加载的训练状态信息
            logger.info(f"global_step       : {global_step}")
            logger.info(f"update_step       : {_update_step}")
            logger.info(f"tokens_seen       : {tokens_seen}")
            logger.info(f"tokens_seen_before: {tokens_seen_before}")
            logger.info(f"n_lora_restarts   : {n_lora_restarts}")
            logger.info(f"Will train for {args.num_training_steps - _update_step} update steps")

            # 如果使用megatron数据集,设置batch_sampler的起始迭代
            if args.megatron_dataset_config is not None:
                train_loader.batch_sampler.start_iter = global_step
        else:
            # 如果训练状态文件不存在,记录警告
            logger.warning(f"Training state file not found at {training_state_path}. Starting from scratch.")

        # 检查并设置pad_token_id
        if model.config.pad_token_id is None or model.config.pad_token_id == -1:
            logger.warning("pad_token_id is not set in the model config. Setting it to eos_token_id.")
            model.config.pad_token_id = model.config.eos_token_id

        # 检查并设置generation_config中的pad_token_id
        if hasattr(model, 'generation_config'):
            if model.generation_config.pad_token_id is None or model.generation_config.pad_token_id == -1:
                logger.warning("pad_token_id is not set in the generation_config. Setting it to config.pad_token_id.")
                model.generation_config.pad_token_id = model.config.pad_token_id

    params_after = sum(p.numel() for p in model.parameters())

    added_floats = params_after - params_before

    # print params and trainable params
    logger.info(f"\n{model}\n")
    logger.info(f"Total params  before LoRA: {params_before / 1_000_000:.2f}M")
    logger.info(f"Total params  after  LoRA: {params_after / 1_000_000:.2f}M")
    logger.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M")
    logger.info(f"In total, added {added_floats / 1_000_000:.2f}M parameters to the model")

    logger.info(f"Saving model to {args.save_dir} every {args.save_every} update steps")

    if args.dtype in ["bf16", "bfloat16"]:
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device)

    n_total_params = sum(p.numel() for p in model.parameters())
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    p_trainable_params = n_trainable_params / n_total_params

    # ##############################
    # Distributed wrapping 根据分布式类型(DDP or FSDP)分布式包装模型
    if args.distributed_type == "fsdp":
        logger.info("Wrapping model with FSDP")
        raise RuntimeError("FSDP is not supported anymore. "
                           "There were a lot of isses with ReLoRA and FSDP "
                           "and no speed or memory improvements.")
        model: Union[FSDP, ReLoRaModel, LlamaForCausalLM] = training_utils.initialize_fsdp(model, dtype=args.dtype)

    elif args.distributed_type == "ddp":
        logger.info("Wrapping model with DDP")
        # 这部分代码是将模型包装成分布式数据并行(DistributedDataParallel, DDP)的形式
        # DDP是PyTorch中用于数据并行训练的一种方法,可以在多个GPU或多台机器上并行训练模型
        # 主要作用如下:
        # 1. 自动同步不同进程间的梯度,实现数据并行
        # 2. 优化通信,提高训练效率
        # 3. 确保模型在所有进程中保持一致
        
        # 参数说明:
        # model: 要包装的模型
        # device_ids: 指定当前进程使用的GPU设备ID
        # output_device: 指定输出结果的设备
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        ) # 把model模型包装起来 仍叫model

        _model = model.module 
        # 这行代码的意思是:
        # 当使用DistributedDataParallel (DDP)包装模型时,
        # 我们需要通过.module属性来访问原始的模型对象。
        # 这是因为DDP会在原始模型外部添加一层包装,
        # 而.module可以让我们绕过这层包装,直接访问到原始模型。
    else:
        _model = model

    # ##############################
    if args.wandb_watch and global_rank == 0:
        _log_freq = 500
        logger.info(f"Tracking model gradients with wandb every {_log_freq} update steps")
        wandb.watch(model, log_freq=_log_freq)

    # Computing the number of parameters is done before wrapping the model with FSDP
    # but gettint the parameters for optimization is done after. This is intentional and doing it other way causes errors.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
    trainable_params_names = [name for name, p in model.named_parameters() if p.requires_grad]

    if args.use_peft and len(lora_params) == 0:
        raise ValueError("No LoRA parameters found")

    # Initialize wandb
    run_config = dict(vars(args))
    run_config.update({
        "tokenizer": dataset_preprocessing_args["tokenizer"],
        "max_lr": run_config.pop("lr"),  # rename lr to max_lr to avoid conflicts with scheduler
        "total_params_M": n_total_params / 1_000_000,
        "trainable_params_M": n_trainable_params / 1_000_000,
        "equivalent_params_M": params_before / 1_000_000,
        "percent_trainable_params": p_trainable_params,
        "name_trainable_params": trainable_params_names,
        "model": model_config.to_dict(),
        "world_size": world_size,
        "device": str(device),
        "dataset_preprocessing_args": dataset_preprocessing_args,
    })

    if global_rank == 0:
        wandb.config.update(run_config, allow_val_change=True)
        wandb.save(os.path.abspath(__file__), policy="now") # save current script

    # 配置optimizer 和 lr
    optimizer_state_keys = None
    optimizer_kwargs = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "betas": (args.adam_beta1, args.adam_beta2),
    }
    if args.optimizer.lower() == "adam":
        logger.info("Using Adam optimizer")
        optimizer = torch.optim.AdamW(trainable_params, **optimizer_kwargs)
        optimizer_state_keys = ["exp_avg", "exp_avg_sq"]
    elif args.optimizer.lower() == "adam_zero":
        logger.info("Using Adam optimizer with ZeRO")
        optimizer = ZeroRedundancyOptimizer(
            trainable_params,
            optimizer_class=torch.optim.AdamW,
            **optimizer_kwargs,
        )
        optimizer_state_keys = ["exp_avg", "exp_avg_sq"]
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")

    scheduler_start_step = update_step # 记录调度器开始迭代时的步数 如果是从头开始训练 则update_step等于0
    # 如果从checkpoint恢复训练 则update_step不等于0 此时scheduler_start_step记录的是从checkpoint恢复前的最后一次参数更新时的步数
    _scheduler_steps = args.num_training_steps - scheduler_start_step # 调度器需要迭代的步数
    logger.info(f"Scheduler will run for {_scheduler_steps} update steps")
    scheduler = training_utils.get_scheculer(
        optimizer=optimizer,
        scheduler_type=args.scheduler,
        num_training_steps=_scheduler_steps,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        cycle_length=args.cycle_length,
        restart_warmup_steps=args.restart_warmup_steps,
        adjust_step=args.adjust_step,
    ) # 生成调度器

    if args.resume_from:
        logger.info("Setting scheduler to the same state as in the checkpoint")
        for _ in range(update_step):
            scheduler.step()
        logger.info(f"Scheduler state restored from {args.resume_from}")
        # current lr
        logger.info(f"Current lr is {optimizer.param_groups[0]['lr']}")

        if args.load_optimizer_state_on_resume:
            _optimizer_dir = args.resume_from
            optimizer_checkpoint = torch.load(os.path.join(_optimizer_dir, "optimizer.pt"), map_location="cpu")
            optimizer.load_state_dict(optimizer_checkpoint["optimizer"]) # 从checkpoint加载优化器状态
            scheduler.load_state_dict(optimizer_checkpoint["scheduler"]) # 从checkpoint加载调度器状态
            update_step = optimizer_checkpoint["update_step"]
            global_step = optimizer_checkpoint["global_step"]
            logger.info(f"Optimizer and scheduler restored from {_optimizer_dir}")

        # check that batch_size did not change or dataloader rewinding won't work
        _training_config_path = os.path.join(args.resume_from, "training_config.yaml")
        if os.path.exists(_training_config_path):
            with open(_training_config_path) as f:
                _old_training_config = yaml.safe_load(f)
            if args.batch_size != _old_training_config["batch_size"]:
                raise RuntimeError("Cannot resume from a checkpoint with a different batch size.")

    if args.dataset_path is not None:
        # Huggingface dataset to dataloader
        logger.info(f"Full training set size: {len(train_dataset)}")
        logger.info(repr(train_dataset))
        train_dataset = datasets.distributed.split_dataset_by_node(train_dataset, rank=global_rank, world_size=world_size)
        eval_dataset = datasets.distributed.split_dataset_by_node(eval_dataset, rank=global_rank, world_size=world_size)
        logger.info(f"Train set size after shard: {len(train_dataset)}")

        _skip_batches = update_step * args.gradient_accumulation
        logger.info(f"Skipping the first {_skip_batches} batches")
        train_loader = SkipDataLoader(
            train_dataset,
            batch_size=args.batch_size,
            collate_fn=default_data_collator,
            skip_batches=_skip_batches,
            num_workers=args.workers,
        )
        eval_loader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            collate_fn=default_data_collator,
            num_workers=args.workers,
        )
        test_loader = None
    else:
        # Megatron dataset to dataloader
        # Initialized earlier in the script
        assert args.megatron_dataset_config is not None
        assert train_loader is not None
        assert eval_loader is not None

    # global steps and others are defined above
    update_time = time.time()
    local_step = 0  # when warmed_up_model is used, local_step != global_step
    loss_info = torch.tensor([0.0, 0.0, 0.0], device=device)  # loss, n_batches, n_NaNs
    n_skipped_batches = 0

    # ###########################################################################################################
    # TRAINING LOOP 包括 前向传播、损失计算、反向传播和参数更新
    # we assert above that the dataset is large enough to train for num_training_steps, so no need for epochs
    # ###########################################################################################################

    prof = maybe_make_profiler(args)

    logger.info(f"Starting training at update step {update_step} with {args.num_training_steps - update_step} update steps")
    if global_rank == 0:
        # fix tqdm visual length to 80 so that the progress bar
        # doesn't jump around when changing from external display to laptop
        pbar = tqdm(total=args.num_training_steps - update_step, desc="Update steps", ncols=80)

    for batch in train_loader:
        global_step += 1
        local_step += 1

        if update_step in args.skip_batches:
            if global_step % args.gradient_accumulation == 0:
                update_step += 1
            continue

        if local_step == 1: logger.info(f"Starting first step")
        if update_step >= args.num_training_steps:
            logger.info(f"Reached max number of update steps (f{args.num_training_steps}). Stopping training.")
            print(f"Rank {global_rank} stopping training.")
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        tokens_seen += batch["input_ids"].numel() * world_size

        loss = model(**batch, labels=batch["input_ids"]).loss # forward 并 计算loss; model.loss

        loss_info[0] += loss.detach()
        loss_info[1] += 1
        loss_info[2] += torch.isnan(loss).float()

        if global_step == 0 and global_rank == 0:
            # log loss without any optimization
            wandb.log({"loss": loss.item(), "update_step": 0}, step=0)

        scaled_loss = loss / args.gradient_accumulation
        scaled_loss.backward() # backward 计算loss对参数的梯度

        if global_step % args.gradient_accumulation != 0: # 如果用了梯度累积 则累积梯度
            continue

        # The below code is only executed during the update step
        if global_rank == 0: pbar.update(1)

        if args.clip_grad_norm > 0: # 如果设置了梯度剪裁 则进行gradient clip以防止梯度爆炸
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.clip_grad_norm, error_if_nonfinite=True)
            if global_rank == 0:
                wandb.log({"grad_norm": grad_norm.item()}, step=global_step)

        dist.all_reduce(loss_info, op=dist.ReduceOp.SUM)
        _loss = loss_info[0] / loss_info[1]  # loss to log in wandb below

        if loss_info[2] == 0:  # no NaNs, update model
            optimizer.step() # 参数更新
            scheduler.step() # 更新学习率
        else:
            logger.error(f"Nan detected in loss_info, {_loss=}, skipping update")
            n_skipped_batches += 1

            if n_skipped_batches > 0.05 * args.num_training_steps:
                logger.error(f"More than 5% of batches skipped due to NaNs, stopping training.")
                break

        optimizer.zero_grad() # 梯度归零
        update_step += 1
        update_time = time.time() - update_time

        loss_info = torch.zeros_like(loss_info)

        """保存model 和 训练状态 这是在training loop里的保存"""
        if local_step > args.gradient_accumulation and update_step % args.save_every == 0:
            logger.info(f"Saving condition met: local_step={local_step}, args.gradient_accumulation={args.gradient_accumulation}, update_step={update_step}, args.save_every={args.save_every}")
            current_model_directory = f"{args.save_dir}/model_{update_step}"
            logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
            
            # 使用 _model 而不是 model
            if _model.config.pad_token_id is None or _model.config.pad_token_id == -1:
                logger.warning("pad_token_id is not set. Setting it to eos_token_id.")
                _model.config.pad_token_id = _model.config.eos_token_id

            if hasattr(_model, 'generation_config'):
                if _model.generation_config.pad_token_id is None or _model.generation_config.pad_token_id == -1:
                    logger.warning("generation_config.pad_token_id is not set. Setting it to config.pad_token_id.")
                    _model.generation_config.pad_token_id = _model.config.pad_token_id

            training_state_checkpoint = {
                "global_step": global_step,
                "update_step": update_step,
                "tokens_seen": tokens_seen,
                "tokens_seen_before": tokens_seen_before,
                "n_lora_restarts": n_lora_restarts,
                "n_optimizer_resets": n_optimizer_resets,
                "update_time": update_time,
            }

            # 然后再保存模型 在training loop中
            save_model(
                _model, # 使用 _model 而不是 model
                optimizer=optimizer,
                scheduler=scheduler,
                training_state_checkpoint=training_state_checkpoint,
                run_config=run_config,
                distributed_type=args.distributed_type,
                save_dir=current_model_directory,
            )
            if args.keep_checkpoints is not None:
                training_utils.delete_old_checkpoints(args.save_dir, keep=args.keep_checkpoints)
        # else:
            # logger.info(f"Saving condition not met: local_step={local_step}, args.gradient_accumulation={args.gradient_accumulation}, update_step={update_step}, args.save_every={args.save_every}")

        # ##############################
        # EVALUATION
        if update_step % args.eval_every == 0:
            logger.info(f"Performing evaluation at step {update_step}")
            total_loss, evaluated_on_tokens = evaluate_model(model, eval_loader, device)

            if global_rank == 0:
                wandb.log({
                    "final_eval_loss": total_loss,
                    "final_eval_tokens": evaluated_on_tokens,
                    },
                    step=global_step,
                )
            logger.info(f"Eval loss at step {update_step}: {total_loss}")
        # ##############################

        # ##############################
        # MERGE AND REINIT

        # restart model after we modify the learning rate, so on the next step after the relora frequency
        can_reset_relora = args.relora is not None and (
            args.resume_from is not None
            or local_step // args.gradient_accumulation >= args.relora # 如果当前步数除以梯度累积的个数大于等于relora 则可以进行lora重置
        )

        if can_reset_relora and (update_step - scheduler_start_step) % args.relora == 1: # 如果可以进行lora重置 并且是lora重置的第一个step
            _lora_reset_time = time.time()
            logger.info(f"{args.resume_from=}, {local_step=}, {args.relora=}, thresh: {local_step // args.gradient_accumulation}")
            logger.info(f"Performing lora reset at update step {update_step}. Current lr is {optimizer.param_groups[0]['lr']}")
            n_lora_restarts += 1

            if args.distributed_type == "ddp":
                model.module.merge_and_reinit() # ddp时用merge_and_reinit()
            elif args.distributed_type == "fsdp":
                model.apply(merge_and_reinit_functional) # fsdp时用merge_and_reinit_functional
            else:
                raise ValueError(f"Unknown distributed type {args.distributed_type}")
            
            _lora_reset_time = time.time() - _lora_reset_time
            logger.info(f"LoRA reset took {_lora_reset_time:.2f}s")

        can_reset_optimizer = args.relora is not None and (
            args.resume_from is not None
            or local_step // args.gradient_accumulation >= args.cycle_length
        )

        if can_reset_optimizer and (update_step - scheduler_start_step) % args.cycle_length == 1: # 如果可以进行优化器重置 并且是优化器重置的第一个step
            # scheduler should provide a new warmup after the reset
            logger.info(f"Performing optimizer reset at update step {update_step}. Current lr is {optimizer.param_groups[0]['lr']}")
            n_optimizer_resets += 1

            training_utils.optimizer_reset(
                optimizer,
                reset_params=lora_params,
                optimizer_state_keys=optimizer_state_keys,
                reset_optimizer_on_relora=args.reset_optimizer_on_relora,
                optimizer_random_pruning=args.optimizer_random_pruning,
                optimizer_magnitude_pruning=args.optimizer_magnitude_pruning,
            )
        # ##############################

        if can_reset_optimizer and (update_step - scheduler_start_step) % args.cycle_length == 2: # 如果可以进行优化器重置 并且是优化器重置的第二个step
            logger.info(f"First step after optimizer reset lr is {optimizer.param_groups[0]['lr']}") # 打印当前的学习率

        lr = optimizer.param_groups[0]["lr"] # 获取当前的学习率
        tokens_in_update = tokens_seen - tokens_seen_before # 计算当前更新步数内的token数
        tokens_seen_before = tokens_seen # 更新tokens_seen_before
        batches_in_update = args.gradient_accumulation * world_size # 计算当前更新步数内的batch数

        if global_rank == 0:
            wandb.log({
                "loss": _loss,
                "lr": lr,
                "update_step": update_step,
                "tokens_seen": tokens_seen,
                "throughput_tokens": tokens_in_update / update_time,
                "throughput_examples": args.total_batch_size / update_time,
                "throughput_batches": batches_in_update / update_time,
                "n_lora_restarts": n_lora_restarts,
                "n_optimizer_resets": n_optimizer_resets,
                },
                step=global_step,
            )
            if args.train_scaling:
                all_scaling_factors = []
                for module in model.modules():
                    if isinstance(module, ReLoRaLinear):
                        all_scaling_factors.append(module.scaling.data.item())
                wandb.log({"lora_scaling": torch.tensor(all_scaling_factors)}, step=global_step)
        update_time = time.time()
        if prof is not None: prof.step()
    else: # for-else statement
        print(f"Warning: reached the end of the dataset. Training stopped, {global_rank=}, {update_step=}")
        logger.warning("Reached the end of the dataset. Training stopped")

    if prof is not None: prof.stop()
    # ###########################################################################################################
    # END of training loop
    # ###########################################################################################################
    logger.info("Training finished")
    if global_rank == 0: pbar.close()

    """保存model 和 训练状态 这是在training loop外的保存"""
    current_model_directory = f"{args.save_dir}/model_{update_step}"
    if not os.path.exists(current_model_directory):
        logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
        # 使用 _model 而不是 model
        if _model.config.pad_token_id is None or _model.config.pad_token_id == -1:
            logger.warning("pad_token_id is not set. Setting it to eos_token_id.")
            _model.config.pad_token_id = _model.config.eos_token_id

        if hasattr(_model, 'generation_config'):
            if _model.generation_config.pad_token_id is None or _model.generation_config.pad_token_id == -1:
                logger.warning("generation_config.pad_token_id is not set. Setting it to config.pad_token_id.")

        training_state_checkpoint = {
            "global_step": global_step,
            "update_step": update_step,
            "tokens_seen": tokens_seen,
            "tokens_seen_before": tokens_seen_before,
            "n_lora_restarts": n_lora_restarts,
            "update_time": update_time,
        }
        save_model(
            _model, # 使用 _model 而不是 model
            optimizer=optimizer,
            scheduler=scheduler,
            training_state_checkpoint=training_state_checkpoint,
            run_config=run_config,
            distributed_type=args.distributed_type,
            save_dir=current_model_directory,
        )

    # Final evaluation 最终评估
    logger.info("Running final evaluation")
    model.eval()
    del loss, optimizer, scheduler
    import gc; gc.collect()
    torch.cuda.empty_cache()

    total_loss, evaluated_on_tokens = evaluate_model(
        model, eval_loader, device,
        target_eval_tokens=100_000_000,
    )

    if global_rank == 0:
        wandb.log({
            "final_eval_loss": total_loss,
            "final_eval_tokens": evaluated_on_tokens,
            },
            step=global_step,
        )
        logger.info(f"Final eval loss: {total_loss}")

    if test_loader is not None:
        logger.info("Running test evaluation (full test set!)")
        total_loss, evaluated_on_tokens = evaluate_model(
            model, test_loader, device,
            target_eval_tokens=-1,
        )

        if global_rank == 0:
            wandb.log({
                "final_test_loss": total_loss,
                "final_test_tokens": evaluated_on_tokens,
                },
                step=global_step,
            )
            logger.info(f"Test loss: {total_loss}")

    if global_rank == 0:
        wandb.finish() # 结束

    logger.info("Script finished successfully")
    print(f"Rank {global_rank} finished successfully")


if __name__ == "__main__": 
    args = parse_args()
    main(args)
