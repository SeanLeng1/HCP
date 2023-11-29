import torch
from torch import nn
import math
import warnings
import os
from transformers import set_seed
import random
import numpy as np
import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import hashlib
from collections import defaultdict

def hash_array(array):
    array_bytes = array.tobytes()
    return hashlib.sha256(array_bytes).hexdigest()

def check_dataset_intersection(train_set, eval_set):
    train_hashes = {hash_array(array) for array in train_set.data}
    eval_hashes = {hash_array(array) for array in eval_set.data}
    intersection = train_hashes & eval_hashes
    has_intersection = len(intersection) > 0
    
    return has_intersection, intersection

def extract_samples_per_class(dataset, num_samples_per_class=20):
    indices_per_class = defaultdict(list)
    class_counts = defaultdict(int)
    for idx in range(len(dataset)):
        input = dataset[idx]
        label = input['labels'].item()  
        indices_per_class[label].append(idx)
        class_counts[label] += 1
    selected_indices = []
    for cls, indices in indices_per_class.items():
        print(f"Class {cls} has {len(indices)} samples.")
        if len(indices) < num_samples_per_class:
            raise ValueError(f"Not enough samples for class {cls}.")
        selected = np.random.choice(indices, num_samples_per_class, replace=False)
        selected_indices.extend(selected)
        print(f"Selected {len(selected)} samples for class {cls}, {len(indices) - len(selected)} samples left.")
    return selected_indices

def print_rank_0(msg, rank=0):
    if rank <= 0:
        print(msg)

def to_device(batch, device, exclude_keys=[]):
    output = {}
    for k, v in batch.items():
        if k not in exclude_keys:
            try:
                output[k] = v.to(device)
            except:
                output[k] = v
    return output

def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor


def set_random_seed(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_optimizer_grouped_parameters(
    model,
    weight_decay,
    lora_lr=5e-4,
    no_decay_name_list=["bias", "LayerNorm.weight"],
    lora_name_list=["lora_right_weight", "lora_left_weight"],
):
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list)
                    and p.requires_grad and not any(nd in n
                                                    for nd in lora_name_list))
            ],
            "weight_decay":
            weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list)
                    and p.requires_grad and any(nd in n
                                                for nd in lora_name_list))
            ],
            "weight_decay":
            weight_decay,
            "lr":
            lora_lr
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n
                        for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
            0.0,
        },
    ]
    if not optimizer_grouped_parameters[1]["params"]:
        optimizer_grouped_parameters.pop(1)
    return optimizer_grouped_parameters



def _z3_params_to_fetch(param_list):
    return [
        p for p in param_list
        if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]

def save_zero_three_model(model_ema, global_rank, save_dir, zero_stage=0):
    zero_stage_3 = (zero_stage == 3)
    os.makedirs(save_dir, exist_ok=True)
    WEIGHTS_NAME = "pytorch_model.bin"
    output_model_file = os.path.join(save_dir, WEIGHTS_NAME)

    model_to_save = model_ema.module if hasattr(model_ema,
                                                'module') else model_ema
    if not zero_stage_3:
        if global_rank == 0:
            torch.save(model_to_save.state_dict(), output_model_file)
    else:
        output_state_dict = {}
        for k, v in model_to_save.named_parameters():

            if hasattr(v, 'ds_id'):
                with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([v
                                                                            ]),
                                                       enabled=zero_stage_3):
                    v_p = v.data.cpu()
            else:
                v_p = v.cpu()
            if global_rank == 0 and "lora" not in k:
                output_state_dict[k] = v_p
        if global_rank == 0:
            torch.save(output_state_dict, output_model_file)
        del output_state_dict