import torch
from torch import nn
import numpy as np
import os
import sys
from model.transformer import Transformer
import data.dataset as dataset
from collections import defaultdict
from utils.misc import print_rank_0, to_device
import logging
import deepspeed
import argparse
from transformers import (
    SchedulerType,
    get_scheduler,
)
from utils.ds_utils import get_train_ds_config
from utils.misc import get_all_reduce_mean
import utils.misc as misc
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
import math
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Eval a transformers model on Alzheimer's dataset.")
    parser.add_argument('--data_path',
                        type=str,
                        default='segmentation_data/',
                        help='Path to the seg training dataset')
    parser.add_argument('--model_name_or_path',
                        type=str,
                        default='Vanilla_FT/',
                        required=True,
                        help='Path to the seg training dataset')
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument(
        '--gradient_checkpointing',
        action='store_true',
        help='Enable HF gradient checkpointing for Actor model.')
    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    parser.add_argument(
        "--lora_learning_rate",
        type=float,
        default=5e-4,
        help=
        "Initial LoRA learning rate (after the potential warmup period) to use."
    )
    ## Tensorboard logging
    parser.add_argument('--enable_tensorboard',
                        action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--tensorboard_path',
                        type=str,
                        default="step2_tensorboard")
    parser.add_argument("--hidden_dim",
                        type=int,
                        default=512,
                        help="hidden dimension of transformer model")
    parser.add_argument("--input_dim",
                        type=int,
                        default=256,
                        help="hidden dimension of mlp")
    parser.add_argument("--mlp_dim",
                        type=int,
                        default=256,
                        help="hidden dimension of mlp")
    parser.add_argument("--num_heads",
                        type=int,
                        default=8,
                        help="number of attention heads")
    parser.add_argument("--layers",
                        type=int,
                        default=12,
                        help="number of transformer layers")
    parser.add_argument("--seq_len",
                        type=int,
                        default=512,
                        help="maximum sequence length")
    parser.add_argument("--dropout_rate",
                        type=float,
                        default=0.0,
                        help="dropout rate")
    parser.add_argument("--downsample_rate",
                        type=int,
                        default=1,
                        help="downsample rate")
    parser.add_argument("--test_env",
                        type=int,
                        default=1,
                        help="test env")
    parser.add_argument('--before',
                        action='store_true',
                        help='Load ckpt before ft.')
    parser.add_argument('--skip_5',
                        action='store_true',
                        help='Skip label 5')
    parser.add_argument('--update',
                            action='store_true',
                            help='reprocess data')
    parser.add_argument("--num_samples_per_class",
                        type=int,
                        default=50,
                        help="test samples for ft")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args



def load_stuff(args, num_class, before):
    model = Transformer(
        input_dim = args.input_dim,
        hidden_dim=args.hidden_dim,
        mlp_dim=args.mlp_dim,
        layers=args.layers,
        num_heads=args.num_heads,
        seq_len=args.seq_len,
        num_classes=num_class,
        drop_out=args.dropout_rate,
        downsample_rate=args.downsample_rate,
    )
    WEIGHTS_NAME = "pytorch_model.bin" if not before else 'pytorch_model_before_ft.bin' 
    output_model_file = os.path.join(args.model_name_or_path, WEIGHTS_NAME)
    checkpoint = torch.load(output_model_file)
    model.load_state_dict(checkpoint)
    print('load model from', output_model_file)
    return model


def eval(model, loader, device, args, name = 'eval'):
    model.eval()
    total_correct = 0
    total_samples = 0
    class_correct = defaultdict(int)
    class_samples = defaultdict(int)

    true_labels = []
    predicted_labels = []

    # check if loader.dataset is Subset
    if hasattr(loader.dataset, 'dataset'):
        reverse_mapping = {v: k for k, v in loader.dataset.dataset.label_mapping.items()}
    else:
        reverse_mapping = {v: k for k, v in loader.dataset.label_mapping.items()}

    with torch.no_grad():
        for step, batch in enumerate(loader):
            batch = to_device(batch, device)
            inputs, attention_mask, labels = batch['data'], batch['attention_mask'], batch['labels']
            outputs = model(inputs, attention_mask)
            preds = torch.argmax(outputs, dim=1)
            preds = preds.detach().cpu()
            labels = labels.detach().cpu()

            true_labels.extend(labels)
            predicted_labels.extend(preds)
            
            
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            for label, pred in zip(labels, preds):
                class_correct[label.item()] += (pred == label).item()
                class_samples[label.item()] += 1

    overall_accuracy = total_correct / total_samples
    per_class_accuracy = {cls: class_correct[cls] / class_samples[cls]
                          for cls in class_samples}
    per_class_accuracy = {reverse_mapping[cls]: per_class_accuracy[cls] for cls in per_class_accuracy}
    
    try:
        overall_accuracy = get_all_reduce_mean(torch.tensor(overall_accuracy)).item()
        per_class_accuracy = {k: get_all_reduce_mean(torch.tensor(v)).item() for k, v in per_class_accuracy.items()}
    except:
        pass

    model.train()

    # draw confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    class_names = [reverse_mapping[i] for i in range(len(reverse_mapping))]
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks + 0.5, class_names, rotation=0)
    plt.yticks(tick_marks + 0.5, class_names)
    plt.savefig(f'{args.output_dir}/{name}_confusion_matrix.png')

    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm_percentage, annot=True, fmt='.2%', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    class_names = [reverse_mapping[i] for i in range(len(reverse_mapping))]
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks + 0.5, class_names, rotation=0)
    plt.yticks(tick_marks + 0.5, class_names)
    plt.savefig(f'{args.output_dir}/{name}_confusion_matrix_percentage.png')

    return {
        "overall_accuracy": overall_accuracy,
        "per_class_accuracy": per_class_accuracy
    }


def main():
    args = parse_args()
    print(args)
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        deepspeed.init_distributed()
    print('device', device)

    print('--------------Test Env: ', args.test_env, '-----------------')
    # train_set = dataset.AlzheimerDataset(args.data_path, args.test_env, args.seq_len)
    # eval_set = dataset.AlzheimerDataset(args.data_path, 1 - args.test_env, args.seq_len)
    train_set = dataset.HCPWMDataset(args.data_path, args.test_env, args.seq_len, args.update, args)
    eval_set = dataset.HCPWMDataset(args.data_path, 1 - args.test_env, args.seq_len, args.update, args)
    eval_indices = np.load(f'{args.model_name_or_path}/remaining_indices.npy')
    eval_set = torch.utils.data.Subset(eval_set, eval_indices)
    num_class = train_set.get_class_number()
    has_intersection, intersection = misc.check_dataset_intersection(train_set, eval_set.dataset)
    if has_intersection:
        print('Intersection: ', intersection)
    else:
        print('No intersection')

    model = load_stuff(args, num_class, before = args.before)
    model.to(device)
    model.eval()
    if args.local_rank == -1:
        train_sampler = SequentialSampler(train_set)
        eval_sampler = SequentialSampler(eval_set)
    else:
        train_sampler = DistributedSampler(train_set)
        eval_sampler = DistributedSampler(eval_set)
    eval_dataloader = DataLoader(eval_set,
                                sampler=eval_sampler,
                                batch_size=args.per_device_eval_batch_size, num_workers=8)
    train_dataloader = DataLoader(train_set,
                                sampler=train_sampler,
                                batch_size=args.per_device_train_batch_size, num_workers=8)
    
    eval_result = eval(model, eval_dataloader, device, args, name = 'eval')

    print('=========> Evaluation Result <=========')
    print('len eval set', len(eval_set))
    print('Overall Accuracy: ', eval_result['overall_accuracy'])
    print('Per Class Accuracy: ', eval_result['per_class_accuracy'])

    print('len train set', len(train_set))
    train_result = eval(model, train_dataloader, device, args, name = 'train')
    print('=========> Training Result <=========')
    print('Overall Accuracy: ', train_result['overall_accuracy'])
    print('Per Class Accuracy: ', train_result['per_class_accuracy'])


if __name__ == "__main__":
    main()