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
import torch.autograd as autograd
from sklearn.linear_model import LogisticRegression
import torch
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on Alzheimer's dataset.")
    parser.add_argument('--data_path',
                        type=str,
                        default='segmentation_data/',
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
                        default="ft_tensorboard")
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


def extract_features_and_labels(model, dataloader, device):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            batch = to_device(batch, device)
            inputs, attention_mask, label = batch['data'], batch['attention_mask'], batch['labels']
            outputs = model.forward_features(inputs, attention_mask)
            features.append(outputs.detach().cpu())
            labels.append(label.detach().cpu())

    features = torch.cat(features, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    return features, labels


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def eval(model, loader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    class_correct = defaultdict(int)
    class_samples = defaultdict(int)

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
    return {
        "overall_accuracy": overall_accuracy,
        "per_class_accuracy": per_class_accuracy
    }


def main():
    args = parse_args()
    if args.local_rank == -1:
        device = torch.device('cuda')
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()
    ds_config = get_train_ds_config(
        offload=args.offload,
        stage=args.zero_stage,
        enable_tensorboard=args.enable_tensorboard,
        tb_path=args.tensorboard_path,
        tb_name='before_ft',
    )
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
        ) * args.gradient_accumulation_steps
    
    misc.set_random_seed(args.seed)
    torch.distributed.barrier()

    # Dataset
    #train_set = dataset.AlzheimerDataset(args.data_path, args.test_env, args.seq_len)
    #eval_set = dataset.AlzheimerDataset(args.data_path, 1 - args.test_env, args.seq_len)
    print_rank_0(f"=================> Train Dataset Created", args.global_rank)
    train_set = dataset.HCPWMDataset(args.data_path, args.test_env, args.seq_len, args.update, args)
    print_rank_0(f"=================> Eval Dataset Created", args.global_rank)
    eval_set = dataset.HCPWMDataset(args.data_path, 1 - args.test_env, args.seq_len, args.update, args)
    print_rank_0(f'f============> full eval set length {len(eval_set)}', args.global_rank)
    original_eval_set = eval_set
    # get retrain classifier set
    selected_indices = misc.extract_samples_per_class(eval_set, num_samples_per_class=args.num_samples_per_class)
    retrain_set = torch.utils.data.Subset(eval_set, selected_indices)
    print_rank_0(f'f============> retrain set length {len(retrain_set)}', args.global_rank)
    remaining_indices = list(set(range(len(original_eval_set))) - set(selected_indices))
    eval_set = torch.utils.data.Subset(original_eval_set, remaining_indices)
    print_rank_0(f'f============> eval set length {len(eval_set)}', args.global_rank)
    # save remaining indices
    np.save(os.path.join(args.output_dir, 'remaining_indices.npy'), remaining_indices)

    num_class = train_set.get_class_number()
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_set)
        eval_sampler = SequentialSampler(eval_set)
        retrain_sampler = SequentialSampler(retrain_set)
    else:
        train_sampler = DistributedSampler(train_set)
        eval_sampler = DistributedSampler(eval_set)
        retrain_sampler = DistributedSampler(retrain_set)

    train_loader = DataLoader(train_set,
                                sampler=train_sampler,
                                batch_size=args.per_device_train_batch_size,
                                num_workers=8)
    eval_loader = DataLoader(eval_set,
                                sampler=eval_sampler,
                                batch_size=args.per_device_eval_batch_size,
                                num_workers=8)
    retrain_loader = DataLoader(retrain_set,
                                sampler=retrain_sampler,
                                batch_size=args.per_device_eval_batch_size,
                                num_workers=8)
    
    # Model
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


    print_rank_0(f"--------Model Created----------------", args.global_rank)
    
    optimizer_grouped_parameters = misc.get_optimizer_grouped_parameters(
        model, args.weight_decay)
    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam

    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              betas=(0.9, 0.95))

    #In case fusedAdam does not work, I do not know why :(
    # optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
    #                           lr=args.learning_rate,
    #                           betas=(0.9, 0.95))
    
    num_update_steps_per_epoch = math.ceil(
        len(train_loader) / args.gradient_accumulation_steps)
    
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        args=args,
        config_params=ds_config,
        dist_init_required=True,
    )
    
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    save_dir = os.path.join(args.output_dir, 'before_ft')
    writer = SummaryWriter(save_dir)
    # Train
    print_rank_0("***** Running training *****", args.global_rank)
    print_rank_0(f"--------Num Classes----------------{num_class}", args.global_rank)
    print_rank_0(f'--------------Test Env: {args.test_env} -----------------', args.local_rank)
    print_rank_0(f"--------train_dataset length---------------- {len(train_set)}",  args.global_rank)
    print_rank_0(f"--------eval_dataset length----------------{len(eval_set)}", args.global_rank)
    print_rank_0(f"--------retrain_dataset length----------------{len(retrain_set)}", args.global_rank)
    print_rank_0(f"--------trainable parameters----------------{count_parameters(model)}", args.global_rank)
    eval_result = eval(model, eval_loader, device)
    print_rank_0(f"Epoch {0}/{args.num_train_epochs} with eval result {eval_result}", args.global_rank)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_loader)}",
            args.global_rank)
        model.train()
        mean_loss = 0.0
        for step, batch in enumerate(train_loader):
            batch = to_device(batch, device)
            inputs, attention_mask, labels = batch['data'], batch['attention_mask'], batch['labels']
            outputs = model(inputs, attention_mask = attention_mask)
            loss = criterion(outputs, labels)
            model.backward(loss)
            model.step()
            mean_loss += loss.item()
        print_rank_0(
            f"Epoch {epoch+1}/{args.num_train_epochs} with loss {mean_loss/(step+1)}", args.global_rank)
        model.tput_timer.update_epoch_count()
        eval_result = eval(model, eval_loader, device)
        writer.add_scalar('eval/overall_accuracy', eval_result['overall_accuracy'], epoch)
        print_rank_0(f"Epoch {epoch+1}/{args.num_train_epochs} with eval result {eval_result}", args.global_rank)
        train_result = eval(model, train_loader, device)
        writer.add_scalar('train/overall_accuracy', train_result['overall_accuracy'], epoch)
        print_rank_0(f"Epoch {epoch+1}/{args.num_train_epochs} with train result {train_result}", args.global_rank)

    if args.output_dir is not None:
        print_rank_0('saving model ...', args.global_rank)

    if args.global_rank == 0:
        WEIGHTS_NAME = "pytorch_model_before_ft.bin"
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), output_model_file)

    # I did not test this function !
    if args.zero_stage == 3:
        misc.save_zero_three_model(model,
                                args.global_rank,
                                args.output_dir,
                                zero_stage=args.zero_stage)


    """FT"""
    model = model.module if hasattr(model, 'module') else model
    ds_config = get_train_ds_config(
        offload=args.offload,
        stage=args.zero_stage,
        enable_tensorboard=args.enable_tensorboard,
        tb_path=args.tensorboard_path,
        tb_name='after_ft',
    )
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
        ) * args.gradient_accumulation_steps

    # reinitalize optimizer
    optimizer_grouped_parameters = misc.get_optimizer_grouped_parameters(
        model, args.weight_decay)
    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              betas=(0.9, 0.95))

    # optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
    #                           lr=args.learning_rate,
    #                           betas=(0.9, 0.95))

    num_update_steps_per_epoch = math.ceil(
        len(retrain_loader) / args.gradient_accumulation_steps)
    
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        args=args,
        config_params=ds_config,
        dist_init_required=True,
    )

    save_dir = os.path.join(args.output_dir, 'after_ft')
    writer = SummaryWriter(save_dir)
    print_rank_0(f"--------trainable parameters----------------{count_parameters(model)}", args.global_rank)
    print_rank_0(f"--------retrain_dataset length----------------{len(retrain_set)}", args.global_rank)
    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs} for retraining classifier, Total Micro Batches {len(retrain_loader)}",
            args.global_rank)
        model.train()
        mean_loss = 0.0
        for step, batch in enumerate(retrain_loader):
            batch = to_device(batch, device)
            inputs, attention_mask, labels = batch['data'], batch['attention_mask'], batch['labels']
            outputs = model(inputs, attention_mask = attention_mask)
            loss = criterion(outputs, labels)
            model.backward(loss)
            model.step()
            mean_loss += loss.item()
        print_rank_0(
            f"Epoch {epoch+1}/{args.num_train_epochs} with loss {mean_loss/(step+1)}", args.global_rank)
        model.tput_timer.update_epoch_count()
        eval_result = eval(model, eval_loader, device)
        writer.add_scalar('eval/overall_accuracy', eval_result['overall_accuracy'], epoch)
        print_rank_0(f"Epoch {epoch+1}/{args.num_train_epochs} with eval result {eval_result}", args.global_rank)
        train_result = eval(model, train_loader, device)
        writer.add_scalar('train/overall_accuracy', train_result['overall_accuracy'], epoch)
        print_rank_0(f"Epoch {epoch+1}/{args.num_train_epochs} with train result {train_result}", args.global_rank)


    r"""
    Regression to Retrain Classifier
    """

    # print_rank_0(f"--------retrain_dataset length----------------{len(retrain_set)}", args.global_rank)
    # model = model.module if hasattr(model, 'module') else model
    # train_features, train_labels = extract_features_and_labels(model, retrain_loader, device)
    # logreg = LogisticRegression(penalty='l1', C=10.0, solver="liblinear",
    #                                     class_weight={0: 1.0, 1: 1.0}, max_iter=1000)
    # logreg.fit(train_features, train_labels)
    # predictions = logreg.predict(train_features)
    # print_rank_0(f"--------train accuracy----------------{np.mean(predictions == train_labels)}", args.global_rank)
    # with torch.no_grad():
    #     model.classifier.weight = torch.nn.Parameter(torch.tensor(logreg.coef_, dtype = torch.float32).to(device))
    #     model.classifier.bias = torch.nn.Parameter(torch.tensor(logreg.intercept_, dtype = torch.float32).to(device))
    # eval_result = eval(model, eval_loader, device)
    # print_rank_0(f"Final with eval result {eval_result}", args.global_rank)
    # train_result = eval(model, train_loader, device)
    # print_rank_0(f"Final with train result {train_result}", args.global_rank)
    
    if args.output_dir is not None:
        print_rank_0('saving model ...', args.global_rank)

        if args.global_rank == 0:
            WEIGHTS_NAME = "pytorch_model.bin"
            output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), output_model_file)

        # I did not test this function !
        if args.zero_stage == 3:
            misc.save_zero_three_model(model,
                                  args.global_rank,
                                  args.output_dir,
                                  zero_stage=args.zero_stage)



if __name__ == "__main__":
    main()

        


    



    

