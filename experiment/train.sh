#!/bin/bash
#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=10-00:00:00
#SBATCH --nodes=1 
#SBATCH --mem=300GB
#SBATCH --cpus-per-task=8      
#SBATCH --gres=gpu:a6000:2
#SBATCH --job-name=job1
#SBATCH --exclude=iris4,iris-hp-z8
#SBATCH --mail-type=END
#SBATCH --mail-user=jleng3@u.rochester.edu
#SBATCH --output=/iris/u/huaxiu/denoise_rlhf/UroSAM/job_output_SAM.txt

cd /iris/u/huaxiu/Alzheimer/Alzheimer/
echo "CUDA Version:"
nvidia-smi | grep -o "CUDA Version: [0-9.]*"
source /iris/u/huaxiu/virtual_env_list/rlhf/bin/activate

OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./Vanilla_FT_keep_5
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=0
fi
mkdir -p $OUTPUT

deepspeed train.py\
    --data_path ./data/alzheimer/HCP_WM \
    --weight_decay 0.1 --gradient_accumulation_steps 4 --zero_stage $ZERO_STAGE \
    --tensorboard_path $OUTPUT \
    --learning_rate 1e-5 \
    --deepspeed \
    --per_device_train_batch_size 2048 \
    --per_device_eval_batch_size 2048 \
    --num_train_epochs 200 \
    --test_env 0 \
    --input_dim 268 \
    --hidden_dim 268 \
    --mlp_dim 1072 \
    --num_heads 4 \
    --layers 12 \
    --seq_len 64 \
    --downsample_rate 1 \
    --dropout_rate 0.0 \
    --num_samples_per_class 50 \
    --update \
    --enable_tensorboard \
    --output_dir $OUTPUT 2> >(tee $OUTPUT/err.log >&2) | tee $OUTPUT/training.log \
    # skip_5 \


OUTPUT2=./Vanilla_FT
mkdir -p $OUTPUT2
deepspeed train.py\
    --data_path ./data/alzheimer/HCP_WM \
    --weight_decay 0.1 --gradient_accumulation_steps 4 --zero_stage $ZERO_STAGE \
    --tensorboard_path $OUTPUT \
    --learning_rate 1e-5 \
    --deepspeed \
    --per_device_train_batch_size 2048 \
    --per_device_eval_batch_size 2048 \
    --num_train_epochs 200 \
    --test_env 0 \
    --input_dim 268 \
    --hidden_dim 268 \
    --mlp_dim 1072 \
    --num_heads 4 \
    --layers 12 \
    --seq_len 64 \
    --downsample_rate 1 \
    --dropout_rate 0.0 \
    --num_samples_per_class 50 \
    --update \
    --skip_5 \
    --enable_tensorboard \
    --output_dir $OUTPUT2 2> >(tee $OUTPUT2/err.log >&2) | tee $OUTPUT2/training.log \
