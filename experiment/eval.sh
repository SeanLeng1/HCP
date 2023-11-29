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
    OUTPUT=./eval_results
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=0
fi
mkdir -p $OUTPUT

deepspeed --num_gpus 1 eval.py\
    --data_path ./data/alzheimer/HCP_WM \
    --tensorboard_path $OUTPUT \
    --deepspeed \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --num_train_epochs 500 \
    --test_env 0 \
    --input_dim 268 \
    --hidden_dim 268 \
    --mlp_dim 1072 \
    --num_heads 4 \
    --layers 12 \
    --seq_len 64 \
    --downsample_rate 1 \
    --model_name_or_path Vanilla_FT \
    --skip_5 \
    --output_dir $OUTPUT 2> >(tee $OUTPUT/err.log >&2) | tee $OUTPUT/training.log \