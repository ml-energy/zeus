#!/bin/bash

MODEL=${1:-bert}
MICROBATCH_SIZE=${2:-4}
NUM_MICROBATCH=${3:-8}
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}
shift 3

CACHE_DIR="/data/jwnchung/merak_cache/"
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}
PERSEUS_PORT=${PERSEUS_PORT:-7787}
PERSEUS_URL="http://$MASTER_ADDR:$PERSEUS_PORT"

export NCCL_ASYNC_ERROR_HANDLING=1

cd /workspace/merak/examples

if [[ $MODEL = t5* ]]; then
  echo $MODEL
  cd language-modeling
  python -m torch.distributed.launch --nnodes=$NNODES --nproc_per_node=$NUM_GPUS --node_rank=$NODE_RANK \
                  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT run_t5.py \
                  --model_name $MODEL \
                  --cache_dir $CACHE_DIR/$MODEL \
                  --output_dir ./output \
                  --num_train_epochs 1000 \
                  --per_device_train_batch_size $MICROBATCH_SIZE --gradient_accumulation_steps $NUM_MICROBATCH \
                  --perseus_url $PERSEUS_URL \
                  $@

elif [[ $MODEL = gpt* ]]; then
  echo $MODEL
  cd language-modeling
  python -m torch.distributed.launch --nnodes=$NNODES --nproc_per_node=$NUM_GPUS --node_rank=$NODE_RANK \
                  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT run_gpt.py \
                  --model_name $MODEL \
                  --cache_dir $CACHE_DIR/$MODEL \
                  --output_dir ./output \
                  --num_train_epochs 100 \
                  --per_device_train_batch_size $MICROBATCH_SIZE --gradient_accumulation_steps $NUM_MICROBATCH \
                  --perseus_url $PERSEUS_URL \
                  $@

elif [[ $MODEL = bloom* ]]; then
  echo $MODEL
  cd language-modeling
  python -m torch.distributed.launch --nnodes=$NNODES --nproc_per_node=$NUM_GPUS --node_rank=$NODE_RANK \
                  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT run_bloom.py \
                  --model_name $MODEL \
                  --cache_dir $CACHE_DIR/$MODEL \
                  --output_dir ./output \
                  --num_train_epochs 100 \
                  --per_device_train_batch_size $MICROBATCH_SIZE --gradient_accumulation_steps $NUM_MICROBATCH \
                  --perseus_url $PERSEUS_URL \
                  $@

elif [[ $MODEL = bert* ]]; then
  echo $MODEL
  cd language-modeling
  python -m torch.distributed.launch --nnodes=$NNODES --nproc_per_node=$NUM_GPUS --node_rank=$NODE_RANK \
                  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT run_bert.py \
                  --model_name $MODEL \
                  --cache_dir $CACHE_DIR/$MODEL \
                  --output_dir ./output \
                  --remove_unused_columns false \
                  --num_train_epochs 100 \
                  --per_device_train_batch_size $MICROBATCH_SIZE --gradient_accumulation_steps $NUM_MICROBATCH \
                  --perseus_url $PERSEUS_URL \
                  $@

elif [[ $MODEL = wide-resnet* ]]; then
  echo $MODEL
  cd torch-models
  IMAGENET_DIR=${IMAGENET_DIR-"/data/imagenet"}
  if [[ -d "$IMAGENET_DIR" ]]; then
    echo Found ImageNet at $IMAGENET_DIR!
  else
    echo Error: Directory $IMAGENET_DIR does not exist.
    echo Please download ImageNet following instructions in https://github.com/pytorch/examples/tree/main/imagenet and set the environment variable 'IMAGENET_DIR' to the directory that contains the ImageNet dataset.
    exit 1
  fi
  python -m torch.distributed.launch --nnodes=$NNODES --nproc_per_node=$NUM_GPUS --node_rank=$NODE_RANK \
                  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT run_torchvision.py \
                  --model_name $MODEL \
                  --cache_dir $CACHE_DIR/$MODEL \
                  --output_dir ./output \
                  --cfg ./torchvision.yaml \
                  --data_path $IMAGENET_DIR \
                  --num_train_epochs 1000 \
                  --per_device_train_batch_size $MICROBATCH_SIZE --gradient_accumulation_steps $NUM_MICROBATCH \
                  --perseus_url $PERSEUS_URL \
                  --cache_sharding False \
                  --dataloader_num_workers 4 \
                  $@

else
  echo Unknown model name: $MODEL
  exit 1
fi
