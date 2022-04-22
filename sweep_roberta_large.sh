#!/bin/bash

FAIRSEQ_USER="${USER}"

FAIRSEQ_DIR="${HOME}/fairseq-py"
# Note: you will need to install fairseq, activate the conda env, load modules
# conda activate fairseq-20200821 (from https://fb.workplace.com/groups/fairseq/permalink/262715387865587/)
# module load anaconda3/2020.11 cudnn/v8.0.3.33-cuda.11.0 cuda/11.0 openmpi/4.1.0/cuda.11.0-gcc.9.3.0
# cd $FAIRSEQ_DIR
# git checkout <branch>
# pip install --editable .

# partition="learnaccel"
# num_trials=-1
# num_gpus_per_node=8
# num_experts_per_gpu=1
type=masking_strategies
# type=second_run
# type=giantFFN
# type=debug

wiki103_dir=/private/home/zeyuliu/dataset/wikitext-103/data-bin
roberta_dir=/datasets01/bookwiki_CC-NEWS_openwebtext_stories-mmap2-bin/121219/bookwiki_CC-NEWS_openwebtext_stories-mmap2-bin
checkpoint_dir="/checkpoint/${FAIRSEQ_USER}/en_dense_lm/${type}/"

PYTHONPATH=. ./fb_sweep/sweep_roberta_large.py \
    --data $roberta_dir \
    --num-trials -1 \
    --num-gpus 1 \
    --checkpoints-dir ${checkpoint_dir} \
    --prefix roberta.large \
    --num-nodes 1 \
    --save-interval 500 \
    --partition learnaccel \
    --constraint volta32gb \
    --snapshot-code \
    --benchmark \
    --dry-run
    # --resume-failed
    # --data $wiki103_dir \
    # --benchmark \
    # --bs 8 \
