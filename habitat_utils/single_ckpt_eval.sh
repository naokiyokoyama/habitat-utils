#!/bin/bash

# We need to fool habitat-baselines into thinking that this job is not distributed.
export SLURM_NTASKS=1

# The first argument is a list of checkpoint paths separated by __CKPT_SEP__
ckpt_list=$1
# The SLURM_PROCID environment variable is the index of the checkpoint we are evaluating.
ckpt_idx=$SLURM_PROCID

SLURM_CHECKPOINT_PATH=`python -c "print('${ckpt_list}'.split('__CKPT_SEP__')[${ckpt_idx}])"`
echo "Evaluating checkpoint ${SLURM_CHECKPOINT_PATH}"

ckpt_basename=$(basename $SLURM_CHECKPOINT_PATH)
basename_no_ext="${ckpt_basename%.*}"
#ckpt_parent_dir=$(dirname $SLURM_CHECKPOINT_PATH)
#ckpt_grandparent_dir=$(dirname $ckpt_parent_dir)
SLURM_LOG_PATH=${SLURM_LOG_DIR}/${basename_no_ext}.log

# Code will be added below this line that will use SLURM_CHECKPOINT_PATH and SLURM_LOG_PATH
# to evaluate the checkpoint.
