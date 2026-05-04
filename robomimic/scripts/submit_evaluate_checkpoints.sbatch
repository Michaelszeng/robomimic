#!/bin/bash
#SBATCH --job-name=evaluate_checkpoints
#SBATCH --account=locomotion
#SBATCH --partition=vision-shared-titanrtx
#SBATCH --exclude=freeman-titanrtx-2,freeman-titanrtx-1,oliva-titanrtx-3
#SBATCH --qos=shared-if-available
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --requeue

# Usage:
#   sbatch robomimic/scripts/submit_evaluate_checkpoints.sbatch <horizon> <n_action_steps>
#
# SLURM runner for `evaluate_checkpoints.sh`.


HORIZON="${1:?Usage: sbatch robomimic/scripts/submit_evaluate_checkpoints.sbatch <horizon> <n_action_steps>}"
N_ACTION_STEPS="${2:?Usage: sbatch robomimic/scripts/submit_evaluate_checkpoints.sbatch <horizon> <n_action_steps>}"

DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")
echo "DATE: $DATE"
echo "TIME: $TIME"
echo "HORIZON: $HORIZON"
echo "N_ACTION_STEPS: $N_ACTION_STEPS"

# ============================================================================
# Evaluation parameters
# ============================================================================
# Path to a .ckpt file or a directory of .ckpt files.

CHECKPOINT_PATH="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/robomimic/2_obs_tool_hang_film/checkpoints"
# CHECKPOINT_PATH="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/robomimic/2_obs_tool_hang_double_enc/checkpoints"
# CHECKPOINT_PATH="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/robomimic/4_obs_tool_hang_double_enc/checkpoints"
# CHECKPOINT_PATH="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/robomimic/8_obs_tool_hang_double_enc/checkpoints"
# CHECKPOINT_PATH="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/robomimic/16_obs_tool_hang_double_enc/checkpoints"
# CHECKPOINT_PATH="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/robomimic/20_obs_tool_hang_double_enc/checkpoints"

N_ENVS=32
N_ROLLOUTS=500
N_VIDEO_TRIALS=20
# ============================================================================

EXPERIMENT_NAME="$(basename "$(dirname "${CHECKPOINT_PATH}")")"
echo "EXPERIMENT_NAME: $EXPERIMENT_NAME"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Node: $HOSTNAME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"

# --- Environment setup ---
echo "Changing directory to $SLURM_SUBMIT_DIR"
cd "$SLURM_SUBMIT_DIR" || { echo "Failed to change directory to $SLURM_SUBMIT_DIR"; exit 1; }
source /etc/profile
source "${SLURM_SUBMIT_DIR}/env/bin/activate"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-0}}}"
echo "CUDA_VISIBLE_DEVICES after activate: ${CUDA_VISIBLE_DEVICES}"

mkdir -p logs

export PYTHONPATH=$PWD:$PYTHONPATH
export HOME=/data/locomotion/michzeng

echo "CHECKPOINT_PATH: $CHECKPOINT_PATH"
echo "HORIZON:         $HORIZON"
echo "N_ACTION_STEPS:  $N_ACTION_STEPS"
echo "N_ENVS:          $N_ENVS"
echo "N_ROLLOUTS:      $N_ROLLOUTS"
echo "N_VIDEO_TRIALS:  $N_VIDEO_TRIALS"
echo "=========================================="

bash robomimic/scripts/evaluate_checkpoints.sh \
    "${CHECKPOINT_PATH}" \
    "${HORIZON}" \
    "${N_ACTION_STEPS}" \
    "${N_ENVS}" \
    "${N_VIDEO_TRIALS}" \
    "${N_ROLLOUTS}" \
    --resume
