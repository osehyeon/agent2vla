#!/bin/bash
# run from train/: mkdir -p logs && STEPS=20000 SAVE=5000 sbatch --account=<account> --time=02:30:00 smolvla_train.sbatch
#SBATCH --job-name=smolvla-cube
#SBATCH --gres=gpu:b200:1
#SBATCH --time=00:15:00
#SBATCH --output=logs/slurm-%j.out
set -e
cd "$SLURM_SUBMIT_DIR"
DATA=${DATA:-$PWD/data/so101_cube_regions_v1}
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
STEPS=${STEPS:-300}; SAVE=${SAVE:-1000000}
OUT=$HOME/runs/$SLURM_JOB_ID
srun uv run lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=local/so101_cube_regions_v1 \
  --dataset.root=$DATA \
  --dataset.video_backend=pyav \
  --rename_map="{\"observation.images.top\": \"observation.images.camera1\", \"observation.images.wrist\": \"observation.images.camera2\"}" \
  --batch_size=64 --num_workers=6 --steps=$STEPS --save_freq=$SAVE --log_freq=25 \
  --output_dir=$OUT --job_name=smolvla_cube_$SLURM_JOB_ID \
  --policy.device=cuda --policy.push_to_hub=false --wandb.enable=false
