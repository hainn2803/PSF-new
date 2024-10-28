#!/bin/bash -e
#SBATCH --job-name=PSF
#SBATCH --output=/lustre/scratch/client/vinai/users/hainn14/PSF/spp_noti/train_flow.out
#SBATCH --error=/lustre/scratch/client/vinai/users/hainn14/PSF/spp_noti/train_flow.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --mem-per-gpu=125G
#SBATCH --cpus-per-gpu=32
#SBATCH --partition=research
#SBATCH --mail-type=all
#SBATCH --mail-user=v.HaiNN14@vinai.io

module purge
module load python/miniconda3/miniconda3

# Corrected line
eval "$(conda shell.bash hook)"

conda activate /lustre/scratch/client/vinai/users/hainn14/envs/PSF
cd /lustre/scratch/client/vinai/users/hainn14/PSF

dataroot="datasets/ShapeNetCore.v2.PC15k/"
category="car"
num_classes=1
batch_size=32
workers=8
nepoch=100
dist="multi"
save_epoch=10
viz_epoch=10
diag_epoch=10
print_freq=100
python3 train_flow.py --category "$category" \
                    --dataroot "$dataroot" \
                    --num_classes $num_classes \
                    --batch_size $batch_size \
                    --workers $workers \
                    --nEpochs $nepoch \
                    --distribution_type $dist \
                    --saveEpoch $save_epoch \
                    --diagEpoch $diag_epoch \
                    --vizEpoch $viz_epoch \
                    --printFreqIter $print_freq
