#!/bin/bash -e
#SBATCH --job-name=PSF-sbatch
#SBATCH --output=/lustre/scratch/client/vinai/users/hainn14/PSF-new/spp_noti/train_flow_spp.out
#SBATCH --error=/lustre/scratch/client/vinai/users/hainn14/PSF-new/spp_noti/train_flow_spp.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=3
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
cd /lustre/scratch/client/vinai/users/hainn14/PSF-new

dataroot="datasets/ShapeNetCore.v2.PC15k/"
category="plane"

num_channels=3
batch_size=256
workers=16
nepoch=20000
dist="multi"

save_epoch=1000
viz_epoch=1000
diag_epoch=1000
print_freq=100

python3 train_flow.py --category "$category" \
                    --dataroot "$dataroot" \
                    --num_channels $num_channels \
                    --batch_size $batch_size \
                    --workers $workers \
                    --nEpochs $nepoch \
                    --distribution_type $dist \
                    --saveEpoch $save_epoch \
                    --diagEpoch $diag_epoch \
                    --vizEpoch $viz_epoch \
                    --printFreqIter $print_freq
