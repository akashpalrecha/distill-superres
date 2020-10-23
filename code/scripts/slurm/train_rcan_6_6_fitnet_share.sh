#!/bin/bash
#SBATCH -n 8
#SBATCH -N 1
#SBATCH -t 0-12:00
#SBATCH -p seas_dgx1
#SBATCH --exclude=seasdgx103
#SBATCH --gres=gpu:1
#SBATCH --mem=32768
#SBATCH -o /n/home00/apalrecha/lab/Deblurring/distill-superres/experiment/slurm_output/rcan_6_6_fitnet_share_x2_%j.out
#SBATCH -e /n/home00/apalrecha/lab/Deblurring/distill-superres/experiment/slurm_output/rcan_6_6_fitnet_share_x2_%j.err

echo "Settinp up env. Loading Conda, MATLAB, etc ..."
source ~/.bashrc
cd /n/home00/apalrecha/lab
echo "Activating EDSR conda environment ..."
source activate envs/edsr
echo "Setup Done!"
cd Deblurring/distill-superres/code
echo ""
echo "---- BEGIN TRAINING ----"
echo ""
python train.py --ckp_dir overall_distillation/rcan/rcan_6_6_fitnet_share/ --scale 2 --teacher [RCAN] --model RCAN --n_resblocks 6 --n_resgroups 6 --alpha 0.5 --feature_loss_used 1 --epochs 100 --patch_size 96 --chop --data_train DIV2K --use_last_feature --use_teacher_weights --freeze_upsampler --features [-1] --feature_distilation_type 1*fitnet
echo ""
echo "---- TRAINING COMPLETE ----"
echo ""