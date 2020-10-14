#!/bin/bash
#SBATCH -n 8
#SBATCH -N 1
#SBATCH -t 0-23:59
#SBATCH -p seas_dgx1
#SBATCH --exclude=seasdgx103
#SBATCH --gres=gpu:1
#SBATCH --mem=32768
#SBATCH -o /n/home00/apalrecha/lab/Deblurring/distill-superres/experiment/slurm_output/rcan_4_4_flickr_plain_x2_%j.out
#SBATCH -e /n/home00/apalrecha/lab/Deblurring/distill-superres/experiment/slurm_output/rcan_4_4_flickr_plain_x2_%j.err

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
python train.py --ckp_dir overall_distillation/rcan/flickr_plain/ --scale 2 --teacher [RCAN] --model RCAN --n_resblocks 4 --n_resgroups 4 --alpha 0.5 --feature_loss_used 0 --epochs 100 --patch_size 96 --chop --data_train DIV2K+Flickr2k
echo ""
echo "---- TRAINING COMPLETE ----"
echo ""