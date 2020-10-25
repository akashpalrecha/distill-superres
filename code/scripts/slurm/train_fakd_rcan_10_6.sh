#!/bin/bash
#SBATCH -n 8
#SBATCH -N 1
#SBATCH -t 0-23:59
#SBATCH -p seas_dgx1
#SBATCH --exclude=seasdgx103
#SBATCH --gres=gpu:1
#SBATCH --mem=32768
#SBATCH -o /n/home00/apalrecha/lab/Deblurring/distill-superres/experiment/slurm_output/SA_rcan_10_6_x2_%j.out
#SBATCH -e /n/home00/apalrecha/lab/Deblurring/distill-superres/experiment/slurm_output/SA_rcan_10_6_x2_%j.err

echo "Settinp up env. Loading Conda, MATLAB, etc ..."
source ~/.bashrc
cd /n/home00/apalrecha/lab
echo "Activating EDSR conda environment ..."
source activate envs/edsr
echo "Loading MATLAB ..."
loadmatlab
echo "Setup Done!"
cd Deblurring/distill-superres/code
echo ""
echo "---- BEGIN TRAINING ----"
echo ""
python train.py --ckp_dir overall_distillation/rcan/SA_rcan_x2/ --scale 2 --teacher [RCAN] --model RCAN --n_resblocks 6 --n_resgroups 10 --alpha 0.5 --feature_loss_used 1 --epochs 100 --patch_size 96 --chop --data_train DIV2K --features [1,2,3] --feature_distilation_type 10*SA
echo ""
echo "---- TRAINING COMPLETE ----"
echo ""
echo "---- BEGIN TESTING ----"
cd ../../EDSR-PyTorch/src/
python main.py --model rcan_student --n_resgroups 10 --n_resblocks 6 --n_feats 64 --scale 2 --ext sep --save_results --save ../../distill-superres/experiment/overall_distillation/rcan/SA_rcan_x2/results --pre_train ../../distill-superres/experiment/overall_distillation/rcan/SA_rcan_x2/model/model_best.pt --test_only --data_test Set5+Set14+B100+Urban100+Manga109+DIV2K --data_range 801-900
echo ""
echo "Start MATLAB Testing"
echo ""
cd ../scripts/matlab_evaluation
./benchmark_eval.sh 2 ../../distill-superres/experiment/overall_distillation/rcan/SA_rcan_x2/results
echo ""
echo "---- TESTING COMPLETE ----"