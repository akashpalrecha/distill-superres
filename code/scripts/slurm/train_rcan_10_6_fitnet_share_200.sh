#!/bin/bash
#SBATCH -n 8
#SBATCH -N 1
#SBATCH -t 0-23:59
#SBATCH -p seas_dgx1
#SBATCH --exclude=seasdgx103
#SBATCH --gres=gpu:1
#SBATCH --mem=32768
#SBATCH -o /n/home00/apalrecha/lab/Deblurring/distill-superres/experiment/slurm_output/rcan_10_6_fitnet_share_200_x2_%j.out
#SBATCH -e /n/home00/apalrecha/lab/Deblurring/distill-superres/experiment/slurm_output/rcan_10_6_fitnet_share_200_x2_%j.err

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
python train.py --ckp_dir overall_distillation/rcan/rcan_10_6_fitnet_share_200/ --scale 2 --teacher [RCAN] --model RCAN --n_resblocks 6 --n_resgroups 10 --alpha 0.5 --feature_loss_used 1 --epochs 200 --patch_size 96 --chop --data_train DIV2K --use_last_feature --use_teacher_weights --freeze_upsampler --features [-1] --feature_distilation_type 1*fitnet --resume -1
echo ""
echo "---- TRAINING COMPLETE ----"
echo ""
echo "---- BEGIN TESTING ----"
cd ../../EDSR-PyTorch/src/
python main.py --model rcan_student --n_resgroups 10 --n_resblocks 6 --n_feats 64 --scale 2 --ext sep --save_results --save ../../distill-superres/experiment/overall_distillation/rcan/rcan_10_6_fitnet_share_200/results     --pre_train ../../distill-superres/experiment/overall_distillation/rcan/rcan_10_6_fitnet_share_200/model/model_best.pt --test_only --data_test Set5+Set14+B100+Urban100+Manga109+DIV2K --data_range 801-900
python main.py --model rcan_student --n_resgroups 10 --n_resblocks 6 --n_feats 64 --scale 2 --ext sep --save_results --save ../../distill-superres/experiment/overall_distillation/rcan/rcan_10_6_fitnet_share_200/results_latest     --pre_train ../../distill-superres/experiment/overall_distillation/rcan/rcan_10_6_fitnet_share_200/model/model_latest.pt --test_only --data_test Set5+Set14+B100+Urban100+Manga109+DIV2K --data_range 801-900
echo ""
echo "Start MATLAB Testing"
echo ""
cd ../scripts/matlab_evaluation
./benchmark_eval.sh 2 ../../distill-superres/experiment/overall_distillation/rcan/rcan_10_6_fitnet_share_200/results
./benchmark_eval.sh 2 ../../distill-superres/experiment/overall_distillation/rcan/rcan_10_6_fitnet_share_200/results_latest
echo ""
echo "---- TESTING COMPLETE ----"