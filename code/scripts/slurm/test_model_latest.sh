#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -t 0-02:00
#SBATCH -p seas_dgx1
#SBATCH --exclude=seasdgx103
#SBATCH --gres=gpu:1
#SBATCH --mem=32768
#SBATCH -o /n/home00/apalrecha/lab/Deblurring/distill-superres/experiment/slurm_output/test_model_latest_%j.out
#SBATCH -e /n/home00/apalrecha/lab/Deblurring/distill-superres/experiment/slurm_output/test_model_latest_%j.err

echo "Settinp up env. Loading Conda, MATLAB, etc ..."
source ~/.bashrc
cd /n/home00/apalrecha/lab
echo "Activating EDSR conda environment ..."
source activate envs/edsr
echo "Loading MATLAB ..."
loadmatlab
echo "Setup Done!"
echo ""
echo "---- BEGIN TESTING ----"

cd Deblurring/EDSR-PyTorch/src
echo "Testing FAKD RCAN 10,6 Model"
python main.py --model rcan_student --n_resgroups 10 --n_resblocks 6 --n_feats 64 --scale 2 --ext sep --save_results --save ../../distill-superres/experiment/overall_distillation/rcan/SA_rcan_x2/results_latest --pre_train ../../distill-superres/experiment/overall_distillation/rcan/SA_rcan_x2/model/model_latest.pt --test_only --data_test Set5+Set14+B100+Urban100+Manga109+DIV2K --data_range 801-900
echo ""
echo "Testing RCAN 10,6 Fitnet-Share model (ours)"
python main.py --model rcan_student --n_resgroups 10 --n_resblocks 6 --n_feats 64 --scale 2 --ext sep --save_results --save ../../distill-superres/experiment/overall_distillation/rcan/rcan_10_6_fitnet_share/results_latest     --pre_train ../../distill-superres/experiment/overall_distillation/rcan/rcan_10_6_fitnet_share/model/model_latest.pt --test_only --data_test Set5+Set14+B100+Urban100+Manga109+DIV2K --data_range 801-900
echo ""

echo ""
echo "Start MATLAB Testing"
echo ""
cd ../scripts/matlab_evaluation
./benchmark_eval.sh 2 ../../distill-superres/experiment/overall_distillation/rcan/SA_rcan_x2/results_latest
./benchmark_eval.sh 2 ../../distill-superres/experiment/overall_distillation/rcan/rcan_10_6_fitnet_share/results_latest
echo ""
echo "---- TESTING COMPLETE ----"