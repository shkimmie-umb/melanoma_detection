#!/bin/bash

#SBATCH --job-name=HAM10000_1_ResNet50V2_150h_150w
#SBATCH -p haehn -q haehn_unlim
#SBATCH -w chimera12

#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -n 6 # Number of cores
#SBATCH --mem=200gb

#SBATCH --gres=gpu:A100:1

#SBATCH -t 01-00:00

#SBATCH --output /home/sanghyuk.kim001/MELANOMA/melanoma-detection-CNN/SLURMS/LOGS/ResNet50V2/%x_%A_%a.out
#SBATCH --error /home/sanghyuk.kim001/MELANOMA/melanoma-detection-CNN/SLURMS/LOGS/ResNet50V2/%x_%A_%a.err
##. /etc/profile,

eval "$(conda shell.bash hook)"
conda activate clean_chimera_env

echo `date`


# For debugging purposes.
python --version
nvcc -V

# Print this sub-job's task ID
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

cd /home/sanghyuk.kim001/MELANOMA/melanoma-detection-CNN/

python train.py --DB HAM10000 --IMG_SIZE 150 150 --CLASSIFIER ResNet50V2 --SELF_AUG 1 --JOB_INDEX $SLURM_ARRAY_TASK_ID

echo "Job ended!"

# end
exit 0;
