#!/bin/bash

#SBATCH --job-name=ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_MNASNet10
#SBATCH -p haehn -q haehn_unlim
#SBATCH -w chimera13

#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -n 4 # Number of cores
#SBATCH --mem=50gb

#SBATCH --gres=gpu:A100:1

#SBATCH -t 01-00:00

#SBATCH --output /home/sanghyuk.kim001/MELANOMA/melanoma-detection-CNN/SLURMS/LOGS/MNASNet10/%x_%A_%a.out
#SBATCH --error /home/sanghyuk.kim001/MELANOMA/melanoma-detection-CNN/SLURMS/LOGS/MNASNet10/%x_%A_%a.err

##. /etc/profile,

eval "$(conda shell.bash hook)"
conda activate ood

echo `date`


# For debugging purposes.
python --version
nvcc -V

# Print this sub-job's task ID
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

cd /home/sanghyuk.kim001/MELANOMA/melanoma-detection-CNN/

python train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE --CLASSIFIER MNASNet10 --JOB_INDEX $SLURM_ARRAY_TASK_ID

echo "Job ended!"

# end
exit 0;
