
export CUDA_VISIBLE_DEVICES=0


mkdir -p ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05
echo "Created SLURMS/LOGS/MNASNet05"


# echo "Started training ISIC2016_MNASNet05
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 --CLASSIFIER MNASNet05 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/ISIC2016_MNASNet05.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/ISIC2016_MNASNet05.err
# echo "Ended training ISIC2016_MNASNet05
wait


# echo "Started training ISIC2017_MNASNet05
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2017 --CLASSIFIER MNASNet05 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/ISIC2017_MNASNet05.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/ISIC2017_MNASNet05.err
# echo "Ended training ISIC2017_MNASNet05
wait


# echo "Started training ISIC2018_MNASNet05
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2018 --CLASSIFIER MNASNet05 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/ISIC2018_MNASNet05.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/ISIC2018_MNASNet05.err
# echo "Ended training ISIC2018_MNASNet05
wait


# echo "Started training ISIC2019_MNASNet05
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2019 --CLASSIFIER MNASNet05 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/ISIC2019_MNASNet05.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/ISIC2019_MNASNet05.err
# echo "Ended training ISIC2019_MNASNet05
wait


# echo "Started training ISIC2020_MNASNet05
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 --CLASSIFIER MNASNet05 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/ISIC2020_MNASNet05.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/ISIC2020_MNASNet05.err
# echo "Ended training ISIC2020_MNASNet05
wait


# echo "Started training _7_point_criteria_MNASNet05
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria --CLASSIFIER MNASNet05 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/_7_point_criteria_MNASNet05.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/_7_point_criteria_MNASNet05.err
# echo "Ended training _7_point_criteria_MNASNet05
wait


# echo "Started training PAD_UFES_20_MNASNet05
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PAD_UFES_20 --CLASSIFIER MNASNet05 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/PAD_UFES_20_MNASNet05.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/PAD_UFES_20_MNASNet05.err
# echo "Ended training PAD_UFES_20_MNASNet05
wait


# echo "Started training MEDNODE_MNASNet05
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB MEDNODE --CLASSIFIER MNASNet05 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/MEDNODE_MNASNet05.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/MEDNODE_MNASNet05.err
# echo "Ended training MEDNODE_MNASNet05
wait


# echo "Started training KaggleMB_MNASNet05
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB KaggleMB --CLASSIFIER MNASNet05 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/KaggleMB_MNASNet05.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/KaggleMB_MNASNet05.err
# echo "Ended training KaggleMB_MNASNet05
wait


# echo "Started training ISIC2016+ISIC2017_MNASNet05
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 --CLASSIFIER MNASNet05 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/ISIC2016+ISIC2017_MNASNet05.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/ISIC2016+ISIC2017_MNASNet05.err
# echo "Ended training ISIC2016+ISIC2017_MNASNet05
wait


# echo "Started training ISIC2018+ISIC2019_MNASNet05
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2018 ISIC2019 --CLASSIFIER MNASNet05 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/ISIC2018+ISIC2019_MNASNet05.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/ISIC2018+ISIC2019_MNASNet05.err
# echo "Ended training ISIC2018+ISIC2019_MNASNet05
wait


# echo "Started training ISIC2020+PH2_MNASNet05
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 PH2 --CLASSIFIER MNASNet05 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/ISIC2020+PH2_MNASNet05.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/ISIC2020+PH2_MNASNet05.err
# echo "Ended training ISIC2020+PH2_MNASNet05
wait


# echo "Started training _7_point_criteria+PAD_UFES_20_MNASNet05
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 --CLASSIFIER MNASNet05 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/_7_point_criteria+PAD_UFES_20_MNASNet05.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/_7_point_criteria+PAD_UFES_20_MNASNet05.err
# echo "Ended training _7_point_criteria+PAD_UFES_20_MNASNet05
wait


# echo "Started training MEDNODE+KaggleMB_MNASNet05
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB MEDNODE KaggleMB --CLASSIFIER MNASNet05 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/MEDNODE+KaggleMB_MNASNet05.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/MEDNODE+KaggleMB_MNASNet05.err
# echo "Ended training MEDNODE+KaggleMB_MNASNet05
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018_MNASNet05
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 --CLASSIFIER MNASNet05 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/ISIC2016+ISIC2017+ISIC2018_MNASNet05.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/ISIC2016+ISIC2017+ISIC2018_MNASNet05.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018_MNASNet05
wait


# echo "Started training ISIC2019+ISIC2020+PH2_MNASNet05
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2019 ISIC2020 PH2 --CLASSIFIER MNASNet05 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/ISIC2019+ISIC2020+PH2_MNASNet05.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/ISIC2019+ISIC2020+PH2_MNASNet05.err
# echo "Ended training ISIC2019+ISIC2020+PH2_MNASNet05
wait


# echo "Started training _7_point_criteria+PAD_UFES_20+MEDNODE_MNASNet05
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 MEDNODE --CLASSIFIER MNASNet05 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/_7_point_criteria+PAD_UFES_20+MEDNODE_MNASNet05.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/_7_point_criteria+PAD_UFES_20+MEDNODE_MNASNet05.err
# echo "Ended training _7_point_criteria+PAD_UFES_20+MEDNODE_MNASNet05
wait


# echo "Started training PAD_UFES_20+MEDNODE+KaggleMB_MNASNet05
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER MNASNet05 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/PAD_UFES_20+MEDNODE+KaggleMB_MNASNet05.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/PAD_UFES_20+MEDNODE+KaggleMB_MNASNet05.err
# echo "Ended training PAD_UFES_20+MEDNODE+KaggleMB_MNASNet05
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019_MNASNet05
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 --CLASSIFIER MNASNet05 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/ISIC2016+ISIC2017+ISIC2018+ISIC2019_MNASNet05.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/ISIC2016+ISIC2017+ISIC2018+ISIC2019_MNASNet05.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019_MNASNet05
wait


# echo "Started training ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_MNASNet05
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER MNASNet05 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_MNASNet05.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_MNASNet05.err
# echo "Ended training ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_MNASNet05
wait


# echo "Started training _7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MNASNet05
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER MNASNet05 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MNASNet05.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MNASNet05.err
# echo "Ended training _7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MNASNet05
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_MNASNet05
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER MNASNet05 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_MNASNet05.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_MNASNet05.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_MNASNet05
wait


# echo "Started training PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MNASNet05
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER MNASNet05 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MNASNet05.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MNASNet05.err
# echo "Ended training PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MNASNet05
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_MNASNet05
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 --CLASSIFIER MNASNet05 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_MNASNet05.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_MNASNet05.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_MNASNet05
wait


# echo "Started training ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MNASNet05
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER MNASNet05 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MNASNet05.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MNASNet05.err
# echo "Ended training ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MNASNet05
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_MNASNet05
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria --CLASSIFIER MNASNet05 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_MNASNet05.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_MNASNet05.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_MNASNet05
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_MNASNet05
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER MNASNet05 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_MNASNet05.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_MNASNet05.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_MNASNet05
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_MNASNet05
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER MNASNet05 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_MNASNet05.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_MNASNet05.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_MNASNet05
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_MNASNet05
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE --CLASSIFIER MNASNet05 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_MNASNet05.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_MNASNet05.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_MNASNet05
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MNASNet05
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER MNASNet05 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MNASNet05.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet05/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MNASNet05.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MNASNet05
wait

