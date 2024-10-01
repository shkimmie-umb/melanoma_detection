
export CUDA_VISIBLE_DEVICES=5


mkdir -p ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152
echo "Created SLURMS/LOGS/ResNet152"


# echo "Started training ISIC2016_ResNet152
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 --CLASSIFIER ResNet152 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/ISIC2016_ResNet152.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/ISIC2016_ResNet152.err
# echo "Ended training ISIC2016_ResNet152
wait


# echo "Started training ISIC2017_ResNet152
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2017 --CLASSIFIER ResNet152 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/ISIC2017_ResNet152.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/ISIC2017_ResNet152.err
# echo "Ended training ISIC2017_ResNet152
wait


# echo "Started training ISIC2018_ResNet152
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2018 --CLASSIFIER ResNet152 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/ISIC2018_ResNet152.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/ISIC2018_ResNet152.err
# echo "Ended training ISIC2018_ResNet152
wait


# echo "Started training ISIC2019_ResNet152
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2019 --CLASSIFIER ResNet152 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/ISIC2019_ResNet152.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/ISIC2019_ResNet152.err
# echo "Ended training ISIC2019_ResNet152
wait


# echo "Started training ISIC2020_ResNet152
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 --CLASSIFIER ResNet152 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/ISIC2020_ResNet152.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/ISIC2020_ResNet152.err
# echo "Ended training ISIC2020_ResNet152
wait


# echo "Started training _7_point_criteria_ResNet152
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria --CLASSIFIER ResNet152 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/_7_point_criteria_ResNet152.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/_7_point_criteria_ResNet152.err
# echo "Ended training _7_point_criteria_ResNet152
wait


# echo "Started training PAD_UFES_20_ResNet152
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PAD_UFES_20 --CLASSIFIER ResNet152 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/PAD_UFES_20_ResNet152.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/PAD_UFES_20_ResNet152.err
# echo "Ended training PAD_UFES_20_ResNet152
wait


# echo "Started training MEDNODE_ResNet152
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB MEDNODE --CLASSIFIER ResNet152 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/MEDNODE_ResNet152.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/MEDNODE_ResNet152.err
# echo "Ended training MEDNODE_ResNet152
wait


# echo "Started training KaggleMB_ResNet152
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB KaggleMB --CLASSIFIER ResNet152 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/KaggleMB_ResNet152.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/KaggleMB_ResNet152.err
# echo "Ended training KaggleMB_ResNet152
wait


# echo "Started training ISIC2016+ISIC2017_ResNet152
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 --CLASSIFIER ResNet152 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/ISIC2016+ISIC2017_ResNet152.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/ISIC2016+ISIC2017_ResNet152.err
# echo "Ended training ISIC2016+ISIC2017_ResNet152
wait


# echo "Started training ISIC2018+ISIC2019_ResNet152
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2018 ISIC2019 --CLASSIFIER ResNet152 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/ISIC2018+ISIC2019_ResNet152.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/ISIC2018+ISIC2019_ResNet152.err
# echo "Ended training ISIC2018+ISIC2019_ResNet152
wait


# echo "Started training ISIC2020+PH2_ResNet152
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 PH2 --CLASSIFIER ResNet152 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/ISIC2020+PH2_ResNet152.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/ISIC2020+PH2_ResNet152.err
# echo "Ended training ISIC2020+PH2_ResNet152
wait


# echo "Started training _7_point_criteria+PAD_UFES_20_ResNet152
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 --CLASSIFIER ResNet152 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/_7_point_criteria+PAD_UFES_20_ResNet152.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/_7_point_criteria+PAD_UFES_20_ResNet152.err
# echo "Ended training _7_point_criteria+PAD_UFES_20_ResNet152
wait


# echo "Started training MEDNODE+KaggleMB_ResNet152
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB MEDNODE KaggleMB --CLASSIFIER ResNet152 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/MEDNODE+KaggleMB_ResNet152.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/MEDNODE+KaggleMB_ResNet152.err
# echo "Ended training MEDNODE+KaggleMB_ResNet152
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018_ResNet152
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 --CLASSIFIER ResNet152 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/ISIC2016+ISIC2017+ISIC2018_ResNet152.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/ISIC2016+ISIC2017+ISIC2018_ResNet152.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018_ResNet152
wait


# echo "Started training ISIC2019+ISIC2020+PH2_ResNet152
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2019 ISIC2020 PH2 --CLASSIFIER ResNet152 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/ISIC2019+ISIC2020+PH2_ResNet152.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/ISIC2019+ISIC2020+PH2_ResNet152.err
# echo "Ended training ISIC2019+ISIC2020+PH2_ResNet152
wait


# echo "Started training _7_point_criteria+PAD_UFES_20+MEDNODE_ResNet152
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 MEDNODE --CLASSIFIER ResNet152 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/_7_point_criteria+PAD_UFES_20+MEDNODE_ResNet152.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/_7_point_criteria+PAD_UFES_20+MEDNODE_ResNet152.err
# echo "Ended training _7_point_criteria+PAD_UFES_20+MEDNODE_ResNet152
wait


# echo "Started training PAD_UFES_20+MEDNODE+KaggleMB_ResNet152
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER ResNet152 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/PAD_UFES_20+MEDNODE+KaggleMB_ResNet152.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/PAD_UFES_20+MEDNODE+KaggleMB_ResNet152.err
# echo "Ended training PAD_UFES_20+MEDNODE+KaggleMB_ResNet152
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019_ResNet152
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 --CLASSIFIER ResNet152 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/ISIC2016+ISIC2017+ISIC2018+ISIC2019_ResNet152.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/ISIC2016+ISIC2017+ISIC2018+ISIC2019_ResNet152.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019_ResNet152
wait


# echo "Started training ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNet152
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER ResNet152 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNet152.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNet152.err
# echo "Ended training ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNet152
wait


# echo "Started training _7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet152
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER ResNet152 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet152.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet152.err
# echo "Ended training _7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet152
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ResNet152
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER ResNet152 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ResNet152.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ResNet152.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ResNet152
wait


# echo "Started training PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet152
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER ResNet152 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet152.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet152.err
# echo "Ended training PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet152
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_ResNet152
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 --CLASSIFIER ResNet152 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_ResNet152.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_ResNet152.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_ResNet152
wait


# echo "Started training ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet152
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER ResNet152 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet152.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet152.err
# echo "Ended training ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet152
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_ResNet152
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria --CLASSIFIER ResNet152 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_ResNet152.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_ResNet152.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_ResNet152
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_ResNet152
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER ResNet152 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_ResNet152.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_ResNet152.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_ResNet152
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNet152
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER ResNet152 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNet152.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNet152.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNet152
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_ResNet152
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE --CLASSIFIER ResNet152 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_ResNet152.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_ResNet152.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_ResNet152
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet152
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER ResNet152 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet152.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet152.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet152
wait

