
export CUDA_VISIBLE_DEVICES=4


mkdir -p ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101
echo "Created SLURMS/LOGS/ResNet101"


# echo "Started training ISIC2016_ResNet101
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 --CLASSIFIER ResNet101 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016_ResNet101.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016_ResNet101.err
# echo "Ended training ISIC2016_ResNet101
wait


# echo "Started training ISIC2017_ResNet101
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2017 --CLASSIFIER ResNet101 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2017_ResNet101.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2017_ResNet101.err
# echo "Ended training ISIC2017_ResNet101
wait


# echo "Started training ISIC2018_ResNet101
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2018 --CLASSIFIER ResNet101 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2018_ResNet101.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2018_ResNet101.err
# echo "Ended training ISIC2018_ResNet101
wait


# echo "Started training ISIC2019_ResNet101
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2019 --CLASSIFIER ResNet101 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2019_ResNet101.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2019_ResNet101.err
# echo "Ended training ISIC2019_ResNet101
wait


# echo "Started training ISIC2020_ResNet101
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 --CLASSIFIER ResNet101 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2020_ResNet101.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2020_ResNet101.err
# echo "Ended training ISIC2020_ResNet101
wait


# echo "Started training _7_point_criteria_ResNet101
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria --CLASSIFIER ResNet101 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/_7_point_criteria_ResNet101.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/_7_point_criteria_ResNet101.err
# echo "Ended training _7_point_criteria_ResNet101
wait


# echo "Started training PAD_UFES_20_ResNet101
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PAD_UFES_20 --CLASSIFIER ResNet101 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/PAD_UFES_20_ResNet101.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/PAD_UFES_20_ResNet101.err
# echo "Ended training PAD_UFES_20_ResNet101
wait


# echo "Started training MEDNODE_ResNet101
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB MEDNODE --CLASSIFIER ResNet101 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/MEDNODE_ResNet101.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/MEDNODE_ResNet101.err
# echo "Ended training MEDNODE_ResNet101
wait


# echo "Started training KaggleMB_ResNet101
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB KaggleMB --CLASSIFIER ResNet101 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/KaggleMB_ResNet101.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/KaggleMB_ResNet101.err
# echo "Ended training KaggleMB_ResNet101
wait


# echo "Started training ISIC2016+ISIC2017_ResNet101
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 --CLASSIFIER ResNet101 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2017_ResNet101.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2017_ResNet101.err
# echo "Ended training ISIC2016+ISIC2017_ResNet101
wait


# echo "Started training ISIC2018+ISIC2019_ResNet101
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2018 ISIC2019 --CLASSIFIER ResNet101 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2018+ISIC2019_ResNet101.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2018+ISIC2019_ResNet101.err
# echo "Ended training ISIC2018+ISIC2019_ResNet101
wait


# echo "Started training ISIC2020+PH2_ResNet101
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 PH2 --CLASSIFIER ResNet101 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2020+PH2_ResNet101.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2020+PH2_ResNet101.err
# echo "Ended training ISIC2020+PH2_ResNet101
wait


# echo "Started training _7_point_criteria+PAD_UFES_20_ResNet101
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 --CLASSIFIER ResNet101 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/_7_point_criteria+PAD_UFES_20_ResNet101.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/_7_point_criteria+PAD_UFES_20_ResNet101.err
# echo "Ended training _7_point_criteria+PAD_UFES_20_ResNet101
wait


# echo "Started training MEDNODE+KaggleMB_ResNet101
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB MEDNODE KaggleMB --CLASSIFIER ResNet101 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/MEDNODE+KaggleMB_ResNet101.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/MEDNODE+KaggleMB_ResNet101.err
# echo "Ended training MEDNODE+KaggleMB_ResNet101
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018_ResNet101
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 --CLASSIFIER ResNet101 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2017+ISIC2018_ResNet101.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2017+ISIC2018_ResNet101.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018_ResNet101
wait


# echo "Started training ISIC2019+ISIC2020+PH2_ResNet101
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2019 ISIC2020 PH2 --CLASSIFIER ResNet101 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2019+ISIC2020+PH2_ResNet101.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2019+ISIC2020+PH2_ResNet101.err
# echo "Ended training ISIC2019+ISIC2020+PH2_ResNet101
wait


# echo "Started training _7_point_criteria+PAD_UFES_20+MEDNODE_ResNet101
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 MEDNODE --CLASSIFIER ResNet101 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/_7_point_criteria+PAD_UFES_20+MEDNODE_ResNet101.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/_7_point_criteria+PAD_UFES_20+MEDNODE_ResNet101.err
# echo "Ended training _7_point_criteria+PAD_UFES_20+MEDNODE_ResNet101
wait


# echo "Started training PAD_UFES_20+MEDNODE+KaggleMB_ResNet101
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER ResNet101 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/PAD_UFES_20+MEDNODE+KaggleMB_ResNet101.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/PAD_UFES_20+MEDNODE+KaggleMB_ResNet101.err
# echo "Ended training PAD_UFES_20+MEDNODE+KaggleMB_ResNet101
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019_ResNet101
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 --CLASSIFIER ResNet101 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2017+ISIC2018+ISIC2019_ResNet101.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2017+ISIC2018+ISIC2019_ResNet101.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019_ResNet101
wait


# echo "Started training ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNet101
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER ResNet101 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNet101.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNet101.err
# echo "Ended training ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNet101
wait


# echo "Started training _7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet101
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER ResNet101 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet101.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet101.err
# echo "Ended training _7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet101
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ResNet101
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER ResNet101 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ResNet101.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ResNet101.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ResNet101
wait


# echo "Started training PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet101
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER ResNet101 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet101.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet101.err
# echo "Ended training PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet101
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_ResNet101
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 --CLASSIFIER ResNet101 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_ResNet101.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_ResNet101.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_ResNet101
wait


# echo "Started training ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet101
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER ResNet101 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet101.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet101.err
# echo "Ended training ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet101
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_ResNet101
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria --CLASSIFIER ResNet101 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_ResNet101.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_ResNet101.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_ResNet101
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_ResNet101
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER ResNet101 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_ResNet101.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_ResNet101.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_ResNet101
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNet101
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER ResNet101 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNet101.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNet101.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNet101
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_ResNet101
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE --CLASSIFIER ResNet101 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_ResNet101.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_ResNet101.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_ResNet101
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet101
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER ResNet101 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet101.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet101.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet101
wait
