
export CUDA_VISIBLE_DEVICES=4


mkdir -p ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101
echo "Created SLURMS/LOGS/ResNet101"


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


# echo "Started training ISIC2016+ISIC2020_ResNet101
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2020 --CLASSIFIER ResNet101 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2020_ResNet101.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2020_ResNet101.err
# echo "Ended training ISIC2016+ISIC2020_ResNet101
wait


# echo "Started training ISIC2016+ISIC2020+PH2_ResNet101
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2020 PH2 --CLASSIFIER ResNet101 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2020+PH2_ResNet101.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2020+PH2_ResNet101.err
# echo "Ended training ISIC2016+ISIC2020+PH2_ResNet101
wait


# echo "Started training ISIC2016+ISIC2018+ISIC2019+ISIC2020_ResNet101
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER ResNet101 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2018+ISIC2019+ISIC2020_ResNet101.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2018+ISIC2019+ISIC2020_ResNet101.err
# echo "Ended training ISIC2016+ISIC2018+ISIC2019+ISIC2020_ResNet101
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ResNet101
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER ResNet101 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ResNet101.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ResNet101.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ResNet101
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_ResNet101
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 MEDNODE KaggleMB --CLASSIFIER ResNet101 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_ResNet101.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_ResNet101.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_ResNet101
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_ResNet101
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 PH2 _7_point_criteria --CLASSIFIER ResNet101 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_ResNet101.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_ResNet101.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_ResNet101
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_ResNet101
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2020 PH2 --CLASSIFIER ResNet101 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_ResNet101.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_ResNet101.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_ResNet101
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_ResNet101
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE --CLASSIFIER ResNet101 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_ResNet101.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_ResNet101.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_ResNet101
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_ResNet101
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 MEDNODE KaggleMB --CLASSIFIER ResNet101 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_ResNet101.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_ResNet101.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_ResNet101
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_ResNet101
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 --CLASSIFIER ResNet101 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_ResNet101.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_ResNet101.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_ResNet101
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_ResNet101
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria --CLASSIFIER ResNet101 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_ResNet101.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_ResNet101.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_ResNet101
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_ResNet101
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER ResNet101 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_ResNet101.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_ResNet101.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_ResNet101
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_ResNet101
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 _7_point_criteria PAD_UFES_20 --CLASSIFIER ResNet101 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_ResNet101.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_ResNet101.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_ResNet101
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_ResNet101
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PAD_UFES_20 MEDNODE --CLASSIFIER ResNet101 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_ResNet101.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_ResNet101.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_ResNet101
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_ResNet101
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 PAD_UFES_20 MEDNODE --CLASSIFIER ResNet101 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_ResNet101.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_ResNet101.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_ResNet101
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_ResNet101
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 MEDNODE KaggleMB --CLASSIFIER ResNet101 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_ResNet101.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_ResNet101.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_ResNet101
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNet101
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER ResNet101 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNet101.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNet101.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNet101
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_ResNet101
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria MEDNODE KaggleMB --CLASSIFIER ResNet101 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_ResNet101.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_ResNet101.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_ResNet101
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet101
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER ResNet101 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet101.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet101/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet101.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet101
wait

