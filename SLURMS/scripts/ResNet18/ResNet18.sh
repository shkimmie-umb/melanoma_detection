
export CUDA_VISIBLE_DEVICES=2


mkdir -p ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18
echo "Created SLURMS/LOGS/ResNet18"


# echo "Started training ISIC2016_ResNet18
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 --CLASSIFIER ResNet18 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/ISIC2016_ResNet18.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/ISIC2016_ResNet18.err
# echo "Ended training ISIC2016_ResNet18
wait


# echo "Started training ISIC2017_ResNet18
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2017 --CLASSIFIER ResNet18 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/ISIC2017_ResNet18.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/ISIC2017_ResNet18.err
# echo "Ended training ISIC2017_ResNet18
wait


# echo "Started training ISIC2018_ResNet18
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2018 --CLASSIFIER ResNet18 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/ISIC2018_ResNet18.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/ISIC2018_ResNet18.err
# echo "Ended training ISIC2018_ResNet18
wait


# echo "Started training ISIC2019_ResNet18
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2019 --CLASSIFIER ResNet18 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/ISIC2019_ResNet18.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/ISIC2019_ResNet18.err
# echo "Ended training ISIC2019_ResNet18
wait


# echo "Started training ISIC2020_ResNet18
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 --CLASSIFIER ResNet18 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/ISIC2020_ResNet18.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/ISIC2020_ResNet18.err
# echo "Ended training ISIC2020_ResNet18
wait


# echo "Started training _7_point_criteria_ResNet18
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria --CLASSIFIER ResNet18 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/_7_point_criteria_ResNet18.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/_7_point_criteria_ResNet18.err
# echo "Ended training _7_point_criteria_ResNet18
wait


# echo "Started training PAD_UFES_20_ResNet18
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PAD_UFES_20 --CLASSIFIER ResNet18 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/PAD_UFES_20_ResNet18.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/PAD_UFES_20_ResNet18.err
# echo "Ended training PAD_UFES_20_ResNet18
wait


# echo "Started training MEDNODE_ResNet18
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB MEDNODE --CLASSIFIER ResNet18 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/MEDNODE_ResNet18.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/MEDNODE_ResNet18.err
# echo "Ended training MEDNODE_ResNet18
wait


# echo "Started training KaggleMB_ResNet18
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB KaggleMB --CLASSIFIER ResNet18 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/KaggleMB_ResNet18.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/KaggleMB_ResNet18.err
# echo "Ended training KaggleMB_ResNet18
wait


# echo "Started training ISIC2016+ISIC2017_ResNet18
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 --CLASSIFIER ResNet18 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/ISIC2016+ISIC2017_ResNet18.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/ISIC2016+ISIC2017_ResNet18.err
# echo "Ended training ISIC2016+ISIC2017_ResNet18
wait


# echo "Started training ISIC2018+ISIC2019_ResNet18
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2018 ISIC2019 --CLASSIFIER ResNet18 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/ISIC2018+ISIC2019_ResNet18.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/ISIC2018+ISIC2019_ResNet18.err
# echo "Ended training ISIC2018+ISIC2019_ResNet18
wait


# echo "Started training ISIC2020+PH2_ResNet18
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 PH2 --CLASSIFIER ResNet18 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/ISIC2020+PH2_ResNet18.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/ISIC2020+PH2_ResNet18.err
# echo "Ended training ISIC2020+PH2_ResNet18
wait


# echo "Started training _7_point_criteria+PAD_UFES_20_ResNet18
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 --CLASSIFIER ResNet18 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/_7_point_criteria+PAD_UFES_20_ResNet18.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/_7_point_criteria+PAD_UFES_20_ResNet18.err
# echo "Ended training _7_point_criteria+PAD_UFES_20_ResNet18
wait


# echo "Started training MEDNODE+KaggleMB_ResNet18
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB MEDNODE KaggleMB --CLASSIFIER ResNet18 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/MEDNODE+KaggleMB_ResNet18.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/MEDNODE+KaggleMB_ResNet18.err
# echo "Ended training MEDNODE+KaggleMB_ResNet18
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018_ResNet18
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 --CLASSIFIER ResNet18 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/ISIC2016+ISIC2017+ISIC2018_ResNet18.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/ISIC2016+ISIC2017+ISIC2018_ResNet18.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018_ResNet18
wait


# echo "Started training ISIC2019+ISIC2020+PH2_ResNet18
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2019 ISIC2020 PH2 --CLASSIFIER ResNet18 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/ISIC2019+ISIC2020+PH2_ResNet18.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/ISIC2019+ISIC2020+PH2_ResNet18.err
# echo "Ended training ISIC2019+ISIC2020+PH2_ResNet18
wait


# echo "Started training _7_point_criteria+PAD_UFES_20+MEDNODE_ResNet18
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 MEDNODE --CLASSIFIER ResNet18 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/_7_point_criteria+PAD_UFES_20+MEDNODE_ResNet18.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/_7_point_criteria+PAD_UFES_20+MEDNODE_ResNet18.err
# echo "Ended training _7_point_criteria+PAD_UFES_20+MEDNODE_ResNet18
wait


# echo "Started training PAD_UFES_20+MEDNODE+KaggleMB_ResNet18
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER ResNet18 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/PAD_UFES_20+MEDNODE+KaggleMB_ResNet18.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/PAD_UFES_20+MEDNODE+KaggleMB_ResNet18.err
# echo "Ended training PAD_UFES_20+MEDNODE+KaggleMB_ResNet18
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019_ResNet18
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 --CLASSIFIER ResNet18 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/ISIC2016+ISIC2017+ISIC2018+ISIC2019_ResNet18.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/ISIC2016+ISIC2017+ISIC2018+ISIC2019_ResNet18.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019_ResNet18
wait


# echo "Started training ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNet18
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER ResNet18 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNet18.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNet18.err
# echo "Ended training ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNet18
wait


# echo "Started training _7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet18
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER ResNet18 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet18.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet18.err
# echo "Ended training _7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet18
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ResNet18
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER ResNet18 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ResNet18.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ResNet18.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ResNet18
wait


# echo "Started training PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet18
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER ResNet18 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet18.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet18.err
# echo "Ended training PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet18
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_ResNet18
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 --CLASSIFIER ResNet18 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_ResNet18.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_ResNet18.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_ResNet18
wait


# echo "Started training ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet18
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER ResNet18 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet18.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet18.err
# echo "Ended training ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet18
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_ResNet18
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria --CLASSIFIER ResNet18 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_ResNet18.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_ResNet18.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_ResNet18
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_ResNet18
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER ResNet18 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_ResNet18.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_ResNet18.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_ResNet18
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNet18
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER ResNet18 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNet18.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNet18.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNet18
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_ResNet18
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE --CLASSIFIER ResNet18 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_ResNet18.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_ResNet18.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_ResNet18
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet18
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER ResNet18 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet18.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet18/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet18.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet18
wait

