
export CUDA_VISIBLE_DEVICES=3


mkdir -p ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34
echo "Created SLURMS/LOGS/ResNet34"


# echo "Started training ISIC2016_ResNet34
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 --CLASSIFIER ResNet34 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016_ResNet34.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016_ResNet34.err
# echo "Ended training ISIC2016_ResNet34
wait


# echo "Started training ISIC2017_ResNet34
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2017 --CLASSIFIER ResNet34 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2017_ResNet34.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2017_ResNet34.err
# echo "Ended training ISIC2017_ResNet34
wait


# echo "Started training ISIC2018_ResNet34
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2018 --CLASSIFIER ResNet34 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2018_ResNet34.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2018_ResNet34.err
# echo "Ended training ISIC2018_ResNet34
wait


# echo "Started training ISIC2019_ResNet34
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2019 --CLASSIFIER ResNet34 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2019_ResNet34.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2019_ResNet34.err
# echo "Ended training ISIC2019_ResNet34
wait


# echo "Started training ISIC2020_ResNet34
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 --CLASSIFIER ResNet34 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2020_ResNet34.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2020_ResNet34.err
# echo "Ended training ISIC2020_ResNet34
wait


# echo "Started training _7_point_criteria_ResNet34
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria --CLASSIFIER ResNet34 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/_7_point_criteria_ResNet34.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/_7_point_criteria_ResNet34.err
# echo "Ended training _7_point_criteria_ResNet34
wait


# echo "Started training PAD_UFES_20_ResNet34
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PAD_UFES_20 --CLASSIFIER ResNet34 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/PAD_UFES_20_ResNet34.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/PAD_UFES_20_ResNet34.err
# echo "Ended training PAD_UFES_20_ResNet34
wait


# echo "Started training MEDNODE_ResNet34
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB MEDNODE --CLASSIFIER ResNet34 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/MEDNODE_ResNet34.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/MEDNODE_ResNet34.err
# echo "Ended training MEDNODE_ResNet34
wait


# echo "Started training KaggleMB_ResNet34
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB KaggleMB --CLASSIFIER ResNet34 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/KaggleMB_ResNet34.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/KaggleMB_ResNet34.err
# echo "Ended training KaggleMB_ResNet34
wait


# echo "Started training ISIC2016+ISIC2017_ResNet34
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 --CLASSIFIER ResNet34 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2017_ResNet34.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2017_ResNet34.err
# echo "Ended training ISIC2016+ISIC2017_ResNet34
wait


# echo "Started training ISIC2018+ISIC2019_ResNet34
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2018 ISIC2019 --CLASSIFIER ResNet34 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2018+ISIC2019_ResNet34.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2018+ISIC2019_ResNet34.err
# echo "Ended training ISIC2018+ISIC2019_ResNet34
wait


# echo "Started training ISIC2020+PH2_ResNet34
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 PH2 --CLASSIFIER ResNet34 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2020+PH2_ResNet34.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2020+PH2_ResNet34.err
# echo "Ended training ISIC2020+PH2_ResNet34
wait


# echo "Started training _7_point_criteria+PAD_UFES_20_ResNet34
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 --CLASSIFIER ResNet34 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/_7_point_criteria+PAD_UFES_20_ResNet34.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/_7_point_criteria+PAD_UFES_20_ResNet34.err
# echo "Ended training _7_point_criteria+PAD_UFES_20_ResNet34
wait


# echo "Started training MEDNODE+KaggleMB_ResNet34
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB MEDNODE KaggleMB --CLASSIFIER ResNet34 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/MEDNODE+KaggleMB_ResNet34.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/MEDNODE+KaggleMB_ResNet34.err
# echo "Ended training MEDNODE+KaggleMB_ResNet34
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018_ResNet34
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 --CLASSIFIER ResNet34 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2017+ISIC2018_ResNet34.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2017+ISIC2018_ResNet34.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018_ResNet34
wait


# echo "Started training ISIC2019+ISIC2020+PH2_ResNet34
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2019 ISIC2020 PH2 --CLASSIFIER ResNet34 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2019+ISIC2020+PH2_ResNet34.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2019+ISIC2020+PH2_ResNet34.err
# echo "Ended training ISIC2019+ISIC2020+PH2_ResNet34
wait


# echo "Started training _7_point_criteria+PAD_UFES_20+MEDNODE_ResNet34
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 MEDNODE --CLASSIFIER ResNet34 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/_7_point_criteria+PAD_UFES_20+MEDNODE_ResNet34.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/_7_point_criteria+PAD_UFES_20+MEDNODE_ResNet34.err
# echo "Ended training _7_point_criteria+PAD_UFES_20+MEDNODE_ResNet34
wait


# echo "Started training PAD_UFES_20+MEDNODE+KaggleMB_ResNet34
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER ResNet34 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/PAD_UFES_20+MEDNODE+KaggleMB_ResNet34.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/PAD_UFES_20+MEDNODE+KaggleMB_ResNet34.err
# echo "Ended training PAD_UFES_20+MEDNODE+KaggleMB_ResNet34
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019_ResNet34
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 --CLASSIFIER ResNet34 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2017+ISIC2018+ISIC2019_ResNet34.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2017+ISIC2018+ISIC2019_ResNet34.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019_ResNet34
wait


# echo "Started training ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNet34
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER ResNet34 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNet34.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNet34.err
# echo "Ended training ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNet34
wait


# echo "Started training _7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet34
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER ResNet34 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet34.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet34.err
# echo "Ended training _7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet34
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ResNet34
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER ResNet34 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ResNet34.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ResNet34.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ResNet34
wait


# echo "Started training PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet34
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER ResNet34 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet34.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet34.err
# echo "Ended training PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet34
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_ResNet34
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 --CLASSIFIER ResNet34 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_ResNet34.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_ResNet34.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_ResNet34
wait


# echo "Started training ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet34
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER ResNet34 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet34.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet34.err
# echo "Ended training ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet34
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_ResNet34
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria --CLASSIFIER ResNet34 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_ResNet34.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_ResNet34.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_ResNet34
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_ResNet34
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER ResNet34 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_ResNet34.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_ResNet34.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_ResNet34
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNet34
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER ResNet34 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNet34.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNet34.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNet34
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_ResNet34
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE --CLASSIFIER ResNet34 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_ResNet34.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_ResNet34.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_ResNet34
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet34
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER ResNet34 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet34.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet34.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet34
wait

