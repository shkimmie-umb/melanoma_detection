
export CUDA_VISIBLE_DEVICES=3


mkdir -p ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2
echo "Created SLURMS/LOGS/MobileNetV2"


# echo "Started training ISIC2016_MobileNetV2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 --CLASSIFIER MobileNetV2 > ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/ISIC2016_MobileNetV2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/ISIC2016_MobileNetV2.err
# echo "Ended training ISIC2016_MobileNetV2
wait


# echo "Started training ISIC2017_MobileNetV2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2017 --CLASSIFIER MobileNetV2 > ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/ISIC2017_MobileNetV2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/ISIC2017_MobileNetV2.err
# echo "Ended training ISIC2017_MobileNetV2
wait


# echo "Started training ISIC2018_MobileNetV2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2018 --CLASSIFIER MobileNetV2 > ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/ISIC2018_MobileNetV2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/ISIC2018_MobileNetV2.err
# echo "Ended training ISIC2018_MobileNetV2
wait


# echo "Started training ISIC2019_MobileNetV2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2019 --CLASSIFIER MobileNetV2 > ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/ISIC2019_MobileNetV2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/ISIC2019_MobileNetV2.err
# echo "Ended training ISIC2019_MobileNetV2
wait


# echo "Started training ISIC2020_MobileNetV2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 --CLASSIFIER MobileNetV2 > ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/ISIC2020_MobileNetV2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/ISIC2020_MobileNetV2.err
# echo "Ended training ISIC2020_MobileNetV2
wait


# echo "Started training _7_point_criteria_MobileNetV2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria --CLASSIFIER MobileNetV2 > ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/_7_point_criteria_MobileNetV2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/_7_point_criteria_MobileNetV2.err
# echo "Ended training _7_point_criteria_MobileNetV2
wait


# echo "Started training PAD_UFES_20_MobileNetV2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PAD_UFES_20 --CLASSIFIER MobileNetV2 > ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/PAD_UFES_20_MobileNetV2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/PAD_UFES_20_MobileNetV2.err
# echo "Ended training PAD_UFES_20_MobileNetV2
wait


# echo "Started training MEDNODE_MobileNetV2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB MEDNODE --CLASSIFIER MobileNetV2 > ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/MEDNODE_MobileNetV2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/MEDNODE_MobileNetV2.err
# echo "Ended training MEDNODE_MobileNetV2
wait


# echo "Started training KaggleMB_MobileNetV2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB KaggleMB --CLASSIFIER MobileNetV2 > ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/KaggleMB_MobileNetV2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/KaggleMB_MobileNetV2.err
# echo "Ended training KaggleMB_MobileNetV2
wait


# echo "Started training ISIC2016+ISIC2017_MobileNetV2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 --CLASSIFIER MobileNetV2 > ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/ISIC2016+ISIC2017_MobileNetV2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/ISIC2016+ISIC2017_MobileNetV2.err
# echo "Ended training ISIC2016+ISIC2017_MobileNetV2
wait


# echo "Started training ISIC2018+ISIC2019_MobileNetV2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2018 ISIC2019 --CLASSIFIER MobileNetV2 > ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/ISIC2018+ISIC2019_MobileNetV2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/ISIC2018+ISIC2019_MobileNetV2.err
# echo "Ended training ISIC2018+ISIC2019_MobileNetV2
wait


# echo "Started training ISIC2020+PH2_MobileNetV2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 PH2 --CLASSIFIER MobileNetV2 > ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/ISIC2020+PH2_MobileNetV2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/ISIC2020+PH2_MobileNetV2.err
# echo "Ended training ISIC2020+PH2_MobileNetV2
wait


# echo "Started training _7_point_criteria+PAD_UFES_20_MobileNetV2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 --CLASSIFIER MobileNetV2 > ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/_7_point_criteria+PAD_UFES_20_MobileNetV2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/_7_point_criteria+PAD_UFES_20_MobileNetV2.err
# echo "Ended training _7_point_criteria+PAD_UFES_20_MobileNetV2
wait


# echo "Started training MEDNODE+KaggleMB_MobileNetV2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB MEDNODE KaggleMB --CLASSIFIER MobileNetV2 > ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/MEDNODE+KaggleMB_MobileNetV2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/MEDNODE+KaggleMB_MobileNetV2.err
# echo "Ended training MEDNODE+KaggleMB_MobileNetV2
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018_MobileNetV2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 --CLASSIFIER MobileNetV2 > ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/ISIC2016+ISIC2017+ISIC2018_MobileNetV2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/ISIC2016+ISIC2017+ISIC2018_MobileNetV2.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018_MobileNetV2
wait


# echo "Started training ISIC2019+ISIC2020+PH2_MobileNetV2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2019 ISIC2020 PH2 --CLASSIFIER MobileNetV2 > ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/ISIC2019+ISIC2020+PH2_MobileNetV2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/ISIC2019+ISIC2020+PH2_MobileNetV2.err
# echo "Ended training ISIC2019+ISIC2020+PH2_MobileNetV2
wait


# echo "Started training _7_point_criteria+PAD_UFES_20+MEDNODE_MobileNetV2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 MEDNODE --CLASSIFIER MobileNetV2 > ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/_7_point_criteria+PAD_UFES_20+MEDNODE_MobileNetV2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/_7_point_criteria+PAD_UFES_20+MEDNODE_MobileNetV2.err
# echo "Ended training _7_point_criteria+PAD_UFES_20+MEDNODE_MobileNetV2
wait


# echo "Started training PAD_UFES_20+MEDNODE+KaggleMB_MobileNetV2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER MobileNetV2 > ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/PAD_UFES_20+MEDNODE+KaggleMB_MobileNetV2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/PAD_UFES_20+MEDNODE+KaggleMB_MobileNetV2.err
# echo "Ended training PAD_UFES_20+MEDNODE+KaggleMB_MobileNetV2
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019_MobileNetV2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 --CLASSIFIER MobileNetV2 > ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/ISIC2016+ISIC2017+ISIC2018+ISIC2019_MobileNetV2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/ISIC2016+ISIC2017+ISIC2018+ISIC2019_MobileNetV2.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019_MobileNetV2
wait


# echo "Started training ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_MobileNetV2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER MobileNetV2 > ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_MobileNetV2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_MobileNetV2.err
# echo "Ended training ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_MobileNetV2
wait


# echo "Started training _7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MobileNetV2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER MobileNetV2 > ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MobileNetV2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MobileNetV2.err
# echo "Ended training _7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MobileNetV2
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_MobileNetV2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER MobileNetV2 > ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_MobileNetV2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_MobileNetV2.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_MobileNetV2
wait


# echo "Started training PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MobileNetV2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER MobileNetV2 > ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MobileNetV2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MobileNetV2.err
# echo "Ended training PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MobileNetV2
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_MobileNetV2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 --CLASSIFIER MobileNetV2 > ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_MobileNetV2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_MobileNetV2.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_MobileNetV2
wait


# echo "Started training ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MobileNetV2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER MobileNetV2 > ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MobileNetV2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MobileNetV2.err
# echo "Ended training ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MobileNetV2
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_MobileNetV2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria --CLASSIFIER MobileNetV2 > ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_MobileNetV2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_MobileNetV2.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_MobileNetV2
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_MobileNetV2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER MobileNetV2 > ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_MobileNetV2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_MobileNetV2.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_MobileNetV2
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_MobileNetV2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER MobileNetV2 > ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_MobileNetV2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_MobileNetV2.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_MobileNetV2
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_MobileNetV2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE --CLASSIFIER MobileNetV2 > ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_MobileNetV2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_MobileNetV2.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_MobileNetV2
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MobileNetV2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER MobileNetV2 > ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MobileNetV2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MobileNetV2.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MobileNetV2
wait

