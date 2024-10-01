
export CUDA_VISIBLE_DEVICES=6


mkdir -p ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10
echo "Created SLURMS/LOGS/ShuffleNetV2x10"


# echo "Started training ISIC2016_ShuffleNetV2x10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 --CLASSIFIER ShuffleNetV2x10 > ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/ISIC2016_ShuffleNetV2x10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/ISIC2016_ShuffleNetV2x10.err
# echo "Ended training ISIC2016_ShuffleNetV2x10
wait


# echo "Started training ISIC2017_ShuffleNetV2x10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2017 --CLASSIFIER ShuffleNetV2x10 > ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/ISIC2017_ShuffleNetV2x10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/ISIC2017_ShuffleNetV2x10.err
# echo "Ended training ISIC2017_ShuffleNetV2x10
wait


# echo "Started training ISIC2018_ShuffleNetV2x10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2018 --CLASSIFIER ShuffleNetV2x10 > ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/ISIC2018_ShuffleNetV2x10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/ISIC2018_ShuffleNetV2x10.err
# echo "Ended training ISIC2018_ShuffleNetV2x10
wait


# echo "Started training ISIC2019_ShuffleNetV2x10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2019 --CLASSIFIER ShuffleNetV2x10 > ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/ISIC2019_ShuffleNetV2x10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/ISIC2019_ShuffleNetV2x10.err
# echo "Ended training ISIC2019_ShuffleNetV2x10
wait


# echo "Started training ISIC2020_ShuffleNetV2x10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 --CLASSIFIER ShuffleNetV2x10 > ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/ISIC2020_ShuffleNetV2x10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/ISIC2020_ShuffleNetV2x10.err
# echo "Ended training ISIC2020_ShuffleNetV2x10
wait


# echo "Started training _7_point_criteria_ShuffleNetV2x10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria --CLASSIFIER ShuffleNetV2x10 > ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/_7_point_criteria_ShuffleNetV2x10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/_7_point_criteria_ShuffleNetV2x10.err
# echo "Ended training _7_point_criteria_ShuffleNetV2x10
wait


# echo "Started training PAD_UFES_20_ShuffleNetV2x10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PAD_UFES_20 --CLASSIFIER ShuffleNetV2x10 > ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/PAD_UFES_20_ShuffleNetV2x10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/PAD_UFES_20_ShuffleNetV2x10.err
# echo "Ended training PAD_UFES_20_ShuffleNetV2x10
wait


# echo "Started training MEDNODE_ShuffleNetV2x10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB MEDNODE --CLASSIFIER ShuffleNetV2x10 > ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/MEDNODE_ShuffleNetV2x10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/MEDNODE_ShuffleNetV2x10.err
# echo "Ended training MEDNODE_ShuffleNetV2x10
wait


# echo "Started training KaggleMB_ShuffleNetV2x10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB KaggleMB --CLASSIFIER ShuffleNetV2x10 > ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/KaggleMB_ShuffleNetV2x10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/KaggleMB_ShuffleNetV2x10.err
# echo "Ended training KaggleMB_ShuffleNetV2x10
wait


# echo "Started training ISIC2016+ISIC2017_ShuffleNetV2x10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 --CLASSIFIER ShuffleNetV2x10 > ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/ISIC2016+ISIC2017_ShuffleNetV2x10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/ISIC2016+ISIC2017_ShuffleNetV2x10.err
# echo "Ended training ISIC2016+ISIC2017_ShuffleNetV2x10
wait


# echo "Started training ISIC2018+ISIC2019_ShuffleNetV2x10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2018 ISIC2019 --CLASSIFIER ShuffleNetV2x10 > ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/ISIC2018+ISIC2019_ShuffleNetV2x10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/ISIC2018+ISIC2019_ShuffleNetV2x10.err
# echo "Ended training ISIC2018+ISIC2019_ShuffleNetV2x10
wait


# echo "Started training ISIC2020+PH2_ShuffleNetV2x10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 PH2 --CLASSIFIER ShuffleNetV2x10 > ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/ISIC2020+PH2_ShuffleNetV2x10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/ISIC2020+PH2_ShuffleNetV2x10.err
# echo "Ended training ISIC2020+PH2_ShuffleNetV2x10
wait


# echo "Started training _7_point_criteria+PAD_UFES_20_ShuffleNetV2x10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 --CLASSIFIER ShuffleNetV2x10 > ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/_7_point_criteria+PAD_UFES_20_ShuffleNetV2x10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/_7_point_criteria+PAD_UFES_20_ShuffleNetV2x10.err
# echo "Ended training _7_point_criteria+PAD_UFES_20_ShuffleNetV2x10
wait


# echo "Started training MEDNODE+KaggleMB_ShuffleNetV2x10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB MEDNODE KaggleMB --CLASSIFIER ShuffleNetV2x10 > ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/MEDNODE+KaggleMB_ShuffleNetV2x10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/MEDNODE+KaggleMB_ShuffleNetV2x10.err
# echo "Ended training MEDNODE+KaggleMB_ShuffleNetV2x10
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018_ShuffleNetV2x10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 --CLASSIFIER ShuffleNetV2x10 > ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/ISIC2016+ISIC2017+ISIC2018_ShuffleNetV2x10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/ISIC2016+ISIC2017+ISIC2018_ShuffleNetV2x10.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018_ShuffleNetV2x10
wait


# echo "Started training ISIC2019+ISIC2020+PH2_ShuffleNetV2x10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2019 ISIC2020 PH2 --CLASSIFIER ShuffleNetV2x10 > ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/ISIC2019+ISIC2020+PH2_ShuffleNetV2x10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/ISIC2019+ISIC2020+PH2_ShuffleNetV2x10.err
# echo "Ended training ISIC2019+ISIC2020+PH2_ShuffleNetV2x10
wait


# echo "Started training _7_point_criteria+PAD_UFES_20+MEDNODE_ShuffleNetV2x10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 MEDNODE --CLASSIFIER ShuffleNetV2x10 > ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/_7_point_criteria+PAD_UFES_20+MEDNODE_ShuffleNetV2x10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/_7_point_criteria+PAD_UFES_20+MEDNODE_ShuffleNetV2x10.err
# echo "Ended training _7_point_criteria+PAD_UFES_20+MEDNODE_ShuffleNetV2x10
wait


# echo "Started training PAD_UFES_20+MEDNODE+KaggleMB_ShuffleNetV2x10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER ShuffleNetV2x10 > ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/PAD_UFES_20+MEDNODE+KaggleMB_ShuffleNetV2x10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/PAD_UFES_20+MEDNODE+KaggleMB_ShuffleNetV2x10.err
# echo "Ended training PAD_UFES_20+MEDNODE+KaggleMB_ShuffleNetV2x10
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019_ShuffleNetV2x10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 --CLASSIFIER ShuffleNetV2x10 > ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/ISIC2016+ISIC2017+ISIC2018+ISIC2019_ShuffleNetV2x10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/ISIC2016+ISIC2017+ISIC2018+ISIC2019_ShuffleNetV2x10.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019_ShuffleNetV2x10
wait


# echo "Started training ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ShuffleNetV2x10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER ShuffleNetV2x10 > ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ShuffleNetV2x10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ShuffleNetV2x10.err
# echo "Ended training ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ShuffleNetV2x10
wait


# echo "Started training _7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ShuffleNetV2x10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER ShuffleNetV2x10 > ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ShuffleNetV2x10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ShuffleNetV2x10.err
# echo "Ended training _7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ShuffleNetV2x10
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ShuffleNetV2x10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER ShuffleNetV2x10 > ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ShuffleNetV2x10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ShuffleNetV2x10.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ShuffleNetV2x10
wait


# echo "Started training PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ShuffleNetV2x10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER ShuffleNetV2x10 > ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ShuffleNetV2x10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ShuffleNetV2x10.err
# echo "Ended training PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ShuffleNetV2x10
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_ShuffleNetV2x10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 --CLASSIFIER ShuffleNetV2x10 > ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_ShuffleNetV2x10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_ShuffleNetV2x10.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_ShuffleNetV2x10
wait


# echo "Started training ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ShuffleNetV2x10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER ShuffleNetV2x10 > ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ShuffleNetV2x10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ShuffleNetV2x10.err
# echo "Ended training ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ShuffleNetV2x10
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_ShuffleNetV2x10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria --CLASSIFIER ShuffleNetV2x10 > ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_ShuffleNetV2x10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_ShuffleNetV2x10.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_ShuffleNetV2x10
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_ShuffleNetV2x10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER ShuffleNetV2x10 > ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_ShuffleNetV2x10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_ShuffleNetV2x10.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_ShuffleNetV2x10
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ShuffleNetV2x10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER ShuffleNetV2x10 > ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ShuffleNetV2x10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ShuffleNetV2x10.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ShuffleNetV2x10
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_ShuffleNetV2x10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE --CLASSIFIER ShuffleNetV2x10 > ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_ShuffleNetV2x10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_ShuffleNetV2x10.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_ShuffleNetV2x10
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ShuffleNetV2x10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER ShuffleNetV2x10 > ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ShuffleNetV2x10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ShuffleNetV2x10.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ShuffleNetV2x10
wait

