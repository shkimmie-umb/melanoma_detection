
export CUDA_VISIBLE_DEVICES=2


mkdir -p ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11
echo "Created SLURMS/LOGS/VGG11"


# echo "Started training ISIC2016_VGG11
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 --CLASSIFIER VGG11 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/ISIC2016_VGG11.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/ISIC2016_VGG11.err
# echo "Ended training ISIC2016_VGG11
wait


# echo "Started training ISIC2017_VGG11
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2017 --CLASSIFIER VGG11 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/ISIC2017_VGG11.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/ISIC2017_VGG11.err
# echo "Ended training ISIC2017_VGG11
wait


# echo "Started training ISIC2018_VGG11
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2018 --CLASSIFIER VGG11 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/ISIC2018_VGG11.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/ISIC2018_VGG11.err
# echo "Ended training ISIC2018_VGG11
wait


# echo "Started training ISIC2019_VGG11
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2019 --CLASSIFIER VGG11 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/ISIC2019_VGG11.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/ISIC2019_VGG11.err
# echo "Ended training ISIC2019_VGG11
wait


# echo "Started training ISIC2020_VGG11
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 --CLASSIFIER VGG11 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/ISIC2020_VGG11.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/ISIC2020_VGG11.err
# echo "Ended training ISIC2020_VGG11
wait


# echo "Started training _7_point_criteria_VGG11
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria --CLASSIFIER VGG11 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/_7_point_criteria_VGG11.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/_7_point_criteria_VGG11.err
# echo "Ended training _7_point_criteria_VGG11
wait


# echo "Started training PAD_UFES_20_VGG11
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PAD_UFES_20 --CLASSIFIER VGG11 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/PAD_UFES_20_VGG11.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/PAD_UFES_20_VGG11.err
# echo "Ended training PAD_UFES_20_VGG11
wait


# echo "Started training MEDNODE_VGG11
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB MEDNODE --CLASSIFIER VGG11 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/MEDNODE_VGG11.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/MEDNODE_VGG11.err
# echo "Ended training MEDNODE_VGG11
wait


# echo "Started training KaggleMB_VGG11
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB KaggleMB --CLASSIFIER VGG11 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/KaggleMB_VGG11.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/KaggleMB_VGG11.err
# echo "Ended training KaggleMB_VGG11
wait


# echo "Started training ISIC2016+ISIC2017_VGG11
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 --CLASSIFIER VGG11 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/ISIC2016+ISIC2017_VGG11.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/ISIC2016+ISIC2017_VGG11.err
# echo "Ended training ISIC2016+ISIC2017_VGG11
wait


# echo "Started training ISIC2018+ISIC2019_VGG11
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2018 ISIC2019 --CLASSIFIER VGG11 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/ISIC2018+ISIC2019_VGG11.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/ISIC2018+ISIC2019_VGG11.err
# echo "Ended training ISIC2018+ISIC2019_VGG11
wait


# echo "Started training ISIC2020+PH2_VGG11
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 PH2 --CLASSIFIER VGG11 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/ISIC2020+PH2_VGG11.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/ISIC2020+PH2_VGG11.err
# echo "Ended training ISIC2020+PH2_VGG11
wait


# echo "Started training _7_point_criteria+PAD_UFES_20_VGG11
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 --CLASSIFIER VGG11 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/_7_point_criteria+PAD_UFES_20_VGG11.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/_7_point_criteria+PAD_UFES_20_VGG11.err
# echo "Ended training _7_point_criteria+PAD_UFES_20_VGG11
wait


# echo "Started training MEDNODE+KaggleMB_VGG11
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB MEDNODE KaggleMB --CLASSIFIER VGG11 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/MEDNODE+KaggleMB_VGG11.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/MEDNODE+KaggleMB_VGG11.err
# echo "Ended training MEDNODE+KaggleMB_VGG11
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018_VGG11
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 --CLASSIFIER VGG11 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/ISIC2016+ISIC2017+ISIC2018_VGG11.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/ISIC2016+ISIC2017+ISIC2018_VGG11.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018_VGG11
wait


# echo "Started training ISIC2019+ISIC2020+PH2_VGG11
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2019 ISIC2020 PH2 --CLASSIFIER VGG11 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/ISIC2019+ISIC2020+PH2_VGG11.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/ISIC2019+ISIC2020+PH2_VGG11.err
# echo "Ended training ISIC2019+ISIC2020+PH2_VGG11
wait


# echo "Started training _7_point_criteria+PAD_UFES_20+MEDNODE_VGG11
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 MEDNODE --CLASSIFIER VGG11 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/_7_point_criteria+PAD_UFES_20+MEDNODE_VGG11.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/_7_point_criteria+PAD_UFES_20+MEDNODE_VGG11.err
# echo "Ended training _7_point_criteria+PAD_UFES_20+MEDNODE_VGG11
wait


# echo "Started training PAD_UFES_20+MEDNODE+KaggleMB_VGG11
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER VGG11 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/PAD_UFES_20+MEDNODE+KaggleMB_VGG11.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/PAD_UFES_20+MEDNODE+KaggleMB_VGG11.err
# echo "Ended training PAD_UFES_20+MEDNODE+KaggleMB_VGG11
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019_VGG11
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 --CLASSIFIER VGG11 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/ISIC2016+ISIC2017+ISIC2018+ISIC2019_VGG11.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/ISIC2016+ISIC2017+ISIC2018+ISIC2019_VGG11.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019_VGG11
wait


# echo "Started training ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_VGG11
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER VGG11 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_VGG11.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_VGG11.err
# echo "Ended training ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_VGG11
wait


# echo "Started training _7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG11
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER VGG11 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG11.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG11.err
# echo "Ended training _7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG11
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_VGG11
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER VGG11 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_VGG11.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_VGG11.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_VGG11
wait


# echo "Started training PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG11
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER VGG11 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG11.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG11.err
# echo "Ended training PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG11
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_VGG11
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 --CLASSIFIER VGG11 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_VGG11.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_VGG11.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_VGG11
wait


# echo "Started training ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG11
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER VGG11 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG11.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG11.err
# echo "Ended training ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG11
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_VGG11
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria --CLASSIFIER VGG11 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_VGG11.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_VGG11.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_VGG11
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_VGG11
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER VGG11 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_VGG11.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_VGG11.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_VGG11
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_VGG11
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER VGG11 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_VGG11.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_VGG11.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_VGG11
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_VGG11
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE --CLASSIFIER VGG11 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_VGG11.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_VGG11.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_VGG11
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG11
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER VGG11 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG11.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG11/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG11.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG11
wait

