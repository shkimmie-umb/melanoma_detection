
export CUDA_VISIBLE_DEVICES=2


mkdir -p ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13
echo "Created SLURMS/LOGS/VGG13"


# echo "Started training ISIC2016_VGG13
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 --CLASSIFIER VGG13 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/ISIC2016_VGG13.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/ISIC2016_VGG13.err
# echo "Ended training ISIC2016_VGG13
wait


# echo "Started training ISIC2017_VGG13
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2017 --CLASSIFIER VGG13 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/ISIC2017_VGG13.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/ISIC2017_VGG13.err
# echo "Ended training ISIC2017_VGG13
wait


# echo "Started training ISIC2018_VGG13
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2018 --CLASSIFIER VGG13 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/ISIC2018_VGG13.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/ISIC2018_VGG13.err
# echo "Ended training ISIC2018_VGG13
wait


# echo "Started training ISIC2019_VGG13
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2019 --CLASSIFIER VGG13 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/ISIC2019_VGG13.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/ISIC2019_VGG13.err
# echo "Ended training ISIC2019_VGG13
wait


# echo "Started training ISIC2020_VGG13
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 --CLASSIFIER VGG13 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/ISIC2020_VGG13.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/ISIC2020_VGG13.err
# echo "Ended training ISIC2020_VGG13
wait


# echo "Started training _7_point_criteria_VGG13
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria --CLASSIFIER VGG13 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/_7_point_criteria_VGG13.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/_7_point_criteria_VGG13.err
# echo "Ended training _7_point_criteria_VGG13
wait


# echo "Started training PAD_UFES_20_VGG13
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PAD_UFES_20 --CLASSIFIER VGG13 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/PAD_UFES_20_VGG13.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/PAD_UFES_20_VGG13.err
# echo "Ended training PAD_UFES_20_VGG13
wait


# echo "Started training MEDNODE_VGG13
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB MEDNODE --CLASSIFIER VGG13 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/MEDNODE_VGG13.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/MEDNODE_VGG13.err
# echo "Ended training MEDNODE_VGG13
wait


# echo "Started training KaggleMB_VGG13
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB KaggleMB --CLASSIFIER VGG13 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/KaggleMB_VGG13.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/KaggleMB_VGG13.err
# echo "Ended training KaggleMB_VGG13
wait


# echo "Started training ISIC2016+ISIC2017_VGG13
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 --CLASSIFIER VGG13 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/ISIC2016+ISIC2017_VGG13.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/ISIC2016+ISIC2017_VGG13.err
# echo "Ended training ISIC2016+ISIC2017_VGG13
wait


# echo "Started training ISIC2018+ISIC2019_VGG13
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2018 ISIC2019 --CLASSIFIER VGG13 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/ISIC2018+ISIC2019_VGG13.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/ISIC2018+ISIC2019_VGG13.err
# echo "Ended training ISIC2018+ISIC2019_VGG13
wait


# echo "Started training ISIC2020+PH2_VGG13
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 PH2 --CLASSIFIER VGG13 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/ISIC2020+PH2_VGG13.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/ISIC2020+PH2_VGG13.err
# echo "Ended training ISIC2020+PH2_VGG13
wait


# echo "Started training _7_point_criteria+PAD_UFES_20_VGG13
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 --CLASSIFIER VGG13 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/_7_point_criteria+PAD_UFES_20_VGG13.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/_7_point_criteria+PAD_UFES_20_VGG13.err
# echo "Ended training _7_point_criteria+PAD_UFES_20_VGG13
wait


# echo "Started training MEDNODE+KaggleMB_VGG13
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB MEDNODE KaggleMB --CLASSIFIER VGG13 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/MEDNODE+KaggleMB_VGG13.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/MEDNODE+KaggleMB_VGG13.err
# echo "Ended training MEDNODE+KaggleMB_VGG13
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018_VGG13
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 --CLASSIFIER VGG13 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/ISIC2016+ISIC2017+ISIC2018_VGG13.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/ISIC2016+ISIC2017+ISIC2018_VGG13.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018_VGG13
wait


# echo "Started training ISIC2019+ISIC2020+PH2_VGG13
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2019 ISIC2020 PH2 --CLASSIFIER VGG13 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/ISIC2019+ISIC2020+PH2_VGG13.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/ISIC2019+ISIC2020+PH2_VGG13.err
# echo "Ended training ISIC2019+ISIC2020+PH2_VGG13
wait


# echo "Started training _7_point_criteria+PAD_UFES_20+MEDNODE_VGG13
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 MEDNODE --CLASSIFIER VGG13 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/_7_point_criteria+PAD_UFES_20+MEDNODE_VGG13.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/_7_point_criteria+PAD_UFES_20+MEDNODE_VGG13.err
# echo "Ended training _7_point_criteria+PAD_UFES_20+MEDNODE_VGG13
wait


# echo "Started training PAD_UFES_20+MEDNODE+KaggleMB_VGG13
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER VGG13 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/PAD_UFES_20+MEDNODE+KaggleMB_VGG13.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/PAD_UFES_20+MEDNODE+KaggleMB_VGG13.err
# echo "Ended training PAD_UFES_20+MEDNODE+KaggleMB_VGG13
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019_VGG13
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 --CLASSIFIER VGG13 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/ISIC2016+ISIC2017+ISIC2018+ISIC2019_VGG13.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/ISIC2016+ISIC2017+ISIC2018+ISIC2019_VGG13.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019_VGG13
wait


# echo "Started training ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_VGG13
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER VGG13 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_VGG13.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_VGG13.err
# echo "Ended training ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_VGG13
wait


# echo "Started training _7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG13
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER VGG13 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG13.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG13.err
# echo "Ended training _7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG13
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_VGG13
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER VGG13 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_VGG13.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_VGG13.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_VGG13
wait


# echo "Started training PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG13
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER VGG13 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG13.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG13.err
# echo "Ended training PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG13
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_VGG13
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 --CLASSIFIER VGG13 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_VGG13.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_VGG13.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_VGG13
wait


# echo "Started training ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG13
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER VGG13 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG13.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG13.err
# echo "Ended training ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG13
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_VGG13
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria --CLASSIFIER VGG13 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_VGG13.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_VGG13.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_VGG13
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_VGG13
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER VGG13 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_VGG13.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_VGG13.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_VGG13
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_VGG13
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER VGG13 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_VGG13.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_VGG13.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_VGG13
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_VGG13
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE --CLASSIFIER VGG13 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_VGG13.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_VGG13.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_VGG13
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG13
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER VGG13 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG13.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG13/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG13.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG13
wait

