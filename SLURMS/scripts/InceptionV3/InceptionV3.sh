
export CUDA_VISIBLE_DEVICES=6


mkdir -p ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3
echo "Created SLURMS/LOGS/InceptionV3"


# echo "Started training ISIC2016_InceptionV3
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 --CLASSIFIER InceptionV3 > ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016_InceptionV3.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016_InceptionV3.err
# echo "Ended training ISIC2016_InceptionV3
wait


# echo "Started training ISIC2017_InceptionV3
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2017 --CLASSIFIER InceptionV3 > ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2017_InceptionV3.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2017_InceptionV3.err
# echo "Ended training ISIC2017_InceptionV3
wait


# echo "Started training ISIC2018_InceptionV3
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2018 --CLASSIFIER InceptionV3 > ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2018_InceptionV3.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2018_InceptionV3.err
# echo "Ended training ISIC2018_InceptionV3
wait


# echo "Started training ISIC2019_InceptionV3
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2019 --CLASSIFIER InceptionV3 > ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2019_InceptionV3.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2019_InceptionV3.err
# echo "Ended training ISIC2019_InceptionV3
wait


# echo "Started training ISIC2020_InceptionV3
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 --CLASSIFIER InceptionV3 > ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2020_InceptionV3.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2020_InceptionV3.err
# echo "Ended training ISIC2020_InceptionV3
wait


# echo "Started training _7_point_criteria_InceptionV3
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria --CLASSIFIER InceptionV3 > ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/_7_point_criteria_InceptionV3.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/_7_point_criteria_InceptionV3.err
# echo "Ended training _7_point_criteria_InceptionV3
wait


# echo "Started training PAD_UFES_20_InceptionV3
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PAD_UFES_20 --CLASSIFIER InceptionV3 > ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/PAD_UFES_20_InceptionV3.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/PAD_UFES_20_InceptionV3.err
# echo "Ended training PAD_UFES_20_InceptionV3
wait


# echo "Started training MEDNODE_InceptionV3
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB MEDNODE --CLASSIFIER InceptionV3 > ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/MEDNODE_InceptionV3.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/MEDNODE_InceptionV3.err
# echo "Ended training MEDNODE_InceptionV3
wait


# echo "Started training KaggleMB_InceptionV3
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB KaggleMB --CLASSIFIER InceptionV3 > ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/KaggleMB_InceptionV3.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/KaggleMB_InceptionV3.err
# echo "Ended training KaggleMB_InceptionV3
wait


# echo "Started training ISIC2016+ISIC2017_InceptionV3
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 --CLASSIFIER InceptionV3 > ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2017_InceptionV3.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2017_InceptionV3.err
# echo "Ended training ISIC2016+ISIC2017_InceptionV3
wait


# echo "Started training ISIC2018+ISIC2019_InceptionV3
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2018 ISIC2019 --CLASSIFIER InceptionV3 > ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2018+ISIC2019_InceptionV3.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2018+ISIC2019_InceptionV3.err
# echo "Ended training ISIC2018+ISIC2019_InceptionV3
wait


# echo "Started training ISIC2020+PH2_InceptionV3
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 PH2 --CLASSIFIER InceptionV3 > ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2020+PH2_InceptionV3.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2020+PH2_InceptionV3.err
# echo "Ended training ISIC2020+PH2_InceptionV3
wait


# echo "Started training _7_point_criteria+PAD_UFES_20_InceptionV3
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 --CLASSIFIER InceptionV3 > ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/_7_point_criteria+PAD_UFES_20_InceptionV3.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/_7_point_criteria+PAD_UFES_20_InceptionV3.err
# echo "Ended training _7_point_criteria+PAD_UFES_20_InceptionV3
wait


# echo "Started training MEDNODE+KaggleMB_InceptionV3
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB MEDNODE KaggleMB --CLASSIFIER InceptionV3 > ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/MEDNODE+KaggleMB_InceptionV3.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/MEDNODE+KaggleMB_InceptionV3.err
# echo "Ended training MEDNODE+KaggleMB_InceptionV3
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018_InceptionV3
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 --CLASSIFIER InceptionV3 > ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2017+ISIC2018_InceptionV3.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2017+ISIC2018_InceptionV3.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018_InceptionV3
wait


# echo "Started training ISIC2019+ISIC2020+PH2_InceptionV3
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2019 ISIC2020 PH2 --CLASSIFIER InceptionV3 > ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2019+ISIC2020+PH2_InceptionV3.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2019+ISIC2020+PH2_InceptionV3.err
# echo "Ended training ISIC2019+ISIC2020+PH2_InceptionV3
wait


# echo "Started training _7_point_criteria+PAD_UFES_20+MEDNODE_InceptionV3
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 MEDNODE --CLASSIFIER InceptionV3 > ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/_7_point_criteria+PAD_UFES_20+MEDNODE_InceptionV3.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/_7_point_criteria+PAD_UFES_20+MEDNODE_InceptionV3.err
# echo "Ended training _7_point_criteria+PAD_UFES_20+MEDNODE_InceptionV3
wait


# echo "Started training PAD_UFES_20+MEDNODE+KaggleMB_InceptionV3
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER InceptionV3 > ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/PAD_UFES_20+MEDNODE+KaggleMB_InceptionV3.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/PAD_UFES_20+MEDNODE+KaggleMB_InceptionV3.err
# echo "Ended training PAD_UFES_20+MEDNODE+KaggleMB_InceptionV3
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019_InceptionV3
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 --CLASSIFIER InceptionV3 > ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2017+ISIC2018+ISIC2019_InceptionV3.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2017+ISIC2018+ISIC2019_InceptionV3.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019_InceptionV3
wait


# echo "Started training ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_InceptionV3
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER InceptionV3 > ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_InceptionV3.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_InceptionV3.err
# echo "Ended training ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_InceptionV3
wait


# echo "Started training _7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_InceptionV3
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER InceptionV3 > ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_InceptionV3.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_InceptionV3.err
# echo "Ended training _7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_InceptionV3
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_InceptionV3
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER InceptionV3 > ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_InceptionV3.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_InceptionV3.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_InceptionV3
wait


# echo "Started training PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_InceptionV3
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER InceptionV3 > ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_InceptionV3.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_InceptionV3.err
# echo "Ended training PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_InceptionV3
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_InceptionV3
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 --CLASSIFIER InceptionV3 > ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_InceptionV3.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_InceptionV3.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_InceptionV3
wait


# echo "Started training ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_InceptionV3
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER InceptionV3 > ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_InceptionV3.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_InceptionV3.err
# echo "Ended training ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_InceptionV3
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_InceptionV3
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria --CLASSIFIER InceptionV3 > ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_InceptionV3.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_InceptionV3.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_InceptionV3
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_InceptionV3
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER InceptionV3 > ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_InceptionV3.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_InceptionV3.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_InceptionV3
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_InceptionV3
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER InceptionV3 > ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_InceptionV3.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_InceptionV3.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_InceptionV3
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_InceptionV3
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE --CLASSIFIER InceptionV3 > ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_InceptionV3.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_InceptionV3.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_InceptionV3
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_InceptionV3
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER InceptionV3 > ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_InceptionV3.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_InceptionV3.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_InceptionV3
wait

