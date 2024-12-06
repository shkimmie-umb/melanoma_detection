
export CUDA_VISIBLE_DEVICES=4


mkdir -p ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3
echo "Created SLURMS/LOGS/InceptionV3"


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


# echo "Started training ISIC2016+ISIC2020_InceptionV3
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2020 --CLASSIFIER InceptionV3 > ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2020_InceptionV3.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2020_InceptionV3.err
# echo "Ended training ISIC2016+ISIC2020_InceptionV3
wait


# echo "Started training ISIC2016+ISIC2020+PH2_InceptionV3
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2020 PH2 --CLASSIFIER InceptionV3 > ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2020+PH2_InceptionV3.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2020+PH2_InceptionV3.err
# echo "Ended training ISIC2016+ISIC2020+PH2_InceptionV3
wait


# echo "Started training ISIC2016+ISIC2018+ISIC2019+ISIC2020_InceptionV3
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER InceptionV3 > ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2018+ISIC2019+ISIC2020_InceptionV3.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2018+ISIC2019+ISIC2020_InceptionV3.err
# echo "Ended training ISIC2016+ISIC2018+ISIC2019+ISIC2020_InceptionV3
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_InceptionV3
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER InceptionV3 > ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_InceptionV3.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_InceptionV3.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_InceptionV3
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_InceptionV3
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 MEDNODE KaggleMB --CLASSIFIER InceptionV3 > ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_InceptionV3.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_InceptionV3.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_InceptionV3
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_InceptionV3
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 PH2 _7_point_criteria --CLASSIFIER InceptionV3 > ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_InceptionV3.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_InceptionV3.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_InceptionV3
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_InceptionV3
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2020 PH2 --CLASSIFIER InceptionV3 > ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_InceptionV3.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_InceptionV3.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_InceptionV3
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_InceptionV3
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE --CLASSIFIER InceptionV3 > ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_InceptionV3.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_InceptionV3.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_InceptionV3
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_InceptionV3
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 MEDNODE KaggleMB --CLASSIFIER InceptionV3 > ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_InceptionV3.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_InceptionV3.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_InceptionV3
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_InceptionV3
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 --CLASSIFIER InceptionV3 > ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_InceptionV3.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_InceptionV3.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_InceptionV3
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_InceptionV3
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria --CLASSIFIER InceptionV3 > ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_InceptionV3.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_InceptionV3.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_InceptionV3
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_InceptionV3
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER InceptionV3 > ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_InceptionV3.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_InceptionV3.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_InceptionV3
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_InceptionV3
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 _7_point_criteria PAD_UFES_20 --CLASSIFIER InceptionV3 > ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_InceptionV3.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_InceptionV3.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_InceptionV3
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_InceptionV3
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PAD_UFES_20 MEDNODE --CLASSIFIER InceptionV3 > ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_InceptionV3.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_InceptionV3.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_InceptionV3
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_InceptionV3
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 PAD_UFES_20 MEDNODE --CLASSIFIER InceptionV3 > ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_InceptionV3.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_InceptionV3.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_InceptionV3
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_InceptionV3
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 MEDNODE KaggleMB --CLASSIFIER InceptionV3 > ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_InceptionV3.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_InceptionV3.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_InceptionV3
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_InceptionV3
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER InceptionV3 > ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_InceptionV3.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_InceptionV3.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_InceptionV3
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_InceptionV3
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria MEDNODE KaggleMB --CLASSIFIER InceptionV3 > ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_InceptionV3.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_InceptionV3.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_InceptionV3
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_InceptionV3
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER InceptionV3 > ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_InceptionV3.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/InceptionV3/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_InceptionV3.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_InceptionV3
wait

