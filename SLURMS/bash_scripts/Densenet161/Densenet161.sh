
export CUDA_VISIBLE_DEVICES=3


mkdir -p ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161
echo "Created SLURMS/LOGS/Densenet161"


# echo "Started training ISIC2018_Densenet161
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2018 --CLASSIFIER Densenet161 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2018_Densenet161.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2018_Densenet161.err
# echo "Ended training ISIC2018_Densenet161
wait


# echo "Started training ISIC2019_Densenet161
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2019 --CLASSIFIER Densenet161 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2019_Densenet161.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2019_Densenet161.err
# echo "Ended training ISIC2019_Densenet161
wait


# echo "Started training ISIC2020_Densenet161
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 --CLASSIFIER Densenet161 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2020_Densenet161.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2020_Densenet161.err
# echo "Ended training ISIC2020_Densenet161
wait


# echo "Started training ISIC2016+ISIC2020_Densenet161
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2020 --CLASSIFIER Densenet161 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2020_Densenet161.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2020_Densenet161.err
# echo "Ended training ISIC2016+ISIC2020_Densenet161
wait


# echo "Started training ISIC2016+ISIC2020+PH2_Densenet161
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2020 PH2 --CLASSIFIER Densenet161 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2020+PH2_Densenet161.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2020+PH2_Densenet161.err
# echo "Ended training ISIC2016+ISIC2020+PH2_Densenet161
wait


# echo "Started training ISIC2016+ISIC2018+ISIC2019+ISIC2020_Densenet161
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER Densenet161 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2018+ISIC2019+ISIC2020_Densenet161.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2018+ISIC2019+ISIC2020_Densenet161.err
# echo "Ended training ISIC2016+ISIC2018+ISIC2019+ISIC2020_Densenet161
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_Densenet161
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER Densenet161 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_Densenet161.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_Densenet161.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_Densenet161
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_Densenet161
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 MEDNODE KaggleMB --CLASSIFIER Densenet161 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_Densenet161.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_Densenet161.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_Densenet161
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_Densenet161
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 PH2 _7_point_criteria --CLASSIFIER Densenet161 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_Densenet161.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_Densenet161.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_Densenet161
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_Densenet161
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2020 PH2 --CLASSIFIER Densenet161 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_Densenet161.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_Densenet161.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_Densenet161
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_Densenet161
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE --CLASSIFIER Densenet161 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_Densenet161.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_Densenet161.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_Densenet161
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_Densenet161
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 MEDNODE KaggleMB --CLASSIFIER Densenet161 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_Densenet161.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_Densenet161.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_Densenet161
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_Densenet161
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 --CLASSIFIER Densenet161 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_Densenet161.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_Densenet161.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_Densenet161
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_Densenet161
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria --CLASSIFIER Densenet161 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_Densenet161.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_Densenet161.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_Densenet161
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_Densenet161
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER Densenet161 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_Densenet161.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_Densenet161.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_Densenet161
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_Densenet161
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 _7_point_criteria PAD_UFES_20 --CLASSIFIER Densenet161 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_Densenet161.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_Densenet161.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_Densenet161
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_Densenet161
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PAD_UFES_20 MEDNODE --CLASSIFIER Densenet161 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_Densenet161.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_Densenet161.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_Densenet161
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_Densenet161
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 PAD_UFES_20 MEDNODE --CLASSIFIER Densenet161 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_Densenet161.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_Densenet161.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_Densenet161
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_Densenet161
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 MEDNODE KaggleMB --CLASSIFIER Densenet161 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_Densenet161.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_Densenet161.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_Densenet161
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_Densenet161
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER Densenet161 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_Densenet161.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_Densenet161.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_Densenet161
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_Densenet161
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria MEDNODE KaggleMB --CLASSIFIER Densenet161 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_Densenet161.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_Densenet161.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_Densenet161
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_Densenet161
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER Densenet161 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_Densenet161.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_Densenet161.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_Densenet161
wait

