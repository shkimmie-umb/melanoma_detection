
export CUDA_VISIBLE_DEVICES=3


mkdir -p ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161
echo "Created SLURMS/LOGS/Densenet161"


# echo "Started training ISIC2016_Densenet161
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 --CLASSIFIER Densenet161 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016_Densenet161.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016_Densenet161.err
# echo "Ended training ISIC2016_Densenet161
wait


# echo "Started training ISIC2017_Densenet161
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2017 --CLASSIFIER Densenet161 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2017_Densenet161.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2017_Densenet161.err
# echo "Ended training ISIC2017_Densenet161
wait


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


# echo "Started training _7_point_criteria_Densenet161
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria --CLASSIFIER Densenet161 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/_7_point_criteria_Densenet161.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/_7_point_criteria_Densenet161.err
# echo "Ended training _7_point_criteria_Densenet161
wait


# echo "Started training PAD_UFES_20_Densenet161
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PAD_UFES_20 --CLASSIFIER Densenet161 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/PAD_UFES_20_Densenet161.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/PAD_UFES_20_Densenet161.err
# echo "Ended training PAD_UFES_20_Densenet161
wait


# echo "Started training MEDNODE_Densenet161
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB MEDNODE --CLASSIFIER Densenet161 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/MEDNODE_Densenet161.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/MEDNODE_Densenet161.err
# echo "Ended training MEDNODE_Densenet161
wait


# echo "Started training KaggleMB_Densenet161
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB KaggleMB --CLASSIFIER Densenet161 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/KaggleMB_Densenet161.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/KaggleMB_Densenet161.err
# echo "Ended training KaggleMB_Densenet161
wait


# echo "Started training ISIC2016+ISIC2017_Densenet161
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 --CLASSIFIER Densenet161 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2017_Densenet161.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2017_Densenet161.err
# echo "Ended training ISIC2016+ISIC2017_Densenet161
wait


# echo "Started training ISIC2018+ISIC2019_Densenet161
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2018 ISIC2019 --CLASSIFIER Densenet161 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2018+ISIC2019_Densenet161.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2018+ISIC2019_Densenet161.err
# echo "Ended training ISIC2018+ISIC2019_Densenet161
wait


# echo "Started training ISIC2020+PH2_Densenet161
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 PH2 --CLASSIFIER Densenet161 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2020+PH2_Densenet161.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2020+PH2_Densenet161.err
# echo "Ended training ISIC2020+PH2_Densenet161
wait


# echo "Started training _7_point_criteria+PAD_UFES_20_Densenet161
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 --CLASSIFIER Densenet161 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/_7_point_criteria+PAD_UFES_20_Densenet161.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/_7_point_criteria+PAD_UFES_20_Densenet161.err
# echo "Ended training _7_point_criteria+PAD_UFES_20_Densenet161
wait


# echo "Started training MEDNODE+KaggleMB_Densenet161
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB MEDNODE KaggleMB --CLASSIFIER Densenet161 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/MEDNODE+KaggleMB_Densenet161.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/MEDNODE+KaggleMB_Densenet161.err
# echo "Ended training MEDNODE+KaggleMB_Densenet161
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018_Densenet161
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 --CLASSIFIER Densenet161 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2017+ISIC2018_Densenet161.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2017+ISIC2018_Densenet161.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018_Densenet161
wait


# echo "Started training ISIC2019+ISIC2020+PH2_Densenet161
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2019 ISIC2020 PH2 --CLASSIFIER Densenet161 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2019+ISIC2020+PH2_Densenet161.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2019+ISIC2020+PH2_Densenet161.err
# echo "Ended training ISIC2019+ISIC2020+PH2_Densenet161
wait


# echo "Started training _7_point_criteria+PAD_UFES_20+MEDNODE_Densenet161
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 MEDNODE --CLASSIFIER Densenet161 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/_7_point_criteria+PAD_UFES_20+MEDNODE_Densenet161.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/_7_point_criteria+PAD_UFES_20+MEDNODE_Densenet161.err
# echo "Ended training _7_point_criteria+PAD_UFES_20+MEDNODE_Densenet161
wait


# echo "Started training PAD_UFES_20+MEDNODE+KaggleMB_Densenet161
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER Densenet161 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/PAD_UFES_20+MEDNODE+KaggleMB_Densenet161.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/PAD_UFES_20+MEDNODE+KaggleMB_Densenet161.err
# echo "Ended training PAD_UFES_20+MEDNODE+KaggleMB_Densenet161
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019_Densenet161
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 --CLASSIFIER Densenet161 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2017+ISIC2018+ISIC2019_Densenet161.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2017+ISIC2018+ISIC2019_Densenet161.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019_Densenet161
wait


# echo "Started training ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_Densenet161
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER Densenet161 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_Densenet161.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_Densenet161.err
# echo "Ended training ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_Densenet161
wait


# echo "Started training _7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_Densenet161
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER Densenet161 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_Densenet161.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_Densenet161.err
# echo "Ended training _7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_Densenet161
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_Densenet161
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER Densenet161 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_Densenet161.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_Densenet161.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_Densenet161
wait


# echo "Started training PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_Densenet161
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER Densenet161 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_Densenet161.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_Densenet161.err
# echo "Ended training PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_Densenet161
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_Densenet161
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 --CLASSIFIER Densenet161 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_Densenet161.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_Densenet161.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_Densenet161
wait


# echo "Started training ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_Densenet161
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER Densenet161 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_Densenet161.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_Densenet161.err
# echo "Ended training ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_Densenet161
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_Densenet161
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria --CLASSIFIER Densenet161 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_Densenet161.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_Densenet161.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_Densenet161
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_Densenet161
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER Densenet161 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_Densenet161.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_Densenet161.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_Densenet161
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_Densenet161
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER Densenet161 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_Densenet161.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_Densenet161.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_Densenet161
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_Densenet161
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE --CLASSIFIER Densenet161 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_Densenet161.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_Densenet161.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_Densenet161
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_Densenet161
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER Densenet161 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_Densenet161.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet161/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_Densenet161.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_Densenet161
wait

