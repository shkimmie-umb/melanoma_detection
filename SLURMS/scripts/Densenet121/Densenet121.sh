
export CUDA_VISIBLE_DEVICES=2


mkdir -p ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121
echo "Created SLURMS/LOGS/Densenet121"


# echo "Started training ISIC2016_Densenet121
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 --CLASSIFIER Densenet121 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/ISIC2016_Densenet121.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/ISIC2016_Densenet121.err
# echo "Ended training ISIC2016_Densenet121
wait


# echo "Started training ISIC2017_Densenet121
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2017 --CLASSIFIER Densenet121 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/ISIC2017_Densenet121.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/ISIC2017_Densenet121.err
# echo "Ended training ISIC2017_Densenet121
wait


# echo "Started training ISIC2018_Densenet121
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2018 --CLASSIFIER Densenet121 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/ISIC2018_Densenet121.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/ISIC2018_Densenet121.err
# echo "Ended training ISIC2018_Densenet121
wait


# echo "Started training ISIC2019_Densenet121
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2019 --CLASSIFIER Densenet121 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/ISIC2019_Densenet121.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/ISIC2019_Densenet121.err
# echo "Ended training ISIC2019_Densenet121
wait


# echo "Started training ISIC2020_Densenet121
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 --CLASSIFIER Densenet121 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/ISIC2020_Densenet121.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/ISIC2020_Densenet121.err
# echo "Ended training ISIC2020_Densenet121
wait


# echo "Started training _7_point_criteria_Densenet121
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria --CLASSIFIER Densenet121 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/_7_point_criteria_Densenet121.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/_7_point_criteria_Densenet121.err
# echo "Ended training _7_point_criteria_Densenet121
wait


# echo "Started training PAD_UFES_20_Densenet121
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PAD_UFES_20 --CLASSIFIER Densenet121 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/PAD_UFES_20_Densenet121.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/PAD_UFES_20_Densenet121.err
# echo "Ended training PAD_UFES_20_Densenet121
wait


# echo "Started training MEDNODE_Densenet121
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB MEDNODE --CLASSIFIER Densenet121 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/MEDNODE_Densenet121.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/MEDNODE_Densenet121.err
# echo "Ended training MEDNODE_Densenet121
wait


# echo "Started training KaggleMB_Densenet121
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB KaggleMB --CLASSIFIER Densenet121 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/KaggleMB_Densenet121.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/KaggleMB_Densenet121.err
# echo "Ended training KaggleMB_Densenet121
wait


# echo "Started training ISIC2016+ISIC2017_Densenet121
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 --CLASSIFIER Densenet121 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/ISIC2016+ISIC2017_Densenet121.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/ISIC2016+ISIC2017_Densenet121.err
# echo "Ended training ISIC2016+ISIC2017_Densenet121
wait


# echo "Started training ISIC2018+ISIC2019_Densenet121
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2018 ISIC2019 --CLASSIFIER Densenet121 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/ISIC2018+ISIC2019_Densenet121.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/ISIC2018+ISIC2019_Densenet121.err
# echo "Ended training ISIC2018+ISIC2019_Densenet121
wait


# echo "Started training ISIC2020+PH2_Densenet121
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 PH2 --CLASSIFIER Densenet121 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/ISIC2020+PH2_Densenet121.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/ISIC2020+PH2_Densenet121.err
# echo "Ended training ISIC2020+PH2_Densenet121
wait


# echo "Started training _7_point_criteria+PAD_UFES_20_Densenet121
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 --CLASSIFIER Densenet121 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/_7_point_criteria+PAD_UFES_20_Densenet121.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/_7_point_criteria+PAD_UFES_20_Densenet121.err
# echo "Ended training _7_point_criteria+PAD_UFES_20_Densenet121
wait


# echo "Started training MEDNODE+KaggleMB_Densenet121
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB MEDNODE KaggleMB --CLASSIFIER Densenet121 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/MEDNODE+KaggleMB_Densenet121.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/MEDNODE+KaggleMB_Densenet121.err
# echo "Ended training MEDNODE+KaggleMB_Densenet121
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018_Densenet121
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 --CLASSIFIER Densenet121 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/ISIC2016+ISIC2017+ISIC2018_Densenet121.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/ISIC2016+ISIC2017+ISIC2018_Densenet121.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018_Densenet121
wait


# echo "Started training ISIC2019+ISIC2020+PH2_Densenet121
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2019 ISIC2020 PH2 --CLASSIFIER Densenet121 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/ISIC2019+ISIC2020+PH2_Densenet121.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/ISIC2019+ISIC2020+PH2_Densenet121.err
# echo "Ended training ISIC2019+ISIC2020+PH2_Densenet121
wait


# echo "Started training _7_point_criteria+PAD_UFES_20+MEDNODE_Densenet121
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 MEDNODE --CLASSIFIER Densenet121 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/_7_point_criteria+PAD_UFES_20+MEDNODE_Densenet121.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/_7_point_criteria+PAD_UFES_20+MEDNODE_Densenet121.err
# echo "Ended training _7_point_criteria+PAD_UFES_20+MEDNODE_Densenet121
wait


# echo "Started training PAD_UFES_20+MEDNODE+KaggleMB_Densenet121
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER Densenet121 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/PAD_UFES_20+MEDNODE+KaggleMB_Densenet121.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/PAD_UFES_20+MEDNODE+KaggleMB_Densenet121.err
# echo "Ended training PAD_UFES_20+MEDNODE+KaggleMB_Densenet121
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019_Densenet121
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 --CLASSIFIER Densenet121 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/ISIC2016+ISIC2017+ISIC2018+ISIC2019_Densenet121.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/ISIC2016+ISIC2017+ISIC2018+ISIC2019_Densenet121.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019_Densenet121
wait


# echo "Started training ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_Densenet121
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER Densenet121 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_Densenet121.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_Densenet121.err
# echo "Ended training ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_Densenet121
wait


# echo "Started training _7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_Densenet121
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER Densenet121 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_Densenet121.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_Densenet121.err
# echo "Ended training _7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_Densenet121
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_Densenet121
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER Densenet121 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_Densenet121.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_Densenet121.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_Densenet121
wait


# echo "Started training PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_Densenet121
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER Densenet121 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_Densenet121.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_Densenet121.err
# echo "Ended training PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_Densenet121
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_Densenet121
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 --CLASSIFIER Densenet121 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_Densenet121.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_Densenet121.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_Densenet121
wait


# echo "Started training ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_Densenet121
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER Densenet121 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_Densenet121.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_Densenet121.err
# echo "Ended training ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_Densenet121
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_Densenet121
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria --CLASSIFIER Densenet121 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_Densenet121.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_Densenet121.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_Densenet121
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_Densenet121
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER Densenet121 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_Densenet121.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_Densenet121.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_Densenet121
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_Densenet121
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER Densenet121 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_Densenet121.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_Densenet121.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_Densenet121
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_Densenet121
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE --CLASSIFIER Densenet121 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_Densenet121.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_Densenet121.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_Densenet121
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_Densenet121
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER Densenet121 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_Densenet121.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_Densenet121.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_Densenet121
wait

