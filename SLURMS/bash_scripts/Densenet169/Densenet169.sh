
export CUDA_VISIBLE_DEVICES=1


mkdir -p ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet169
echo "Created SLURMS/LOGS/Densenet169"


# echo "Started training ISIC2018_Densenet169
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2018 --CLASSIFIER Densenet169 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet169/ISIC2018_Densenet169.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet169/ISIC2018_Densenet169.err
# echo "Ended training ISIC2018_Densenet169
wait


# echo "Started training ISIC2019_Densenet169
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2019 --CLASSIFIER Densenet169 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet169/ISIC2019_Densenet169.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet169/ISIC2019_Densenet169.err
# echo "Ended training ISIC2019_Densenet169
wait


# echo "Started training ISIC2020_Densenet169
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 --CLASSIFIER Densenet169 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet169/ISIC2020_Densenet169.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet169/ISIC2020_Densenet169.err
# echo "Ended training ISIC2020_Densenet169
wait


# echo "Started training ISIC2016+ISIC2020_Densenet169
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2020 --CLASSIFIER Densenet169 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet169/ISIC2016+ISIC2020_Densenet169.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet169/ISIC2016+ISIC2020_Densenet169.err
# echo "Ended training ISIC2016+ISIC2020_Densenet169
wait


# echo "Started training ISIC2016+ISIC2020+PH2_Densenet169
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2020 PH2 --CLASSIFIER Densenet169 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet169/ISIC2016+ISIC2020+PH2_Densenet169.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet169/ISIC2016+ISIC2020+PH2_Densenet169.err
# echo "Ended training ISIC2016+ISIC2020+PH2_Densenet169
wait


# echo "Started training ISIC2016+ISIC2018+ISIC2019+ISIC2020_Densenet169
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER Densenet169 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet169/ISIC2016+ISIC2018+ISIC2019+ISIC2020_Densenet169.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet169/ISIC2016+ISIC2018+ISIC2019+ISIC2020_Densenet169.err
# echo "Ended training ISIC2016+ISIC2018+ISIC2019+ISIC2020_Densenet169
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_Densenet169
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER Densenet169 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet169/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_Densenet169.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet169/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_Densenet169.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_Densenet169
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_Densenet169
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 MEDNODE KaggleMB --CLASSIFIER Densenet169 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet169/ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_Densenet169.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet169/ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_Densenet169.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_Densenet169
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_Densenet169
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 PH2 _7_point_criteria --CLASSIFIER Densenet169 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet169/ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_Densenet169.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet169/ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_Densenet169.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_Densenet169
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_Densenet169
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2020 PH2 --CLASSIFIER Densenet169 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet169/ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_Densenet169.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet169/ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_Densenet169.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_Densenet169
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_Densenet169
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE --CLASSIFIER Densenet169 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet169/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_Densenet169.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet169/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_Densenet169.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_Densenet169
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_Densenet169
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 MEDNODE KaggleMB --CLASSIFIER Densenet169 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet169/ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_Densenet169.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet169/ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_Densenet169.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_Densenet169
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_Densenet169
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 --CLASSIFIER Densenet169 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet169/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_Densenet169.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet169/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_Densenet169.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_Densenet169
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_Densenet169
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria --CLASSIFIER Densenet169 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet169/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_Densenet169.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet169/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_Densenet169.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_Densenet169
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_Densenet169
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER Densenet169 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet169/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_Densenet169.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet169/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_Densenet169.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_Densenet169
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_Densenet169
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 _7_point_criteria PAD_UFES_20 --CLASSIFIER Densenet169 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet169/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_Densenet169.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet169/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_Densenet169.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_Densenet169
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_Densenet169
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PAD_UFES_20 MEDNODE --CLASSIFIER Densenet169 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet169/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_Densenet169.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet169/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_Densenet169.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_Densenet169
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_Densenet169
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 PAD_UFES_20 MEDNODE --CLASSIFIER Densenet169 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet169/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_Densenet169.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet169/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_Densenet169.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_Densenet169
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_Densenet169
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 MEDNODE KaggleMB --CLASSIFIER Densenet169 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet169/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_Densenet169.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet169/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_Densenet169.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_Densenet169
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_Densenet169
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER Densenet169 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet169/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_Densenet169.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet169/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_Densenet169.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_Densenet169
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_Densenet169
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria MEDNODE KaggleMB --CLASSIFIER Densenet169 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet169/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_Densenet169.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet169/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_Densenet169.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_Densenet169
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_Densenet169
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER Densenet169 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet169/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_Densenet169.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet169/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_Densenet169.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_Densenet169
wait

