
export CUDA_VISIBLE_DEVICES=2


mkdir -p ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201
echo "Created SLURMS/LOGS/Densenet201"


# echo "Started training ISIC2018_Densenet201
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2018 --CLASSIFIER Densenet201 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2018_Densenet201.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2018_Densenet201.err
# echo "Ended training ISIC2018_Densenet201
wait


# echo "Started training ISIC2019_Densenet201
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2019 --CLASSIFIER Densenet201 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2019_Densenet201.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2019_Densenet201.err
# echo "Ended training ISIC2019_Densenet201
wait


# echo "Started training ISIC2020_Densenet201
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 --CLASSIFIER Densenet201 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2020_Densenet201.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2020_Densenet201.err
# echo "Ended training ISIC2020_Densenet201
wait


# echo "Started training ISIC2016+ISIC2020_Densenet201
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2020 --CLASSIFIER Densenet201 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2016+ISIC2020_Densenet201.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2016+ISIC2020_Densenet201.err
# echo "Ended training ISIC2016+ISIC2020_Densenet201
wait


# echo "Started training ISIC2016+ISIC2020+PH2_Densenet201
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2020 PH2 --CLASSIFIER Densenet201 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2016+ISIC2020+PH2_Densenet201.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2016+ISIC2020+PH2_Densenet201.err
# echo "Ended training ISIC2016+ISIC2020+PH2_Densenet201
wait


# echo "Started training ISIC2016+ISIC2018+ISIC2019+ISIC2020_Densenet201
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER Densenet201 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2016+ISIC2018+ISIC2019+ISIC2020_Densenet201.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2016+ISIC2018+ISIC2019+ISIC2020_Densenet201.err
# echo "Ended training ISIC2016+ISIC2018+ISIC2019+ISIC2020_Densenet201
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_Densenet201
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER Densenet201 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_Densenet201.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_Densenet201.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_Densenet201
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_Densenet201
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 MEDNODE KaggleMB --CLASSIFIER Densenet201 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_Densenet201.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_Densenet201.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_Densenet201
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_Densenet201
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 PH2 _7_point_criteria --CLASSIFIER Densenet201 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_Densenet201.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_Densenet201.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_Densenet201
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_Densenet201
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2020 PH2 --CLASSIFIER Densenet201 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_Densenet201.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_Densenet201.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_Densenet201
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_Densenet201
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE --CLASSIFIER Densenet201 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_Densenet201.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_Densenet201.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_Densenet201
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_Densenet201
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 MEDNODE KaggleMB --CLASSIFIER Densenet201 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_Densenet201.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_Densenet201.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_Densenet201
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_Densenet201
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 --CLASSIFIER Densenet201 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_Densenet201.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_Densenet201.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_Densenet201
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_Densenet201
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria --CLASSIFIER Densenet201 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_Densenet201.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_Densenet201.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_Densenet201
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_Densenet201
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER Densenet201 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_Densenet201.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_Densenet201.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_Densenet201
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_Densenet201
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 _7_point_criteria PAD_UFES_20 --CLASSIFIER Densenet201 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_Densenet201.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_Densenet201.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_Densenet201
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_Densenet201
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PAD_UFES_20 MEDNODE --CLASSIFIER Densenet201 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_Densenet201.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_Densenet201.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_Densenet201
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_Densenet201
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 PAD_UFES_20 MEDNODE --CLASSIFIER Densenet201 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_Densenet201.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_Densenet201.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_Densenet201
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_Densenet201
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 MEDNODE KaggleMB --CLASSIFIER Densenet201 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_Densenet201.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_Densenet201.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_Densenet201
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_Densenet201
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER Densenet201 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_Densenet201.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_Densenet201.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_Densenet201
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_Densenet201
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria MEDNODE KaggleMB --CLASSIFIER Densenet201 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_Densenet201.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_Densenet201.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_Densenet201
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_Densenet201
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER Densenet201 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_Densenet201.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_Densenet201.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_Densenet201
wait

