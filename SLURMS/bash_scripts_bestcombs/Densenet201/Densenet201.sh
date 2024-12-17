
export CUDA_VISIBLE_DEVICES=3,4


mkdir -p ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201
echo "Created SLURMS/LOGS/Densenet201"

# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_Densenet201
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE --CLASSIFIER Densenet201 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_Densenet201.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_Densenet201.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_Densenet201
wait

# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_Densenet201
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 _7_point_criteria PAD_UFES_20 --CLASSIFIER Densenet201 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_Densenet201.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_Densenet201.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_Densenet201
wait

# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_Densenet201
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 MEDNODE KaggleMB --CLASSIFIER Densenet201 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_Densenet201.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_Densenet201.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_Densenet201
wait

# echo "Started training ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_Densenet201
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 MEDNODE KaggleMB --CLASSIFIER Densenet201 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_Densenet201.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_Densenet201.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_Densenet201
wait

# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_Densenet201
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 MEDNODE KaggleMB --CLASSIFIER Densenet201 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_Densenet201.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet201/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_Densenet201.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_Densenet201
wait
