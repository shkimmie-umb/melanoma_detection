
export CUDA_VISIBLE_DEVICES=3


mkdir -p ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet50
echo "Created SLURMS/LOGS/ResNet50"

# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ResNet50
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER ResNet50 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet50/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ResNet50.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet50/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ResNet50.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ResNet50
wait

# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_ResNet50
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE --CLASSIFIER ResNet50 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet50/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_ResNet50.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet50/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_ResNet50.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_ResNet50
wait

# echo "Started training ISIC2018_ResNet50
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2018 --CLASSIFIER ResNet50 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet50/ISIC2018_ResNet50.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet50/ISIC2018_ResNet50.err
# echo "Ended training ISIC2018_ResNet50
wait

# echo "Started training ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_ResNet50
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 MEDNODE KaggleMB --CLASSIFIER ResNet50 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet50/ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_ResNet50.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet50/ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_ResNet50.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_ResNet50
wait

# echo "Started training ISIC2020_ResNet50
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 --CLASSIFIER ResNet50 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet50/ISIC2020_ResNet50.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet50/ISIC2020_ResNet50.err
# echo "Ended training ISIC2020_ResNet50
wait

# echo "Started training ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_ResNet50
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 PH2 _7_point_criteria --CLASSIFIER ResNet50 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet50/ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_ResNet50.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet50/ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_ResNet50.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_ResNet50
wait
