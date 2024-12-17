
export CUDA_VISIBLE_DEVICES=0


mkdir -p ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152
echo "Created SLURMS/LOGS/ResNet152"

# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_ResNet152
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 --CLASSIFIER ResNet152 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_ResNet152.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_ResNet152.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_ResNet152
wait

# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ResNet152
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER ResNet152 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ResNet152.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ResNet152.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ResNet152
wait

# echo "Started training ISIC2019_ResNet152
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2019 --CLASSIFIER ResNet152 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/ISIC2019_ResNet152.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/ISIC2019_ResNet152.err
# echo "Ended training ISIC2019_ResNet152
wait

# echo "Started training ISIC2016+ISIC2018+ISIC2019+ISIC2020_ResNet152
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER ResNet152 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/ISIC2016+ISIC2018+ISIC2019+ISIC2020_ResNet152.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/ISIC2016+ISIC2018+ISIC2019+ISIC2020_ResNet152.err
# echo "Ended training ISIC2016+ISIC2018+ISIC2019+ISIC2020_ResNet152
wait

# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_ResNet152
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 MEDNODE KaggleMB --CLASSIFIER ResNet152 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_ResNet152.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet152/ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_ResNet152.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_ResNet152
wait