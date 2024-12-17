
export CUDA_VISIBLE_DEVICES=7


mkdir -p ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB1
echo "Created SLURMS/LOGS/EfficientNetB1"

# echo "Started training ISIC2016+ISIC2018+ISIC2019+ISIC2020_EfficientNetB1
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER EfficientNetB1 > ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB1/ISIC2016+ISIC2018+ISIC2019+ISIC2020_EfficientNetB1.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB1/ISIC2016+ISIC2018+ISIC2019+ISIC2020_EfficientNetB1.err
# echo "Ended training ISIC2016+ISIC2018+ISIC2019+ISIC2020_EfficientNetB1
wait

# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_EfficientNetB1
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PAD_UFES_20 MEDNODE --CLASSIFIER EfficientNetB1 > ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB1/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_EfficientNetB1.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB1/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_EfficientNetB1.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_EfficientNetB1
wait