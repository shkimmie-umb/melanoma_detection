
export CUDA_VISIBLE_DEVICES=2


mkdir -p ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB2
echo "Created SLURMS/LOGS/EfficientNetB2"






# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_EfficientNetB2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER EfficientNetB2 > ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_EfficientNetB2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_EfficientNetB2.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_EfficientNetB2
wait


# echo "Started training ISIC2016+ISIC2020_EfficientNetB2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2020 --CLASSIFIER EfficientNetB2 > ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB2/ISIC2016+ISIC2020_EfficientNetB2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB2/ISIC2016+ISIC2020_EfficientNetB2.err
# echo "Ended training ISIC2016+ISIC2020_EfficientNetB2
wait