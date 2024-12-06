
export CUDA_VISIBLE_DEVICES=1


mkdir -p ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB6
echo "Created SLURMS/LOGS/EfficientNetB6"

# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_EfficientNetB6
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2020 PH2 --CLASSIFIER EfficientNetB6 > ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB6/ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_EfficientNetB6.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB6/ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_EfficientNetB6.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_EfficientNetB6
wait