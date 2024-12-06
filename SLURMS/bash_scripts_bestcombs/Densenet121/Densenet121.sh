
export CUDA_VISIBLE_DEVICES=1,2


mkdir -p ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121
echo "Created SLURMS/LOGS/Densenet121"

# echo "Started training ISIC2020_Densenet121
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 --CLASSIFIER Densenet121 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/ISIC2020_Densenet121.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/ISIC2020_Densenet121.err
# echo "Ended training ISIC2020_Densenet121
wait


# echo "Started training ISIC2016+ISIC2020+PH2_Densenet121
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2020 PH2 --CLASSIFIER Densenet121 > ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/ISIC2016+ISIC2020+PH2_Densenet121.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/Densenet121/ISIC2016+ISIC2020+PH2_Densenet121.err
# echo "Ended training ISIC2016+ISIC2020+PH2_Densenet121
wait
