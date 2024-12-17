
export CUDA_VISIBLE_DEVICES=2


mkdir -p ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB2
echo "Created SLURMS/LOGS/EfficientNetB2"


# echo "Started training ISIC2018_EfficientNetB2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2018 --CLASSIFIER EfficientNetB2 > ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB2/ISIC2018_EfficientNetB2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB2/ISIC2018_EfficientNetB2.err
# echo "Ended training ISIC2018_EfficientNetB2
wait


# echo "Started training ISIC2019_EfficientNetB2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2019 --CLASSIFIER EfficientNetB2 > ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB2/ISIC2019_EfficientNetB2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB2/ISIC2019_EfficientNetB2.err
# echo "Ended training ISIC2019_EfficientNetB2
wait


# echo "Started training ISIC2020_EfficientNetB2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 --CLASSIFIER EfficientNetB2 > ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB2/ISIC2020_EfficientNetB2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB2/ISIC2020_EfficientNetB2.err
# echo "Ended training ISIC2020_EfficientNetB2
wait


# echo "Started training ISIC2016+ISIC2020_EfficientNetB2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2020 --CLASSIFIER EfficientNetB2 > ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB2/ISIC2016+ISIC2020_EfficientNetB2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB2/ISIC2016+ISIC2020_EfficientNetB2.err
# echo "Ended training ISIC2016+ISIC2020_EfficientNetB2
wait


# echo "Started training ISIC2016+ISIC2020+PH2_EfficientNetB2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2020 PH2 --CLASSIFIER EfficientNetB2 > ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB2/ISIC2016+ISIC2020+PH2_EfficientNetB2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB2/ISIC2016+ISIC2020+PH2_EfficientNetB2.err
# echo "Ended training ISIC2016+ISIC2020+PH2_EfficientNetB2
wait


# echo "Started training ISIC2016+ISIC2018+ISIC2019+ISIC2020_EfficientNetB2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER EfficientNetB2 > ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB2/ISIC2016+ISIC2018+ISIC2019+ISIC2020_EfficientNetB2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB2/ISIC2016+ISIC2018+ISIC2019+ISIC2020_EfficientNetB2.err
# echo "Ended training ISIC2016+ISIC2018+ISIC2019+ISIC2020_EfficientNetB2
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_EfficientNetB2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER EfficientNetB2 > ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_EfficientNetB2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_EfficientNetB2.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_EfficientNetB2
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_EfficientNetB2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 MEDNODE KaggleMB --CLASSIFIER EfficientNetB2 > ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB2/ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_EfficientNetB2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB2/ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_EfficientNetB2.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_EfficientNetB2
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_EfficientNetB2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 PH2 _7_point_criteria --CLASSIFIER EfficientNetB2 > ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB2/ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_EfficientNetB2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB2/ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_EfficientNetB2.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_EfficientNetB2
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_EfficientNetB2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2020 PH2 --CLASSIFIER EfficientNetB2 > ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB2/ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_EfficientNetB2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB2/ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_EfficientNetB2.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_EfficientNetB2
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_EfficientNetB2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE --CLASSIFIER EfficientNetB2 > ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_EfficientNetB2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_EfficientNetB2.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_EfficientNetB2
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_EfficientNetB2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 MEDNODE KaggleMB --CLASSIFIER EfficientNetB2 > ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_EfficientNetB2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_EfficientNetB2.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_EfficientNetB2
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_EfficientNetB2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 --CLASSIFIER EfficientNetB2 > ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_EfficientNetB2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_EfficientNetB2.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_EfficientNetB2
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_EfficientNetB2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria --CLASSIFIER EfficientNetB2 > ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_EfficientNetB2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_EfficientNetB2.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_EfficientNetB2
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_EfficientNetB2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER EfficientNetB2 > ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_EfficientNetB2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_EfficientNetB2.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_EfficientNetB2
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_EfficientNetB2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 _7_point_criteria PAD_UFES_20 --CLASSIFIER EfficientNetB2 > ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_EfficientNetB2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_EfficientNetB2.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_EfficientNetB2
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_EfficientNetB2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PAD_UFES_20 MEDNODE --CLASSIFIER EfficientNetB2 > ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_EfficientNetB2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_EfficientNetB2.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_EfficientNetB2
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_EfficientNetB2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 PAD_UFES_20 MEDNODE --CLASSIFIER EfficientNetB2 > ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_EfficientNetB2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_EfficientNetB2.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_EfficientNetB2
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_EfficientNetB2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 MEDNODE KaggleMB --CLASSIFIER EfficientNetB2 > ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_EfficientNetB2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_EfficientNetB2.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_EfficientNetB2
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_EfficientNetB2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER EfficientNetB2 > ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_EfficientNetB2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_EfficientNetB2.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_EfficientNetB2
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_EfficientNetB2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria MEDNODE KaggleMB --CLASSIFIER EfficientNetB2 > ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_EfficientNetB2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_EfficientNetB2.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_EfficientNetB2
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_EfficientNetB2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER EfficientNetB2 > ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_EfficientNetB2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/EfficientNetB2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_EfficientNetB2.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_EfficientNetB2
wait

