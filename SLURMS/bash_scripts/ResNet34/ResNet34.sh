
export CUDA_VISIBLE_DEVICES=2


mkdir -p ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34
echo "Created SLURMS/LOGS/ResNet34"


# echo "Started training ISIC2018_ResNet34
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2018 --CLASSIFIER ResNet34 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2018_ResNet34.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2018_ResNet34.err
# echo "Ended training ISIC2018_ResNet34
wait


# echo "Started training ISIC2019_ResNet34
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2019 --CLASSIFIER ResNet34 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2019_ResNet34.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2019_ResNet34.err
# echo "Ended training ISIC2019_ResNet34
wait


# echo "Started training ISIC2020_ResNet34
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 --CLASSIFIER ResNet34 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2020_ResNet34.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2020_ResNet34.err
# echo "Ended training ISIC2020_ResNet34
wait


# echo "Started training ISIC2016+ISIC2020_ResNet34
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2020 --CLASSIFIER ResNet34 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2020_ResNet34.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2020_ResNet34.err
# echo "Ended training ISIC2016+ISIC2020_ResNet34
wait


# echo "Started training ISIC2016+ISIC2020+PH2_ResNet34
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2020 PH2 --CLASSIFIER ResNet34 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2020+PH2_ResNet34.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2020+PH2_ResNet34.err
# echo "Ended training ISIC2016+ISIC2020+PH2_ResNet34
wait


# echo "Started training ISIC2016+ISIC2018+ISIC2019+ISIC2020_ResNet34
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER ResNet34 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2018+ISIC2019+ISIC2020_ResNet34.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2018+ISIC2019+ISIC2020_ResNet34.err
# echo "Ended training ISIC2016+ISIC2018+ISIC2019+ISIC2020_ResNet34
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ResNet34
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER ResNet34 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ResNet34.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ResNet34.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ResNet34
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_ResNet34
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 MEDNODE KaggleMB --CLASSIFIER ResNet34 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_ResNet34.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_ResNet34.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_ResNet34
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_ResNet34
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 PH2 _7_point_criteria --CLASSIFIER ResNet34 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_ResNet34.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_ResNet34.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_ResNet34
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_ResNet34
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2020 PH2 --CLASSIFIER ResNet34 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_ResNet34.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_ResNet34.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_ResNet34
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_ResNet34
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE --CLASSIFIER ResNet34 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_ResNet34.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_ResNet34.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_ResNet34
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_ResNet34
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 MEDNODE KaggleMB --CLASSIFIER ResNet34 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_ResNet34.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_ResNet34.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_ResNet34
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_ResNet34
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 --CLASSIFIER ResNet34 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_ResNet34.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_ResNet34.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_ResNet34
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_ResNet34
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria --CLASSIFIER ResNet34 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_ResNet34.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_ResNet34.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_ResNet34
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_ResNet34
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER ResNet34 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_ResNet34.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_ResNet34.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_ResNet34
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_ResNet34
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 _7_point_criteria PAD_UFES_20 --CLASSIFIER ResNet34 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_ResNet34.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_ResNet34.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_ResNet34
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_ResNet34
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PAD_UFES_20 MEDNODE --CLASSIFIER ResNet34 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_ResNet34.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_ResNet34.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_ResNet34
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_ResNet34
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 PAD_UFES_20 MEDNODE --CLASSIFIER ResNet34 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_ResNet34.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_ResNet34.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_ResNet34
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_ResNet34
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 MEDNODE KaggleMB --CLASSIFIER ResNet34 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_ResNet34.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_ResNet34.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_ResNet34
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNet34
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER ResNet34 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNet34.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNet34.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNet34
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_ResNet34
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria MEDNODE KaggleMB --CLASSIFIER ResNet34 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_ResNet34.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_ResNet34.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_ResNet34
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet34
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER ResNet34 > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet34.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNet34/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet34.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNet34
wait

