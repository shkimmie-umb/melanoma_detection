
export CUDA_VISIBLE_DEVICES=3


mkdir -p ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet101_2
echo "Created SLURMS/LOGS/WideResNet101_2"


# echo "Started training ISIC2018_WideResNet101_2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2018 --CLASSIFIER WideResNet101_2 > ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet101_2/ISIC2018_WideResNet101_2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet101_2/ISIC2018_WideResNet101_2.err
# echo "Ended training ISIC2018_WideResNet101_2
wait


# echo "Started training ISIC2019_WideResNet101_2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2019 --CLASSIFIER WideResNet101_2 > ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet101_2/ISIC2019_WideResNet101_2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet101_2/ISIC2019_WideResNet101_2.err
# echo "Ended training ISIC2019_WideResNet101_2
wait


# echo "Started training ISIC2020_WideResNet101_2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 --CLASSIFIER WideResNet101_2 > ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet101_2/ISIC2020_WideResNet101_2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet101_2/ISIC2020_WideResNet101_2.err
# echo "Ended training ISIC2020_WideResNet101_2
wait


# echo "Started training ISIC2016+ISIC2020_WideResNet101_2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2020 --CLASSIFIER WideResNet101_2 > ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet101_2/ISIC2016+ISIC2020_WideResNet101_2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet101_2/ISIC2016+ISIC2020_WideResNet101_2.err
# echo "Ended training ISIC2016+ISIC2020_WideResNet101_2
wait


# echo "Started training ISIC2016+ISIC2020+PH2_WideResNet101_2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2020 PH2 --CLASSIFIER WideResNet101_2 > ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet101_2/ISIC2016+ISIC2020+PH2_WideResNet101_2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet101_2/ISIC2016+ISIC2020+PH2_WideResNet101_2.err
# echo "Ended training ISIC2016+ISIC2020+PH2_WideResNet101_2
wait


# echo "Started training ISIC2016+ISIC2018+ISIC2019+ISIC2020_WideResNet101_2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER WideResNet101_2 > ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet101_2/ISIC2016+ISIC2018+ISIC2019+ISIC2020_WideResNet101_2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet101_2/ISIC2016+ISIC2018+ISIC2019+ISIC2020_WideResNet101_2.err
# echo "Ended training ISIC2016+ISIC2018+ISIC2019+ISIC2020_WideResNet101_2
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_WideResNet101_2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER WideResNet101_2 > ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet101_2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_WideResNet101_2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet101_2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_WideResNet101_2.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_WideResNet101_2
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_WideResNet101_2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 MEDNODE KaggleMB --CLASSIFIER WideResNet101_2 > ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet101_2/ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_WideResNet101_2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet101_2/ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_WideResNet101_2.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_WideResNet101_2
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_WideResNet101_2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 PH2 _7_point_criteria --CLASSIFIER WideResNet101_2 > ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet101_2/ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_WideResNet101_2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet101_2/ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_WideResNet101_2.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_WideResNet101_2
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_WideResNet101_2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2020 PH2 --CLASSIFIER WideResNet101_2 > ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet101_2/ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_WideResNet101_2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet101_2/ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_WideResNet101_2.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_WideResNet101_2
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_WideResNet101_2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE --CLASSIFIER WideResNet101_2 > ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet101_2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_WideResNet101_2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet101_2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_WideResNet101_2.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_WideResNet101_2
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_WideResNet101_2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 MEDNODE KaggleMB --CLASSIFIER WideResNet101_2 > ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet101_2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_WideResNet101_2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet101_2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_WideResNet101_2.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_WideResNet101_2
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_WideResNet101_2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 --CLASSIFIER WideResNet101_2 > ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet101_2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_WideResNet101_2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet101_2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_WideResNet101_2.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_WideResNet101_2
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_WideResNet101_2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria --CLASSIFIER WideResNet101_2 > ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet101_2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_WideResNet101_2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet101_2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_WideResNet101_2.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_WideResNet101_2
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_WideResNet101_2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER WideResNet101_2 > ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet101_2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_WideResNet101_2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet101_2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_WideResNet101_2.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_WideResNet101_2
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_WideResNet101_2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 _7_point_criteria PAD_UFES_20 --CLASSIFIER WideResNet101_2 > ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet101_2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_WideResNet101_2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet101_2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_WideResNet101_2.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_WideResNet101_2
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_WideResNet101_2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PAD_UFES_20 MEDNODE --CLASSIFIER WideResNet101_2 > ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet101_2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_WideResNet101_2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet101_2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_WideResNet101_2.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_WideResNet101_2
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_WideResNet101_2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 PAD_UFES_20 MEDNODE --CLASSIFIER WideResNet101_2 > ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet101_2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_WideResNet101_2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet101_2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_WideResNet101_2.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_WideResNet101_2
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_WideResNet101_2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 MEDNODE KaggleMB --CLASSIFIER WideResNet101_2 > ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet101_2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_WideResNet101_2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet101_2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_WideResNet101_2.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_WideResNet101_2
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_WideResNet101_2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER WideResNet101_2 > ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet101_2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_WideResNet101_2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet101_2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_WideResNet101_2.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_WideResNet101_2
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_WideResNet101_2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria MEDNODE KaggleMB --CLASSIFIER WideResNet101_2 > ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet101_2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_WideResNet101_2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet101_2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_WideResNet101_2.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_WideResNet101_2
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_WideResNet101_2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER WideResNet101_2 > ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet101_2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_WideResNet101_2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet101_2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_WideResNet101_2.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_WideResNet101_2
wait

