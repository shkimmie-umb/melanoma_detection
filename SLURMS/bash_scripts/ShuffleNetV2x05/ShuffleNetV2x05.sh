
export CUDA_VISIBLE_DEVICES=5


mkdir -p ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x05
echo "Created SLURMS/LOGS/ShuffleNetV2x05"


# echo "Started training ISIC2018_ShuffleNetV2x05
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2018 --CLASSIFIER ShuffleNetV2x05 > ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x05/ISIC2018_ShuffleNetV2x05.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x05/ISIC2018_ShuffleNetV2x05.err
# echo "Ended training ISIC2018_ShuffleNetV2x05
wait


# echo "Started training ISIC2019_ShuffleNetV2x05
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2019 --CLASSIFIER ShuffleNetV2x05 > ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x05/ISIC2019_ShuffleNetV2x05.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x05/ISIC2019_ShuffleNetV2x05.err
# echo "Ended training ISIC2019_ShuffleNetV2x05
wait


# echo "Started training ISIC2020_ShuffleNetV2x05
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 --CLASSIFIER ShuffleNetV2x05 > ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x05/ISIC2020_ShuffleNetV2x05.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x05/ISIC2020_ShuffleNetV2x05.err
# echo "Ended training ISIC2020_ShuffleNetV2x05
wait


# echo "Started training ISIC2016+ISIC2020_ShuffleNetV2x05
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2020 --CLASSIFIER ShuffleNetV2x05 > ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x05/ISIC2016+ISIC2020_ShuffleNetV2x05.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x05/ISIC2016+ISIC2020_ShuffleNetV2x05.err
# echo "Ended training ISIC2016+ISIC2020_ShuffleNetV2x05
wait


# echo "Started training ISIC2016+ISIC2020+PH2_ShuffleNetV2x05
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2020 PH2 --CLASSIFIER ShuffleNetV2x05 > ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x05/ISIC2016+ISIC2020+PH2_ShuffleNetV2x05.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x05/ISIC2016+ISIC2020+PH2_ShuffleNetV2x05.err
# echo "Ended training ISIC2016+ISIC2020+PH2_ShuffleNetV2x05
wait


# echo "Started training ISIC2016+ISIC2018+ISIC2019+ISIC2020_ShuffleNetV2x05
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER ShuffleNetV2x05 > ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x05/ISIC2016+ISIC2018+ISIC2019+ISIC2020_ShuffleNetV2x05.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x05/ISIC2016+ISIC2018+ISIC2019+ISIC2020_ShuffleNetV2x05.err
# echo "Ended training ISIC2016+ISIC2018+ISIC2019+ISIC2020_ShuffleNetV2x05
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ShuffleNetV2x05
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER ShuffleNetV2x05 > ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x05/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ShuffleNetV2x05.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x05/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ShuffleNetV2x05.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ShuffleNetV2x05
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_ShuffleNetV2x05
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 MEDNODE KaggleMB --CLASSIFIER ShuffleNetV2x05 > ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x05/ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_ShuffleNetV2x05.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x05/ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_ShuffleNetV2x05.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_ShuffleNetV2x05
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_ShuffleNetV2x05
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 PH2 _7_point_criteria --CLASSIFIER ShuffleNetV2x05 > ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x05/ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_ShuffleNetV2x05.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x05/ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_ShuffleNetV2x05.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_ShuffleNetV2x05
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_ShuffleNetV2x05
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2020 PH2 --CLASSIFIER ShuffleNetV2x05 > ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x05/ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_ShuffleNetV2x05.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x05/ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_ShuffleNetV2x05.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_ShuffleNetV2x05
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_ShuffleNetV2x05
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE --CLASSIFIER ShuffleNetV2x05 > ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x05/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_ShuffleNetV2x05.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x05/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_ShuffleNetV2x05.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_ShuffleNetV2x05
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_ShuffleNetV2x05
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 MEDNODE KaggleMB --CLASSIFIER ShuffleNetV2x05 > ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x05/ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_ShuffleNetV2x05.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x05/ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_ShuffleNetV2x05.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_ShuffleNetV2x05
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_ShuffleNetV2x05
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 --CLASSIFIER ShuffleNetV2x05 > ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x05/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_ShuffleNetV2x05.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x05/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_ShuffleNetV2x05.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_ShuffleNetV2x05
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_ShuffleNetV2x05
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria --CLASSIFIER ShuffleNetV2x05 > ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x05/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_ShuffleNetV2x05.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x05/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_ShuffleNetV2x05.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_ShuffleNetV2x05
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_ShuffleNetV2x05
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER ShuffleNetV2x05 > ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x05/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_ShuffleNetV2x05.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x05/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_ShuffleNetV2x05.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_ShuffleNetV2x05
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_ShuffleNetV2x05
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 _7_point_criteria PAD_UFES_20 --CLASSIFIER ShuffleNetV2x05 > ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x05/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_ShuffleNetV2x05.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x05/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_ShuffleNetV2x05.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_ShuffleNetV2x05
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_ShuffleNetV2x05
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PAD_UFES_20 MEDNODE --CLASSIFIER ShuffleNetV2x05 > ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x05/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_ShuffleNetV2x05.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x05/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_ShuffleNetV2x05.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_ShuffleNetV2x05
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_ShuffleNetV2x05
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 PAD_UFES_20 MEDNODE --CLASSIFIER ShuffleNetV2x05 > ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x05/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_ShuffleNetV2x05.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x05/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_ShuffleNetV2x05.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_ShuffleNetV2x05
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_ShuffleNetV2x05
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 MEDNODE KaggleMB --CLASSIFIER ShuffleNetV2x05 > ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x05/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_ShuffleNetV2x05.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x05/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_ShuffleNetV2x05.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_ShuffleNetV2x05
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ShuffleNetV2x05
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER ShuffleNetV2x05 > ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x05/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ShuffleNetV2x05.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x05/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ShuffleNetV2x05.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ShuffleNetV2x05
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_ShuffleNetV2x05
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria MEDNODE KaggleMB --CLASSIFIER ShuffleNetV2x05 > ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x05/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_ShuffleNetV2x05.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x05/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_ShuffleNetV2x05.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_ShuffleNetV2x05
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ShuffleNetV2x05
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER ShuffleNetV2x05 > ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x05/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ShuffleNetV2x05.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ShuffleNetV2x05/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ShuffleNetV2x05.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ShuffleNetV2x05
wait

