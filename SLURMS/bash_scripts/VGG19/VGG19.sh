
export CUDA_VISIBLE_DEVICES=2


mkdir -p ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19
echo "Created SLURMS/LOGS/VGG19"


# echo "Started training ISIC2018_VGG19
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2018 --CLASSIFIER VGG19 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19/ISIC2018_VGG19.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19/ISIC2018_VGG19.err
# echo "Ended training ISIC2018_VGG19
wait


# echo "Started training ISIC2019_VGG19
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2019 --CLASSIFIER VGG19 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19/ISIC2019_VGG19.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19/ISIC2019_VGG19.err
# echo "Ended training ISIC2019_VGG19
wait


# echo "Started training ISIC2020_VGG19
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 --CLASSIFIER VGG19 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19/ISIC2020_VGG19.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19/ISIC2020_VGG19.err
# echo "Ended training ISIC2020_VGG19
wait


# echo "Started training ISIC2016+ISIC2020_VGG19
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2020 --CLASSIFIER VGG19 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19/ISIC2016+ISIC2020_VGG19.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19/ISIC2016+ISIC2020_VGG19.err
# echo "Ended training ISIC2016+ISIC2020_VGG19
wait


# echo "Started training ISIC2016+ISIC2020+PH2_VGG19
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2020 PH2 --CLASSIFIER VGG19 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19/ISIC2016+ISIC2020+PH2_VGG19.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19/ISIC2016+ISIC2020+PH2_VGG19.err
# echo "Ended training ISIC2016+ISIC2020+PH2_VGG19
wait


# echo "Started training ISIC2016+ISIC2018+ISIC2019+ISIC2020_VGG19
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER VGG19 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19/ISIC2016+ISIC2018+ISIC2019+ISIC2020_VGG19.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19/ISIC2016+ISIC2018+ISIC2019+ISIC2020_VGG19.err
# echo "Ended training ISIC2016+ISIC2018+ISIC2019+ISIC2020_VGG19
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_VGG19
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER VGG19 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_VGG19.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_VGG19.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_VGG19
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_VGG19
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 MEDNODE KaggleMB --CLASSIFIER VGG19 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19/ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_VGG19.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19/ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_VGG19.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_VGG19
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_VGG19
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 PH2 _7_point_criteria --CLASSIFIER VGG19 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19/ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_VGG19.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19/ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_VGG19.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_VGG19
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_VGG19
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2020 PH2 --CLASSIFIER VGG19 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19/ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_VGG19.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19/ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_VGG19.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_VGG19
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_VGG19
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE --CLASSIFIER VGG19 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_VGG19.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_VGG19.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_VGG19
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_VGG19
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 MEDNODE KaggleMB --CLASSIFIER VGG19 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19/ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_VGG19.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19/ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_VGG19.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_VGG19
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_VGG19
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 --CLASSIFIER VGG19 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_VGG19.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_VGG19.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_VGG19
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_VGG19
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria --CLASSIFIER VGG19 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_VGG19.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_VGG19.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_VGG19
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_VGG19
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER VGG19 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_VGG19.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_VGG19.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_VGG19
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_VGG19
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 _7_point_criteria PAD_UFES_20 --CLASSIFIER VGG19 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_VGG19.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_VGG19.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_VGG19
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_VGG19
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PAD_UFES_20 MEDNODE --CLASSIFIER VGG19 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_VGG19.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_VGG19.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_VGG19
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_VGG19
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 PAD_UFES_20 MEDNODE --CLASSIFIER VGG19 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_VGG19.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_VGG19.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_VGG19
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_VGG19
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 MEDNODE KaggleMB --CLASSIFIER VGG19 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_VGG19.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_VGG19.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_VGG19
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_VGG19
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER VGG19 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_VGG19.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_VGG19.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_VGG19
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_VGG19
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria MEDNODE KaggleMB --CLASSIFIER VGG19 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_VGG19.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_VGG19.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_VGG19
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG19
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER VGG19 > ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG19.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/VGG19/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG19.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_VGG19
wait

