
export CUDA_VISIBLE_DEVICES=3


mkdir -p ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10
echo "Created SLURMS/LOGS/SqueezeNet10"


# echo "Started training ISIC2018_SqueezeNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2018 --CLASSIFIER SqueezeNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2018_SqueezeNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2018_SqueezeNet10.err
# echo "Ended training ISIC2018_SqueezeNet10
wait


# echo "Started training ISIC2019_SqueezeNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2019 --CLASSIFIER SqueezeNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2019_SqueezeNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2019_SqueezeNet10.err
# echo "Ended training ISIC2019_SqueezeNet10
wait


# echo "Started training ISIC2020_SqueezeNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 --CLASSIFIER SqueezeNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2020_SqueezeNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2020_SqueezeNet10.err
# echo "Ended training ISIC2020_SqueezeNet10
wait


# echo "Started training ISIC2016+ISIC2020_SqueezeNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2020 --CLASSIFIER SqueezeNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2020_SqueezeNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2020_SqueezeNet10.err
# echo "Ended training ISIC2016+ISIC2020_SqueezeNet10
wait


# echo "Started training ISIC2016+ISIC2020+PH2_SqueezeNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2020 PH2 --CLASSIFIER SqueezeNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2020+PH2_SqueezeNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2020+PH2_SqueezeNet10.err
# echo "Ended training ISIC2016+ISIC2020+PH2_SqueezeNet10
wait


# echo "Started training ISIC2016+ISIC2018+ISIC2019+ISIC2020_SqueezeNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER SqueezeNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2018+ISIC2019+ISIC2020_SqueezeNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2018+ISIC2019+ISIC2020_SqueezeNet10.err
# echo "Ended training ISIC2016+ISIC2018+ISIC2019+ISIC2020_SqueezeNet10
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_SqueezeNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER SqueezeNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_SqueezeNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_SqueezeNet10.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_SqueezeNet10
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_SqueezeNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 MEDNODE KaggleMB --CLASSIFIER SqueezeNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_SqueezeNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_SqueezeNet10.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_SqueezeNet10
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_SqueezeNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 PH2 _7_point_criteria --CLASSIFIER SqueezeNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_SqueezeNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_SqueezeNet10.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_SqueezeNet10
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_SqueezeNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2020 PH2 --CLASSIFIER SqueezeNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_SqueezeNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_SqueezeNet10.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_SqueezeNet10
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_SqueezeNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE --CLASSIFIER SqueezeNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_SqueezeNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_SqueezeNet10.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_SqueezeNet10
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_SqueezeNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 MEDNODE KaggleMB --CLASSIFIER SqueezeNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_SqueezeNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_SqueezeNet10.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_SqueezeNet10
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_SqueezeNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 --CLASSIFIER SqueezeNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_SqueezeNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_SqueezeNet10.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_SqueezeNet10
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_SqueezeNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria --CLASSIFIER SqueezeNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_SqueezeNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_SqueezeNet10.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_SqueezeNet10
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_SqueezeNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER SqueezeNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_SqueezeNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_SqueezeNet10.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_SqueezeNet10
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_SqueezeNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 _7_point_criteria PAD_UFES_20 --CLASSIFIER SqueezeNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_SqueezeNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_SqueezeNet10.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_SqueezeNet10
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_SqueezeNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PAD_UFES_20 MEDNODE --CLASSIFIER SqueezeNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_SqueezeNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_SqueezeNet10.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_SqueezeNet10
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_SqueezeNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 PAD_UFES_20 MEDNODE --CLASSIFIER SqueezeNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_SqueezeNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_SqueezeNet10.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_SqueezeNet10
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_SqueezeNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 MEDNODE KaggleMB --CLASSIFIER SqueezeNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_SqueezeNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_SqueezeNet10.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_SqueezeNet10
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_SqueezeNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER SqueezeNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_SqueezeNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_SqueezeNet10.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_SqueezeNet10
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_SqueezeNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria MEDNODE KaggleMB --CLASSIFIER SqueezeNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_SqueezeNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_SqueezeNet10.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_SqueezeNet10
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_SqueezeNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER SqueezeNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_SqueezeNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_SqueezeNet10.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_SqueezeNet10
wait

