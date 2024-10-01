
export CUDA_VISIBLE_DEVICES=6


mkdir -p ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10
echo "Created SLURMS/LOGS/SqueezeNet10"


# echo "Started training ISIC2016_SqueezeNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 --CLASSIFIER SqueezeNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016_SqueezeNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016_SqueezeNet10.err
# echo "Ended training ISIC2016_SqueezeNet10
wait


# echo "Started training ISIC2017_SqueezeNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2017 --CLASSIFIER SqueezeNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2017_SqueezeNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2017_SqueezeNet10.err
# echo "Ended training ISIC2017_SqueezeNet10
wait


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


# echo "Started training _7_point_criteria_SqueezeNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria --CLASSIFIER SqueezeNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/_7_point_criteria_SqueezeNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/_7_point_criteria_SqueezeNet10.err
# echo "Ended training _7_point_criteria_SqueezeNet10
wait


# echo "Started training PAD_UFES_20_SqueezeNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PAD_UFES_20 --CLASSIFIER SqueezeNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/PAD_UFES_20_SqueezeNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/PAD_UFES_20_SqueezeNet10.err
# echo "Ended training PAD_UFES_20_SqueezeNet10
wait


# echo "Started training MEDNODE_SqueezeNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB MEDNODE --CLASSIFIER SqueezeNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/MEDNODE_SqueezeNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/MEDNODE_SqueezeNet10.err
# echo "Ended training MEDNODE_SqueezeNet10
wait


# echo "Started training KaggleMB_SqueezeNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB KaggleMB --CLASSIFIER SqueezeNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/KaggleMB_SqueezeNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/KaggleMB_SqueezeNet10.err
# echo "Ended training KaggleMB_SqueezeNet10
wait


# echo "Started training ISIC2016+ISIC2017_SqueezeNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 --CLASSIFIER SqueezeNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2017_SqueezeNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2017_SqueezeNet10.err
# echo "Ended training ISIC2016+ISIC2017_SqueezeNet10
wait


# echo "Started training ISIC2018+ISIC2019_SqueezeNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2018 ISIC2019 --CLASSIFIER SqueezeNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2018+ISIC2019_SqueezeNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2018+ISIC2019_SqueezeNet10.err
# echo "Ended training ISIC2018+ISIC2019_SqueezeNet10
wait


# echo "Started training ISIC2020+PH2_SqueezeNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 PH2 --CLASSIFIER SqueezeNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2020+PH2_SqueezeNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2020+PH2_SqueezeNet10.err
# echo "Ended training ISIC2020+PH2_SqueezeNet10
wait


# echo "Started training _7_point_criteria+PAD_UFES_20_SqueezeNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 --CLASSIFIER SqueezeNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/_7_point_criteria+PAD_UFES_20_SqueezeNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/_7_point_criteria+PAD_UFES_20_SqueezeNet10.err
# echo "Ended training _7_point_criteria+PAD_UFES_20_SqueezeNet10
wait


# echo "Started training MEDNODE+KaggleMB_SqueezeNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB MEDNODE KaggleMB --CLASSIFIER SqueezeNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/MEDNODE+KaggleMB_SqueezeNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/MEDNODE+KaggleMB_SqueezeNet10.err
# echo "Ended training MEDNODE+KaggleMB_SqueezeNet10
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018_SqueezeNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 --CLASSIFIER SqueezeNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2017+ISIC2018_SqueezeNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2017+ISIC2018_SqueezeNet10.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018_SqueezeNet10
wait


# echo "Started training ISIC2019+ISIC2020+PH2_SqueezeNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2019 ISIC2020 PH2 --CLASSIFIER SqueezeNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2019+ISIC2020+PH2_SqueezeNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2019+ISIC2020+PH2_SqueezeNet10.err
# echo "Ended training ISIC2019+ISIC2020+PH2_SqueezeNet10
wait


# echo "Started training _7_point_criteria+PAD_UFES_20+MEDNODE_SqueezeNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 MEDNODE --CLASSIFIER SqueezeNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/_7_point_criteria+PAD_UFES_20+MEDNODE_SqueezeNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/_7_point_criteria+PAD_UFES_20+MEDNODE_SqueezeNet10.err
# echo "Ended training _7_point_criteria+PAD_UFES_20+MEDNODE_SqueezeNet10
wait


# echo "Started training PAD_UFES_20+MEDNODE+KaggleMB_SqueezeNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER SqueezeNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/PAD_UFES_20+MEDNODE+KaggleMB_SqueezeNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/PAD_UFES_20+MEDNODE+KaggleMB_SqueezeNet10.err
# echo "Ended training PAD_UFES_20+MEDNODE+KaggleMB_SqueezeNet10
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019_SqueezeNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 --CLASSIFIER SqueezeNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019_SqueezeNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019_SqueezeNet10.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019_SqueezeNet10
wait


# echo "Started training ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_SqueezeNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER SqueezeNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_SqueezeNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_SqueezeNet10.err
# echo "Ended training ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_SqueezeNet10
wait


# echo "Started training _7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_SqueezeNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER SqueezeNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_SqueezeNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_SqueezeNet10.err
# echo "Ended training _7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_SqueezeNet10
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_SqueezeNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER SqueezeNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_SqueezeNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_SqueezeNet10.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_SqueezeNet10
wait


# echo "Started training PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_SqueezeNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER SqueezeNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_SqueezeNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_SqueezeNet10.err
# echo "Ended training PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_SqueezeNet10
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_SqueezeNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 --CLASSIFIER SqueezeNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_SqueezeNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_SqueezeNet10.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_SqueezeNet10
wait


# echo "Started training ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_SqueezeNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER SqueezeNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_SqueezeNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_SqueezeNet10.err
# echo "Ended training ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_SqueezeNet10
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_SqueezeNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria --CLASSIFIER SqueezeNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_SqueezeNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_SqueezeNet10.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_SqueezeNet10
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_SqueezeNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER SqueezeNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_SqueezeNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_SqueezeNet10.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_SqueezeNet10
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_SqueezeNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER SqueezeNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_SqueezeNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_SqueezeNet10.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_SqueezeNet10
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_SqueezeNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE --CLASSIFIER SqueezeNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_SqueezeNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_SqueezeNet10.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_SqueezeNet10
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_SqueezeNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER SqueezeNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_SqueezeNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/SqueezeNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_SqueezeNet10.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_SqueezeNet10
wait

