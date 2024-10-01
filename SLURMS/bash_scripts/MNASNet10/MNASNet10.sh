
export CUDA_VISIBLE_DEVICES=7


mkdir -p ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10
echo "Created SLURMS/LOGS/MNASNet10"


# echo "Started training ISIC2016_MNASNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 --CLASSIFIER MNASNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/ISIC2016_MNASNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/ISIC2016_MNASNet10.err
# echo "Ended training ISIC2016_MNASNet10
wait


# echo "Started training ISIC2017_MNASNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2017 --CLASSIFIER MNASNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/ISIC2017_MNASNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/ISIC2017_MNASNet10.err
# echo "Ended training ISIC2017_MNASNet10
wait


# echo "Started training ISIC2018_MNASNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2018 --CLASSIFIER MNASNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/ISIC2018_MNASNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/ISIC2018_MNASNet10.err
# echo "Ended training ISIC2018_MNASNet10
wait


# echo "Started training ISIC2019_MNASNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2019 --CLASSIFIER MNASNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/ISIC2019_MNASNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/ISIC2019_MNASNet10.err
# echo "Ended training ISIC2019_MNASNet10
wait


# echo "Started training ISIC2020_MNASNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 --CLASSIFIER MNASNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/ISIC2020_MNASNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/ISIC2020_MNASNet10.err
# echo "Ended training ISIC2020_MNASNet10
wait


# echo "Started training _7_point_criteria_MNASNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria --CLASSIFIER MNASNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/_7_point_criteria_MNASNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/_7_point_criteria_MNASNet10.err
# echo "Ended training _7_point_criteria_MNASNet10
wait


# echo "Started training PAD_UFES_20_MNASNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PAD_UFES_20 --CLASSIFIER MNASNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/PAD_UFES_20_MNASNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/PAD_UFES_20_MNASNet10.err
# echo "Ended training PAD_UFES_20_MNASNet10
wait


# echo "Started training MEDNODE_MNASNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB MEDNODE --CLASSIFIER MNASNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/MEDNODE_MNASNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/MEDNODE_MNASNet10.err
# echo "Ended training MEDNODE_MNASNet10
wait


# echo "Started training KaggleMB_MNASNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB KaggleMB --CLASSIFIER MNASNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/KaggleMB_MNASNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/KaggleMB_MNASNet10.err
# echo "Ended training KaggleMB_MNASNet10
wait


# echo "Started training ISIC2016+ISIC2017_MNASNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 --CLASSIFIER MNASNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/ISIC2016+ISIC2017_MNASNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/ISIC2016+ISIC2017_MNASNet10.err
# echo "Ended training ISIC2016+ISIC2017_MNASNet10
wait


# echo "Started training ISIC2018+ISIC2019_MNASNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2018 ISIC2019 --CLASSIFIER MNASNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/ISIC2018+ISIC2019_MNASNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/ISIC2018+ISIC2019_MNASNet10.err
# echo "Ended training ISIC2018+ISIC2019_MNASNet10
wait


# echo "Started training ISIC2020+PH2_MNASNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 PH2 --CLASSIFIER MNASNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/ISIC2020+PH2_MNASNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/ISIC2020+PH2_MNASNet10.err
# echo "Ended training ISIC2020+PH2_MNASNet10
wait


# echo "Started training _7_point_criteria+PAD_UFES_20_MNASNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 --CLASSIFIER MNASNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/_7_point_criteria+PAD_UFES_20_MNASNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/_7_point_criteria+PAD_UFES_20_MNASNet10.err
# echo "Ended training _7_point_criteria+PAD_UFES_20_MNASNet10
wait


# echo "Started training MEDNODE+KaggleMB_MNASNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB MEDNODE KaggleMB --CLASSIFIER MNASNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/MEDNODE+KaggleMB_MNASNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/MEDNODE+KaggleMB_MNASNet10.err
# echo "Ended training MEDNODE+KaggleMB_MNASNet10
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018_MNASNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 --CLASSIFIER MNASNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/ISIC2016+ISIC2017+ISIC2018_MNASNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/ISIC2016+ISIC2017+ISIC2018_MNASNet10.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018_MNASNet10
wait


# echo "Started training ISIC2019+ISIC2020+PH2_MNASNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2019 ISIC2020 PH2 --CLASSIFIER MNASNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/ISIC2019+ISIC2020+PH2_MNASNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/ISIC2019+ISIC2020+PH2_MNASNet10.err
# echo "Ended training ISIC2019+ISIC2020+PH2_MNASNet10
wait


# echo "Started training _7_point_criteria+PAD_UFES_20+MEDNODE_MNASNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 MEDNODE --CLASSIFIER MNASNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/_7_point_criteria+PAD_UFES_20+MEDNODE_MNASNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/_7_point_criteria+PAD_UFES_20+MEDNODE_MNASNet10.err
# echo "Ended training _7_point_criteria+PAD_UFES_20+MEDNODE_MNASNet10
wait


# echo "Started training PAD_UFES_20+MEDNODE+KaggleMB_MNASNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER MNASNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/PAD_UFES_20+MEDNODE+KaggleMB_MNASNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/PAD_UFES_20+MEDNODE+KaggleMB_MNASNet10.err
# echo "Ended training PAD_UFES_20+MEDNODE+KaggleMB_MNASNet10
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019_MNASNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 --CLASSIFIER MNASNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019_MNASNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019_MNASNet10.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019_MNASNet10
wait


# echo "Started training ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_MNASNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER MNASNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_MNASNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_MNASNet10.err
# echo "Ended training ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_MNASNet10
wait


# echo "Started training _7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MNASNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER MNASNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MNASNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MNASNet10.err
# echo "Ended training _7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MNASNet10
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_MNASNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER MNASNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_MNASNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_MNASNet10.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_MNASNet10
wait


# echo "Started training PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MNASNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER MNASNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MNASNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MNASNet10.err
# echo "Ended training PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MNASNet10
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_MNASNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 --CLASSIFIER MNASNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_MNASNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_MNASNet10.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_MNASNet10
wait


# echo "Started training ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MNASNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER MNASNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MNASNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MNASNet10.err
# echo "Ended training ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MNASNet10
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_MNASNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria --CLASSIFIER MNASNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_MNASNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_MNASNet10.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_MNASNet10
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_MNASNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER MNASNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_MNASNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_MNASNet10.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_MNASNet10
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_MNASNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER MNASNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_MNASNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_MNASNet10.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_MNASNet10
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_MNASNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE --CLASSIFIER MNASNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_MNASNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_MNASNet10.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_MNASNet10
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MNASNet10
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER MNASNet10 > ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MNASNet10.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MNASNet10/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MNASNet10.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MNASNet10
wait

