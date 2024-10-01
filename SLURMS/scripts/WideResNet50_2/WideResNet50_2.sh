
export CUDA_VISIBLE_DEVICES=1


mkdir -p ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2
echo "Created SLURMS/LOGS/WideResNet50_2"


# echo "Started training ISIC2016_WideResNet50_2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 --CLASSIFIER WideResNet50_2 > ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/ISIC2016_WideResNet50_2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/ISIC2016_WideResNet50_2.err
# echo "Ended training ISIC2016_WideResNet50_2
wait


# echo "Started training ISIC2017_WideResNet50_2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2017 --CLASSIFIER WideResNet50_2 > ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/ISIC2017_WideResNet50_2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/ISIC2017_WideResNet50_2.err
# echo "Ended training ISIC2017_WideResNet50_2
wait


# echo "Started training ISIC2018_WideResNet50_2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2018 --CLASSIFIER WideResNet50_2 > ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/ISIC2018_WideResNet50_2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/ISIC2018_WideResNet50_2.err
# echo "Ended training ISIC2018_WideResNet50_2
wait


# echo "Started training ISIC2019_WideResNet50_2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2019 --CLASSIFIER WideResNet50_2 > ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/ISIC2019_WideResNet50_2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/ISIC2019_WideResNet50_2.err
# echo "Ended training ISIC2019_WideResNet50_2
wait


# echo "Started training ISIC2020_WideResNet50_2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 --CLASSIFIER WideResNet50_2 > ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/ISIC2020_WideResNet50_2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/ISIC2020_WideResNet50_2.err
# echo "Ended training ISIC2020_WideResNet50_2
wait


# echo "Started training _7_point_criteria_WideResNet50_2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria --CLASSIFIER WideResNet50_2 > ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/_7_point_criteria_WideResNet50_2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/_7_point_criteria_WideResNet50_2.err
# echo "Ended training _7_point_criteria_WideResNet50_2
wait


# echo "Started training PAD_UFES_20_WideResNet50_2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PAD_UFES_20 --CLASSIFIER WideResNet50_2 > ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/PAD_UFES_20_WideResNet50_2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/PAD_UFES_20_WideResNet50_2.err
# echo "Ended training PAD_UFES_20_WideResNet50_2
wait


# echo "Started training MEDNODE_WideResNet50_2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB MEDNODE --CLASSIFIER WideResNet50_2 > ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/MEDNODE_WideResNet50_2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/MEDNODE_WideResNet50_2.err
# echo "Ended training MEDNODE_WideResNet50_2
wait


# echo "Started training KaggleMB_WideResNet50_2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB KaggleMB --CLASSIFIER WideResNet50_2 > ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/KaggleMB_WideResNet50_2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/KaggleMB_WideResNet50_2.err
# echo "Ended training KaggleMB_WideResNet50_2
wait


# echo "Started training ISIC2016+ISIC2017_WideResNet50_2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 --CLASSIFIER WideResNet50_2 > ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/ISIC2016+ISIC2017_WideResNet50_2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/ISIC2016+ISIC2017_WideResNet50_2.err
# echo "Ended training ISIC2016+ISIC2017_WideResNet50_2
wait


# echo "Started training ISIC2018+ISIC2019_WideResNet50_2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2018 ISIC2019 --CLASSIFIER WideResNet50_2 > ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/ISIC2018+ISIC2019_WideResNet50_2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/ISIC2018+ISIC2019_WideResNet50_2.err
# echo "Ended training ISIC2018+ISIC2019_WideResNet50_2
wait


# echo "Started training ISIC2020+PH2_WideResNet50_2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 PH2 --CLASSIFIER WideResNet50_2 > ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/ISIC2020+PH2_WideResNet50_2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/ISIC2020+PH2_WideResNet50_2.err
# echo "Ended training ISIC2020+PH2_WideResNet50_2
wait


# echo "Started training _7_point_criteria+PAD_UFES_20_WideResNet50_2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 --CLASSIFIER WideResNet50_2 > ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/_7_point_criteria+PAD_UFES_20_WideResNet50_2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/_7_point_criteria+PAD_UFES_20_WideResNet50_2.err
# echo "Ended training _7_point_criteria+PAD_UFES_20_WideResNet50_2
wait


# echo "Started training MEDNODE+KaggleMB_WideResNet50_2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB MEDNODE KaggleMB --CLASSIFIER WideResNet50_2 > ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/MEDNODE+KaggleMB_WideResNet50_2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/MEDNODE+KaggleMB_WideResNet50_2.err
# echo "Ended training MEDNODE+KaggleMB_WideResNet50_2
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018_WideResNet50_2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 --CLASSIFIER WideResNet50_2 > ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/ISIC2016+ISIC2017+ISIC2018_WideResNet50_2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/ISIC2016+ISIC2017+ISIC2018_WideResNet50_2.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018_WideResNet50_2
wait


# echo "Started training ISIC2019+ISIC2020+PH2_WideResNet50_2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2019 ISIC2020 PH2 --CLASSIFIER WideResNet50_2 > ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/ISIC2019+ISIC2020+PH2_WideResNet50_2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/ISIC2019+ISIC2020+PH2_WideResNet50_2.err
# echo "Ended training ISIC2019+ISIC2020+PH2_WideResNet50_2
wait


# echo "Started training _7_point_criteria+PAD_UFES_20+MEDNODE_WideResNet50_2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 MEDNODE --CLASSIFIER WideResNet50_2 > ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/_7_point_criteria+PAD_UFES_20+MEDNODE_WideResNet50_2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/_7_point_criteria+PAD_UFES_20+MEDNODE_WideResNet50_2.err
# echo "Ended training _7_point_criteria+PAD_UFES_20+MEDNODE_WideResNet50_2
wait


# echo "Started training PAD_UFES_20+MEDNODE+KaggleMB_WideResNet50_2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER WideResNet50_2 > ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/PAD_UFES_20+MEDNODE+KaggleMB_WideResNet50_2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/PAD_UFES_20+MEDNODE+KaggleMB_WideResNet50_2.err
# echo "Ended training PAD_UFES_20+MEDNODE+KaggleMB_WideResNet50_2
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019_WideResNet50_2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 --CLASSIFIER WideResNet50_2 > ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/ISIC2016+ISIC2017+ISIC2018+ISIC2019_WideResNet50_2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/ISIC2016+ISIC2017+ISIC2018+ISIC2019_WideResNet50_2.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019_WideResNet50_2
wait


# echo "Started training ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_WideResNet50_2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER WideResNet50_2 > ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_WideResNet50_2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_WideResNet50_2.err
# echo "Ended training ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_WideResNet50_2
wait


# echo "Started training _7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_WideResNet50_2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER WideResNet50_2 > ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_WideResNet50_2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_WideResNet50_2.err
# echo "Ended training _7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_WideResNet50_2
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_WideResNet50_2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER WideResNet50_2 > ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_WideResNet50_2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_WideResNet50_2.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_WideResNet50_2
wait


# echo "Started training PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_WideResNet50_2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER WideResNet50_2 > ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_WideResNet50_2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_WideResNet50_2.err
# echo "Ended training PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_WideResNet50_2
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_WideResNet50_2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 --CLASSIFIER WideResNet50_2 > ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_WideResNet50_2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_WideResNet50_2.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_WideResNet50_2
wait


# echo "Started training ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_WideResNet50_2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER WideResNet50_2 > ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_WideResNet50_2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_WideResNet50_2.err
# echo "Ended training ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_WideResNet50_2
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_WideResNet50_2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria --CLASSIFIER WideResNet50_2 > ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_WideResNet50_2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_WideResNet50_2.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_WideResNet50_2
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_WideResNet50_2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER WideResNet50_2 > ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_WideResNet50_2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_WideResNet50_2.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_WideResNet50_2
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_WideResNet50_2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER WideResNet50_2 > ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_WideResNet50_2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_WideResNet50_2.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_WideResNet50_2
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_WideResNet50_2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE --CLASSIFIER WideResNet50_2 > ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_WideResNet50_2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_WideResNet50_2.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_WideResNet50_2
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_WideResNet50_2
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER WideResNet50_2 > ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_WideResNet50_2.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/WideResNet50_2/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_WideResNet50_2.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_WideResNet50_2
wait

