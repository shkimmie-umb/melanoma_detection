
export CUDA_VISIBLE_DEVICES=3


mkdir -p ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d
echo "Created SLURMS/LOGS/ResNeXt5032x4d"


# echo "Started training ISIC2016_ResNeXt5032x4d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 --CLASSIFIER ResNeXt5032x4d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016_ResNeXt5032x4d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016_ResNeXt5032x4d.err
# echo "Ended training ISIC2016_ResNeXt5032x4d
wait


# echo "Started training ISIC2017_ResNeXt5032x4d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2017 --CLASSIFIER ResNeXt5032x4d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2017_ResNeXt5032x4d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2017_ResNeXt5032x4d.err
# echo "Ended training ISIC2017_ResNeXt5032x4d
wait


# echo "Started training ISIC2018_ResNeXt5032x4d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2018 --CLASSIFIER ResNeXt5032x4d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2018_ResNeXt5032x4d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2018_ResNeXt5032x4d.err
# echo "Ended training ISIC2018_ResNeXt5032x4d
wait


# echo "Started training ISIC2019_ResNeXt5032x4d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2019 --CLASSIFIER ResNeXt5032x4d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2019_ResNeXt5032x4d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2019_ResNeXt5032x4d.err
# echo "Ended training ISIC2019_ResNeXt5032x4d
wait


# echo "Started training ISIC2020_ResNeXt5032x4d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 --CLASSIFIER ResNeXt5032x4d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2020_ResNeXt5032x4d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2020_ResNeXt5032x4d.err
# echo "Ended training ISIC2020_ResNeXt5032x4d
wait


# echo "Started training _7_point_criteria_ResNeXt5032x4d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria --CLASSIFIER ResNeXt5032x4d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/_7_point_criteria_ResNeXt5032x4d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/_7_point_criteria_ResNeXt5032x4d.err
# echo "Ended training _7_point_criteria_ResNeXt5032x4d
wait


# echo "Started training PAD_UFES_20_ResNeXt5032x4d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PAD_UFES_20 --CLASSIFIER ResNeXt5032x4d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/PAD_UFES_20_ResNeXt5032x4d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/PAD_UFES_20_ResNeXt5032x4d.err
# echo "Ended training PAD_UFES_20_ResNeXt5032x4d
wait


# echo "Started training MEDNODE_ResNeXt5032x4d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB MEDNODE --CLASSIFIER ResNeXt5032x4d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/MEDNODE_ResNeXt5032x4d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/MEDNODE_ResNeXt5032x4d.err
# echo "Ended training MEDNODE_ResNeXt5032x4d
wait


# echo "Started training KaggleMB_ResNeXt5032x4d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB KaggleMB --CLASSIFIER ResNeXt5032x4d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/KaggleMB_ResNeXt5032x4d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/KaggleMB_ResNeXt5032x4d.err
# echo "Ended training KaggleMB_ResNeXt5032x4d
wait


# echo "Started training ISIC2016+ISIC2017_ResNeXt5032x4d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 --CLASSIFIER ResNeXt5032x4d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2017_ResNeXt5032x4d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2017_ResNeXt5032x4d.err
# echo "Ended training ISIC2016+ISIC2017_ResNeXt5032x4d
wait


# echo "Started training ISIC2018+ISIC2019_ResNeXt5032x4d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2018 ISIC2019 --CLASSIFIER ResNeXt5032x4d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2018+ISIC2019_ResNeXt5032x4d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2018+ISIC2019_ResNeXt5032x4d.err
# echo "Ended training ISIC2018+ISIC2019_ResNeXt5032x4d
wait


# echo "Started training ISIC2020+PH2_ResNeXt5032x4d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 PH2 --CLASSIFIER ResNeXt5032x4d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2020+PH2_ResNeXt5032x4d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2020+PH2_ResNeXt5032x4d.err
# echo "Ended training ISIC2020+PH2_ResNeXt5032x4d
wait


# echo "Started training _7_point_criteria+PAD_UFES_20_ResNeXt5032x4d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 --CLASSIFIER ResNeXt5032x4d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/_7_point_criteria+PAD_UFES_20_ResNeXt5032x4d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/_7_point_criteria+PAD_UFES_20_ResNeXt5032x4d.err
# echo "Ended training _7_point_criteria+PAD_UFES_20_ResNeXt5032x4d
wait


# echo "Started training MEDNODE+KaggleMB_ResNeXt5032x4d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB MEDNODE KaggleMB --CLASSIFIER ResNeXt5032x4d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/MEDNODE+KaggleMB_ResNeXt5032x4d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/MEDNODE+KaggleMB_ResNeXt5032x4d.err
# echo "Ended training MEDNODE+KaggleMB_ResNeXt5032x4d
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018_ResNeXt5032x4d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 --CLASSIFIER ResNeXt5032x4d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2017+ISIC2018_ResNeXt5032x4d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2017+ISIC2018_ResNeXt5032x4d.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018_ResNeXt5032x4d
wait


# echo "Started training ISIC2019+ISIC2020+PH2_ResNeXt5032x4d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2019 ISIC2020 PH2 --CLASSIFIER ResNeXt5032x4d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2019+ISIC2020+PH2_ResNeXt5032x4d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2019+ISIC2020+PH2_ResNeXt5032x4d.err
# echo "Ended training ISIC2019+ISIC2020+PH2_ResNeXt5032x4d
wait


# echo "Started training _7_point_criteria+PAD_UFES_20+MEDNODE_ResNeXt5032x4d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 MEDNODE --CLASSIFIER ResNeXt5032x4d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/_7_point_criteria+PAD_UFES_20+MEDNODE_ResNeXt5032x4d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/_7_point_criteria+PAD_UFES_20+MEDNODE_ResNeXt5032x4d.err
# echo "Ended training _7_point_criteria+PAD_UFES_20+MEDNODE_ResNeXt5032x4d
wait


# echo "Started training PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt5032x4d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER ResNeXt5032x4d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt5032x4d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt5032x4d.err
# echo "Ended training PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt5032x4d
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019_ResNeXt5032x4d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 --CLASSIFIER ResNeXt5032x4d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2017+ISIC2018+ISIC2019_ResNeXt5032x4d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2017+ISIC2018+ISIC2019_ResNeXt5032x4d.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019_ResNeXt5032x4d
wait


# echo "Started training ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNeXt5032x4d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER ResNeXt5032x4d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNeXt5032x4d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNeXt5032x4d.err
# echo "Ended training ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNeXt5032x4d
wait


# echo "Started training _7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt5032x4d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER ResNeXt5032x4d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt5032x4d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt5032x4d.err
# echo "Ended training _7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt5032x4d
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ResNeXt5032x4d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER ResNeXt5032x4d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ResNeXt5032x4d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ResNeXt5032x4d.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ResNeXt5032x4d
wait


# echo "Started training PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt5032x4d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER ResNeXt5032x4d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt5032x4d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt5032x4d.err
# echo "Ended training PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt5032x4d
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_ResNeXt5032x4d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 --CLASSIFIER ResNeXt5032x4d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_ResNeXt5032x4d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_ResNeXt5032x4d.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_ResNeXt5032x4d
wait


# echo "Started training ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt5032x4d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER ResNeXt5032x4d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt5032x4d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt5032x4d.err
# echo "Ended training ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt5032x4d
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_ResNeXt5032x4d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria --CLASSIFIER ResNeXt5032x4d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_ResNeXt5032x4d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_ResNeXt5032x4d.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_ResNeXt5032x4d
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt5032x4d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER ResNeXt5032x4d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt5032x4d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt5032x4d.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt5032x4d
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNeXt5032x4d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER ResNeXt5032x4d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNeXt5032x4d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNeXt5032x4d.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNeXt5032x4d
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_ResNeXt5032x4d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE --CLASSIFIER ResNeXt5032x4d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_ResNeXt5032x4d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_ResNeXt5032x4d.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_ResNeXt5032x4d
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt5032x4d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER ResNeXt5032x4d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt5032x4d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt5032x4d.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt5032x4d
wait

