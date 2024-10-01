
export CUDA_VISIBLE_DEVICES=0


mkdir -p ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d
echo "Created SLURMS/LOGS/ResNeXt10132x8d"


# echo "Started training ISIC2016_ResNeXt10132x8d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 --CLASSIFIER ResNeXt10132x8d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016_ResNeXt10132x8d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016_ResNeXt10132x8d.err
# echo "Ended training ISIC2016_ResNeXt10132x8d
wait


# echo "Started training ISIC2017_ResNeXt10132x8d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2017 --CLASSIFIER ResNeXt10132x8d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2017_ResNeXt10132x8d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2017_ResNeXt10132x8d.err
# echo "Ended training ISIC2017_ResNeXt10132x8d
wait


# echo "Started training ISIC2018_ResNeXt10132x8d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2018 --CLASSIFIER ResNeXt10132x8d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2018_ResNeXt10132x8d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2018_ResNeXt10132x8d.err
# echo "Ended training ISIC2018_ResNeXt10132x8d
wait


# echo "Started training ISIC2019_ResNeXt10132x8d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2019 --CLASSIFIER ResNeXt10132x8d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2019_ResNeXt10132x8d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2019_ResNeXt10132x8d.err
# echo "Ended training ISIC2019_ResNeXt10132x8d
wait


# echo "Started training ISIC2020_ResNeXt10132x8d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 --CLASSIFIER ResNeXt10132x8d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2020_ResNeXt10132x8d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2020_ResNeXt10132x8d.err
# echo "Ended training ISIC2020_ResNeXt10132x8d
wait


# echo "Started training _7_point_criteria_ResNeXt10132x8d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria --CLASSIFIER ResNeXt10132x8d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/_7_point_criteria_ResNeXt10132x8d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/_7_point_criteria_ResNeXt10132x8d.err
# echo "Ended training _7_point_criteria_ResNeXt10132x8d
wait


# echo "Started training PAD_UFES_20_ResNeXt10132x8d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PAD_UFES_20 --CLASSIFIER ResNeXt10132x8d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/PAD_UFES_20_ResNeXt10132x8d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/PAD_UFES_20_ResNeXt10132x8d.err
# echo "Ended training PAD_UFES_20_ResNeXt10132x8d
wait


# echo "Started training MEDNODE_ResNeXt10132x8d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB MEDNODE --CLASSIFIER ResNeXt10132x8d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/MEDNODE_ResNeXt10132x8d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/MEDNODE_ResNeXt10132x8d.err
# echo "Ended training MEDNODE_ResNeXt10132x8d
wait


# echo "Started training KaggleMB_ResNeXt10132x8d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB KaggleMB --CLASSIFIER ResNeXt10132x8d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/KaggleMB_ResNeXt10132x8d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/KaggleMB_ResNeXt10132x8d.err
# echo "Ended training KaggleMB_ResNeXt10132x8d
wait


# echo "Started training ISIC2016+ISIC2017_ResNeXt10132x8d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 --CLASSIFIER ResNeXt10132x8d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2017_ResNeXt10132x8d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2017_ResNeXt10132x8d.err
# echo "Ended training ISIC2016+ISIC2017_ResNeXt10132x8d
wait


# echo "Started training ISIC2018+ISIC2019_ResNeXt10132x8d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2018 ISIC2019 --CLASSIFIER ResNeXt10132x8d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2018+ISIC2019_ResNeXt10132x8d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2018+ISIC2019_ResNeXt10132x8d.err
# echo "Ended training ISIC2018+ISIC2019_ResNeXt10132x8d
wait


# echo "Started training ISIC2020+PH2_ResNeXt10132x8d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 PH2 --CLASSIFIER ResNeXt10132x8d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2020+PH2_ResNeXt10132x8d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2020+PH2_ResNeXt10132x8d.err
# echo "Ended training ISIC2020+PH2_ResNeXt10132x8d
wait


# echo "Started training _7_point_criteria+PAD_UFES_20_ResNeXt10132x8d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 --CLASSIFIER ResNeXt10132x8d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/_7_point_criteria+PAD_UFES_20_ResNeXt10132x8d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/_7_point_criteria+PAD_UFES_20_ResNeXt10132x8d.err
# echo "Ended training _7_point_criteria+PAD_UFES_20_ResNeXt10132x8d
wait


# echo "Started training MEDNODE+KaggleMB_ResNeXt10132x8d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB MEDNODE KaggleMB --CLASSIFIER ResNeXt10132x8d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/MEDNODE+KaggleMB_ResNeXt10132x8d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/MEDNODE+KaggleMB_ResNeXt10132x8d.err
# echo "Ended training MEDNODE+KaggleMB_ResNeXt10132x8d
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018_ResNeXt10132x8d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 --CLASSIFIER ResNeXt10132x8d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2017+ISIC2018_ResNeXt10132x8d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2017+ISIC2018_ResNeXt10132x8d.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018_ResNeXt10132x8d
wait


# echo "Started training ISIC2019+ISIC2020+PH2_ResNeXt10132x8d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2019 ISIC2020 PH2 --CLASSIFIER ResNeXt10132x8d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2019+ISIC2020+PH2_ResNeXt10132x8d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2019+ISIC2020+PH2_ResNeXt10132x8d.err
# echo "Ended training ISIC2019+ISIC2020+PH2_ResNeXt10132x8d
wait


# echo "Started training _7_point_criteria+PAD_UFES_20+MEDNODE_ResNeXt10132x8d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 MEDNODE --CLASSIFIER ResNeXt10132x8d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/_7_point_criteria+PAD_UFES_20+MEDNODE_ResNeXt10132x8d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/_7_point_criteria+PAD_UFES_20+MEDNODE_ResNeXt10132x8d.err
# echo "Ended training _7_point_criteria+PAD_UFES_20+MEDNODE_ResNeXt10132x8d
wait


# echo "Started training PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt10132x8d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER ResNeXt10132x8d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt10132x8d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt10132x8d.err
# echo "Ended training PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt10132x8d
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019_ResNeXt10132x8d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 --CLASSIFIER ResNeXt10132x8d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2017+ISIC2018+ISIC2019_ResNeXt10132x8d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2017+ISIC2018+ISIC2019_ResNeXt10132x8d.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019_ResNeXt10132x8d
wait


# echo "Started training ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNeXt10132x8d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER ResNeXt10132x8d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNeXt10132x8d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNeXt10132x8d.err
# echo "Ended training ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNeXt10132x8d
wait


# echo "Started training _7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt10132x8d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER ResNeXt10132x8d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt10132x8d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt10132x8d.err
# echo "Ended training _7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt10132x8d
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ResNeXt10132x8d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER ResNeXt10132x8d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ResNeXt10132x8d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ResNeXt10132x8d.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ResNeXt10132x8d
wait


# echo "Started training PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt10132x8d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER ResNeXt10132x8d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt10132x8d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt10132x8d.err
# echo "Ended training PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt10132x8d
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_ResNeXt10132x8d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 --CLASSIFIER ResNeXt10132x8d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_ResNeXt10132x8d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_ResNeXt10132x8d.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_ResNeXt10132x8d
wait


# echo "Started training ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt10132x8d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER ResNeXt10132x8d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt10132x8d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt10132x8d.err
# echo "Ended training ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt10132x8d
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_ResNeXt10132x8d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria --CLASSIFIER ResNeXt10132x8d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_ResNeXt10132x8d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_ResNeXt10132x8d.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_ResNeXt10132x8d
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt10132x8d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER ResNeXt10132x8d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt10132x8d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt10132x8d.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt10132x8d
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNeXt10132x8d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER ResNeXt10132x8d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNeXt10132x8d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNeXt10132x8d.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNeXt10132x8d
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_ResNeXt10132x8d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE --CLASSIFIER ResNeXt10132x8d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_ResNeXt10132x8d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_ResNeXt10132x8d.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_ResNeXt10132x8d
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt10132x8d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER ResNeXt10132x8d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt10132x8d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt10132x8d.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt10132x8d
wait

