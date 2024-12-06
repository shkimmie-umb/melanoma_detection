
export CUDA_VISIBLE_DEVICES=7


mkdir -p ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d
echo "Created SLURMS/LOGS/ResNeXt10132x8d"


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


# echo "Started training ISIC2016+ISIC2020_ResNeXt10132x8d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2020 --CLASSIFIER ResNeXt10132x8d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2020_ResNeXt10132x8d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2020_ResNeXt10132x8d.err
# echo "Ended training ISIC2016+ISIC2020_ResNeXt10132x8d
wait


# echo "Started training ISIC2016+ISIC2020+PH2_ResNeXt10132x8d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2020 PH2 --CLASSIFIER ResNeXt10132x8d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2020+PH2_ResNeXt10132x8d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2020+PH2_ResNeXt10132x8d.err
# echo "Ended training ISIC2016+ISIC2020+PH2_ResNeXt10132x8d
wait


# echo "Started training ISIC2016+ISIC2018+ISIC2019+ISIC2020_ResNeXt10132x8d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER ResNeXt10132x8d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2018+ISIC2019+ISIC2020_ResNeXt10132x8d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2018+ISIC2019+ISIC2020_ResNeXt10132x8d.err
# echo "Ended training ISIC2016+ISIC2018+ISIC2019+ISIC2020_ResNeXt10132x8d
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ResNeXt10132x8d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER ResNeXt10132x8d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ResNeXt10132x8d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ResNeXt10132x8d.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ResNeXt10132x8d
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_ResNeXt10132x8d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 MEDNODE KaggleMB --CLASSIFIER ResNeXt10132x8d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_ResNeXt10132x8d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_ResNeXt10132x8d.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_ResNeXt10132x8d
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_ResNeXt10132x8d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 PH2 _7_point_criteria --CLASSIFIER ResNeXt10132x8d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_ResNeXt10132x8d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_ResNeXt10132x8d.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_ResNeXt10132x8d
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_ResNeXt10132x8d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2020 PH2 --CLASSIFIER ResNeXt10132x8d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_ResNeXt10132x8d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_ResNeXt10132x8d.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_ResNeXt10132x8d
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_ResNeXt10132x8d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE --CLASSIFIER ResNeXt10132x8d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_ResNeXt10132x8d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_ResNeXt10132x8d.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_ResNeXt10132x8d
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_ResNeXt10132x8d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 MEDNODE KaggleMB --CLASSIFIER ResNeXt10132x8d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_ResNeXt10132x8d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_ResNeXt10132x8d.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_ResNeXt10132x8d
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_ResNeXt10132x8d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 --CLASSIFIER ResNeXt10132x8d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_ResNeXt10132x8d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_ResNeXt10132x8d.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_ResNeXt10132x8d
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_ResNeXt10132x8d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria --CLASSIFIER ResNeXt10132x8d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_ResNeXt10132x8d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_ResNeXt10132x8d.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_ResNeXt10132x8d
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt10132x8d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER ResNeXt10132x8d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt10132x8d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt10132x8d.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt10132x8d
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_ResNeXt10132x8d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 _7_point_criteria PAD_UFES_20 --CLASSIFIER ResNeXt10132x8d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_ResNeXt10132x8d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_ResNeXt10132x8d.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_ResNeXt10132x8d
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_ResNeXt10132x8d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PAD_UFES_20 MEDNODE --CLASSIFIER ResNeXt10132x8d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_ResNeXt10132x8d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_ResNeXt10132x8d.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_ResNeXt10132x8d
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_ResNeXt10132x8d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 PAD_UFES_20 MEDNODE --CLASSIFIER ResNeXt10132x8d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_ResNeXt10132x8d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_ResNeXt10132x8d.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_ResNeXt10132x8d
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_ResNeXt10132x8d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 MEDNODE KaggleMB --CLASSIFIER ResNeXt10132x8d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_ResNeXt10132x8d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_ResNeXt10132x8d.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_ResNeXt10132x8d
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNeXt10132x8d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER ResNeXt10132x8d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNeXt10132x8d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNeXt10132x8d.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNeXt10132x8d
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_ResNeXt10132x8d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria MEDNODE KaggleMB --CLASSIFIER ResNeXt10132x8d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_ResNeXt10132x8d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_ResNeXt10132x8d.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_ResNeXt10132x8d
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt10132x8d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER ResNeXt10132x8d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt10132x8d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt10132x8d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt10132x8d.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt10132x8d
wait

