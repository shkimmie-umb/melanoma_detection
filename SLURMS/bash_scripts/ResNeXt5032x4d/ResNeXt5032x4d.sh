
export CUDA_VISIBLE_DEVICES=6


mkdir -p ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d
echo "Created SLURMS/LOGS/ResNeXt5032x4d"


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


# echo "Started training ISIC2016+ISIC2020_ResNeXt5032x4d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2020 --CLASSIFIER ResNeXt5032x4d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2020_ResNeXt5032x4d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2020_ResNeXt5032x4d.err
# echo "Ended training ISIC2016+ISIC2020_ResNeXt5032x4d
wait


# echo "Started training ISIC2016+ISIC2020+PH2_ResNeXt5032x4d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2020 PH2 --CLASSIFIER ResNeXt5032x4d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2020+PH2_ResNeXt5032x4d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2020+PH2_ResNeXt5032x4d.err
# echo "Ended training ISIC2016+ISIC2020+PH2_ResNeXt5032x4d
wait


# echo "Started training ISIC2016+ISIC2018+ISIC2019+ISIC2020_ResNeXt5032x4d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER ResNeXt5032x4d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2018+ISIC2019+ISIC2020_ResNeXt5032x4d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2018+ISIC2019+ISIC2020_ResNeXt5032x4d.err
# echo "Ended training ISIC2016+ISIC2018+ISIC2019+ISIC2020_ResNeXt5032x4d
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ResNeXt5032x4d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER ResNeXt5032x4d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ResNeXt5032x4d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ResNeXt5032x4d.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_ResNeXt5032x4d
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_ResNeXt5032x4d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 MEDNODE KaggleMB --CLASSIFIER ResNeXt5032x4d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_ResNeXt5032x4d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_ResNeXt5032x4d.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_ResNeXt5032x4d
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_ResNeXt5032x4d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 PH2 _7_point_criteria --CLASSIFIER ResNeXt5032x4d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_ResNeXt5032x4d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_ResNeXt5032x4d.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_ResNeXt5032x4d
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_ResNeXt5032x4d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2020 PH2 --CLASSIFIER ResNeXt5032x4d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_ResNeXt5032x4d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_ResNeXt5032x4d.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_ResNeXt5032x4d
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_ResNeXt5032x4d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE --CLASSIFIER ResNeXt5032x4d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_ResNeXt5032x4d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_ResNeXt5032x4d.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_ResNeXt5032x4d
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_ResNeXt5032x4d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 MEDNODE KaggleMB --CLASSIFIER ResNeXt5032x4d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_ResNeXt5032x4d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_ResNeXt5032x4d.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_ResNeXt5032x4d
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_ResNeXt5032x4d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 --CLASSIFIER ResNeXt5032x4d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_ResNeXt5032x4d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_ResNeXt5032x4d.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_ResNeXt5032x4d
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_ResNeXt5032x4d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria --CLASSIFIER ResNeXt5032x4d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_ResNeXt5032x4d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_ResNeXt5032x4d.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_ResNeXt5032x4d
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt5032x4d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER ResNeXt5032x4d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt5032x4d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt5032x4d.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt5032x4d
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_ResNeXt5032x4d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 _7_point_criteria PAD_UFES_20 --CLASSIFIER ResNeXt5032x4d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_ResNeXt5032x4d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_ResNeXt5032x4d.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_ResNeXt5032x4d
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_ResNeXt5032x4d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PAD_UFES_20 MEDNODE --CLASSIFIER ResNeXt5032x4d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_ResNeXt5032x4d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_ResNeXt5032x4d.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_ResNeXt5032x4d
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_ResNeXt5032x4d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 PAD_UFES_20 MEDNODE --CLASSIFIER ResNeXt5032x4d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_ResNeXt5032x4d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_ResNeXt5032x4d.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_ResNeXt5032x4d
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_ResNeXt5032x4d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 MEDNODE KaggleMB --CLASSIFIER ResNeXt5032x4d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_ResNeXt5032x4d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_ResNeXt5032x4d.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_ResNeXt5032x4d
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNeXt5032x4d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER ResNeXt5032x4d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNeXt5032x4d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNeXt5032x4d.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_ResNeXt5032x4d
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_ResNeXt5032x4d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria MEDNODE KaggleMB --CLASSIFIER ResNeXt5032x4d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_ResNeXt5032x4d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_ResNeXt5032x4d.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_ResNeXt5032x4d
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt5032x4d
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER ResNeXt5032x4d > ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt5032x4d.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/ResNeXt5032x4d/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt5032x4d.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_ResNeXt5032x4d
wait

