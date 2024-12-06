
export CUDA_VISIBLE_DEVICES=4


mkdir -p ~/sansa/melanoma_detection/SLURMS/LOGS/MelaD
echo "Created SLURMS/LOGS/MelaD"


# echo "Started training ISIC2018_MelaD
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2018 --CLASSIFIER MelaD > ~/sansa/melanoma_detection/SLURMS/LOGS/MelaD/ISIC2018_MelaD.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MelaD/ISIC2018_MelaD.err
# echo "Ended training ISIC2018_MelaD
wait


# echo "Started training ISIC2019_MelaD
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2019 --CLASSIFIER MelaD > ~/sansa/melanoma_detection/SLURMS/LOGS/MelaD/ISIC2019_MelaD.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MelaD/ISIC2019_MelaD.err
# echo "Ended training ISIC2019_MelaD
wait


# echo "Started training ISIC2020_MelaD
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 --CLASSIFIER MelaD > ~/sansa/melanoma_detection/SLURMS/LOGS/MelaD/ISIC2020_MelaD.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MelaD/ISIC2020_MelaD.err
# echo "Ended training ISIC2020_MelaD
wait


# echo "Started training ISIC2016+ISIC2020_MelaD
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2020 --CLASSIFIER MelaD > ~/sansa/melanoma_detection/SLURMS/LOGS/MelaD/ISIC2016+ISIC2020_MelaD.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MelaD/ISIC2016+ISIC2020_MelaD.err
# echo "Ended training ISIC2016+ISIC2020_MelaD
wait


# echo "Started training ISIC2016+ISIC2020+PH2_MelaD
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2020 PH2 --CLASSIFIER MelaD > ~/sansa/melanoma_detection/SLURMS/LOGS/MelaD/ISIC2016+ISIC2020+PH2_MelaD.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MelaD/ISIC2016+ISIC2020+PH2_MelaD.err
# echo "Ended training ISIC2016+ISIC2020+PH2_MelaD
wait


# echo "Started training ISIC2016+ISIC2018+ISIC2019+ISIC2020_MelaD
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER MelaD > ~/sansa/melanoma_detection/SLURMS/LOGS/MelaD/ISIC2016+ISIC2018+ISIC2019+ISIC2020_MelaD.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MelaD/ISIC2016+ISIC2018+ISIC2019+ISIC2020_MelaD.err
# echo "Ended training ISIC2016+ISIC2018+ISIC2019+ISIC2020_MelaD
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_MelaD
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER MelaD > ~/sansa/melanoma_detection/SLURMS/LOGS/MelaD/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_MelaD.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MelaD/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_MelaD.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_MelaD
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_MelaD
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 MEDNODE KaggleMB --CLASSIFIER MelaD > ~/sansa/melanoma_detection/SLURMS/LOGS/MelaD/ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_MelaD.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MelaD/ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_MelaD.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_MelaD
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_MelaD
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 PH2 _7_point_criteria --CLASSIFIER MelaD > ~/sansa/melanoma_detection/SLURMS/LOGS/MelaD/ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_MelaD.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MelaD/ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_MelaD.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_MelaD
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_MelaD
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2020 PH2 --CLASSIFIER MelaD > ~/sansa/melanoma_detection/SLURMS/LOGS/MelaD/ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_MelaD.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MelaD/ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_MelaD.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_MelaD
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_MelaD
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE --CLASSIFIER MelaD > ~/sansa/melanoma_detection/SLURMS/LOGS/MelaD/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_MelaD.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MelaD/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_MelaD.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_MelaD
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_MelaD
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 MEDNODE KaggleMB --CLASSIFIER MelaD > ~/sansa/melanoma_detection/SLURMS/LOGS/MelaD/ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_MelaD.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MelaD/ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_MelaD.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_MelaD
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_MelaD
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 --CLASSIFIER MelaD > ~/sansa/melanoma_detection/SLURMS/LOGS/MelaD/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_MelaD.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MelaD/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_MelaD.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_MelaD
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_MelaD
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria --CLASSIFIER MelaD > ~/sansa/melanoma_detection/SLURMS/LOGS/MelaD/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_MelaD.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MelaD/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_MelaD.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_MelaD
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_MelaD
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER MelaD > ~/sansa/melanoma_detection/SLURMS/LOGS/MelaD/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_MelaD.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MelaD/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_MelaD.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_MelaD
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_MelaD
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 _7_point_criteria PAD_UFES_20 --CLASSIFIER MelaD > ~/sansa/melanoma_detection/SLURMS/LOGS/MelaD/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_MelaD.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MelaD/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_MelaD.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_MelaD
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_MelaD
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PAD_UFES_20 MEDNODE --CLASSIFIER MelaD > ~/sansa/melanoma_detection/SLURMS/LOGS/MelaD/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_MelaD.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MelaD/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_MelaD.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_MelaD
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_MelaD
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 PAD_UFES_20 MEDNODE --CLASSIFIER MelaD > ~/sansa/melanoma_detection/SLURMS/LOGS/MelaD/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_MelaD.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MelaD/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_MelaD.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_MelaD
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_MelaD
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 MEDNODE KaggleMB --CLASSIFIER MelaD > ~/sansa/melanoma_detection/SLURMS/LOGS/MelaD/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_MelaD.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MelaD/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_MelaD.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_MelaD
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_MelaD
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER MelaD > ~/sansa/melanoma_detection/SLURMS/LOGS/MelaD/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_MelaD.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MelaD/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_MelaD.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_MelaD
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_MelaD
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria MEDNODE KaggleMB --CLASSIFIER MelaD > ~/sansa/melanoma_detection/SLURMS/LOGS/MelaD/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_MelaD.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MelaD/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_MelaD.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_MelaD
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MelaD
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER MelaD > ~/sansa/melanoma_detection/SLURMS/LOGS/MelaD/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MelaD.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MelaD/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MelaD.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MelaD
wait

