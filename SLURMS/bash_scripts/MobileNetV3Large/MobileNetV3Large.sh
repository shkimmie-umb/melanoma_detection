
export CUDA_VISIBLE_DEVICES=1


mkdir -p ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV3Large
echo "Created SLURMS/LOGS/MobileNetV3Large"


# echo "Started training ISIC2018_MobileNetV3Large
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2018 --CLASSIFIER MobileNetV3Large > ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV3Large/ISIC2018_MobileNetV3Large.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV3Large/ISIC2018_MobileNetV3Large.err
# echo "Ended training ISIC2018_MobileNetV3Large
wait


# echo "Started training ISIC2019_MobileNetV3Large
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2019 --CLASSIFIER MobileNetV3Large > ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV3Large/ISIC2019_MobileNetV3Large.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV3Large/ISIC2019_MobileNetV3Large.err
# echo "Ended training ISIC2019_MobileNetV3Large
wait


# echo "Started training ISIC2020_MobileNetV3Large
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 --CLASSIFIER MobileNetV3Large > ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV3Large/ISIC2020_MobileNetV3Large.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV3Large/ISIC2020_MobileNetV3Large.err
# echo "Ended training ISIC2020_MobileNetV3Large
wait


# echo "Started training ISIC2016+ISIC2020_MobileNetV3Large
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2020 --CLASSIFIER MobileNetV3Large > ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV3Large/ISIC2016+ISIC2020_MobileNetV3Large.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV3Large/ISIC2016+ISIC2020_MobileNetV3Large.err
# echo "Ended training ISIC2016+ISIC2020_MobileNetV3Large
wait


# echo "Started training ISIC2016+ISIC2020+PH2_MobileNetV3Large
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2020 PH2 --CLASSIFIER MobileNetV3Large > ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV3Large/ISIC2016+ISIC2020+PH2_MobileNetV3Large.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV3Large/ISIC2016+ISIC2020+PH2_MobileNetV3Large.err
# echo "Ended training ISIC2016+ISIC2020+PH2_MobileNetV3Large
wait


# echo "Started training ISIC2016+ISIC2018+ISIC2019+ISIC2020_MobileNetV3Large
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER MobileNetV3Large > ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV3Large/ISIC2016+ISIC2018+ISIC2019+ISIC2020_MobileNetV3Large.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV3Large/ISIC2016+ISIC2018+ISIC2019+ISIC2020_MobileNetV3Large.err
# echo "Ended training ISIC2016+ISIC2018+ISIC2019+ISIC2020_MobileNetV3Large
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_MobileNetV3Large
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER MobileNetV3Large > ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV3Large/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_MobileNetV3Large.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV3Large/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_MobileNetV3Large.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_MobileNetV3Large
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_MobileNetV3Large
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 MEDNODE KaggleMB --CLASSIFIER MobileNetV3Large > ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV3Large/ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_MobileNetV3Large.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV3Large/ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_MobileNetV3Large.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_MobileNetV3Large
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_MobileNetV3Large
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 PH2 _7_point_criteria --CLASSIFIER MobileNetV3Large > ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV3Large/ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_MobileNetV3Large.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV3Large/ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_MobileNetV3Large.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_MobileNetV3Large
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_MobileNetV3Large
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2020 PH2 --CLASSIFIER MobileNetV3Large > ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV3Large/ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_MobileNetV3Large.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV3Large/ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_MobileNetV3Large.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_MobileNetV3Large
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_MobileNetV3Large
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE --CLASSIFIER MobileNetV3Large > ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV3Large/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_MobileNetV3Large.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV3Large/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_MobileNetV3Large.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_MobileNetV3Large
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_MobileNetV3Large
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 MEDNODE KaggleMB --CLASSIFIER MobileNetV3Large > ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV3Large/ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_MobileNetV3Large.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV3Large/ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_MobileNetV3Large.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_MobileNetV3Large
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_MobileNetV3Large
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 --CLASSIFIER MobileNetV3Large > ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV3Large/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_MobileNetV3Large.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV3Large/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_MobileNetV3Large.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_MobileNetV3Large
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_MobileNetV3Large
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria --CLASSIFIER MobileNetV3Large > ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV3Large/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_MobileNetV3Large.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV3Large/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_MobileNetV3Large.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_MobileNetV3Large
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_MobileNetV3Large
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER MobileNetV3Large > ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV3Large/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_MobileNetV3Large.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV3Large/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_MobileNetV3Large.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_MobileNetV3Large
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_MobileNetV3Large
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 _7_point_criteria PAD_UFES_20 --CLASSIFIER MobileNetV3Large > ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV3Large/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_MobileNetV3Large.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV3Large/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_MobileNetV3Large.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_MobileNetV3Large
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_MobileNetV3Large
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PAD_UFES_20 MEDNODE --CLASSIFIER MobileNetV3Large > ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV3Large/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_MobileNetV3Large.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV3Large/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_MobileNetV3Large.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_MobileNetV3Large
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_MobileNetV3Large
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 PAD_UFES_20 MEDNODE --CLASSIFIER MobileNetV3Large > ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV3Large/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_MobileNetV3Large.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV3Large/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_MobileNetV3Large.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_MobileNetV3Large
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_MobileNetV3Large
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 MEDNODE KaggleMB --CLASSIFIER MobileNetV3Large > ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV3Large/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_MobileNetV3Large.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV3Large/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_MobileNetV3Large.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_MobileNetV3Large
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_MobileNetV3Large
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER MobileNetV3Large > ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV3Large/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_MobileNetV3Large.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV3Large/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_MobileNetV3Large.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_MobileNetV3Large
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_MobileNetV3Large
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria MEDNODE KaggleMB --CLASSIFIER MobileNetV3Large > ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV3Large/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_MobileNetV3Large.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV3Large/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_MobileNetV3Large.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_MobileNetV3Large
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MobileNetV3Large
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER MobileNetV3Large > ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV3Large/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MobileNetV3Large.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/MobileNetV3Large/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MobileNetV3Large.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_MobileNetV3Large
wait

