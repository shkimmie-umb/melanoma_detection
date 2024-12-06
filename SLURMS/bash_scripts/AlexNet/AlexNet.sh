
export CUDA_VISIBLE_DEVICES=0


mkdir -p ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet
echo "Created SLURMS/LOGS/AlexNet"


# echo "Started training ISIC2018_AlexNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2018 --CLASSIFIER AlexNet > ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2018_AlexNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2018_AlexNet.err
# echo "Ended training ISIC2018_AlexNet
wait


# echo "Started training ISIC2019_AlexNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2019 --CLASSIFIER AlexNet > ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2019_AlexNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2019_AlexNet.err
# echo "Ended training ISIC2019_AlexNet
wait


# echo "Started training ISIC2020_AlexNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 --CLASSIFIER AlexNet > ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2020_AlexNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2020_AlexNet.err
# echo "Ended training ISIC2020_AlexNet
wait


# echo "Started training ISIC2016+ISIC2020_AlexNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2020 --CLASSIFIER AlexNet > ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2020_AlexNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2020_AlexNet.err
# echo "Ended training ISIC2016+ISIC2020_AlexNet
wait


# echo "Started training ISIC2016+ISIC2020+PH2_AlexNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2020 PH2 --CLASSIFIER AlexNet > ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2020+PH2_AlexNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2020+PH2_AlexNet.err
# echo "Ended training ISIC2016+ISIC2020+PH2_AlexNet
wait


# echo "Started training ISIC2016+ISIC2018+ISIC2019+ISIC2020_AlexNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER AlexNet > ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2018+ISIC2019+ISIC2020_AlexNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2018+ISIC2019+ISIC2020_AlexNet.err
# echo "Ended training ISIC2016+ISIC2018+ISIC2019+ISIC2020_AlexNet
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_AlexNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER AlexNet > ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_AlexNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_AlexNet.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_AlexNet
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_AlexNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 MEDNODE KaggleMB --CLASSIFIER AlexNet > ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_AlexNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_AlexNet.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_AlexNet
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_AlexNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 PH2 _7_point_criteria --CLASSIFIER AlexNet > ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_AlexNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_AlexNet.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_AlexNet
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_AlexNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2020 PH2 --CLASSIFIER AlexNet > ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_AlexNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_AlexNet.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_AlexNet
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_AlexNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE --CLASSIFIER AlexNet > ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_AlexNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_AlexNet.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_AlexNet
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_AlexNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 MEDNODE KaggleMB --CLASSIFIER AlexNet > ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_AlexNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_AlexNet.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_AlexNet
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_AlexNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 --CLASSIFIER AlexNet > ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_AlexNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_AlexNet.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_AlexNet
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_AlexNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria --CLASSIFIER AlexNet > ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_AlexNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_AlexNet.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_AlexNet
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_AlexNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER AlexNet > ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_AlexNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_AlexNet.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_AlexNet
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_AlexNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 _7_point_criteria PAD_UFES_20 --CLASSIFIER AlexNet > ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_AlexNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_AlexNet.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_AlexNet
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_AlexNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PAD_UFES_20 MEDNODE --CLASSIFIER AlexNet > ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_AlexNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_AlexNet.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_AlexNet
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_AlexNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 PAD_UFES_20 MEDNODE --CLASSIFIER AlexNet > ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_AlexNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_AlexNet.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_AlexNet
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_AlexNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 MEDNODE KaggleMB --CLASSIFIER AlexNet > ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_AlexNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_AlexNet.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_AlexNet
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_AlexNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER AlexNet > ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_AlexNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_AlexNet.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_AlexNet
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_AlexNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria MEDNODE KaggleMB --CLASSIFIER AlexNet > ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_AlexNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_AlexNet.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_AlexNet
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_AlexNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER AlexNet > ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_AlexNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_AlexNet.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_AlexNet
wait

