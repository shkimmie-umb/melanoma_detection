
export CUDA_VISIBLE_DEVICES=0


mkdir -p ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet
echo "Created SLURMS/LOGS/AlexNet"


# # echo "Started training ISIC2016_AlexNet
# nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 --CLASSIFIER AlexNet > ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016_AlexNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016_AlexNet.err
# # echo "Ended training ISIC2016_AlexNet
# wait


# # echo "Started training ISIC2017_AlexNet
# nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2017 --CLASSIFIER AlexNet > ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2017_AlexNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2017_AlexNet.err
# # echo "Ended training ISIC2017_AlexNet
# wait


# # echo "Started training ISIC2018_AlexNet
# nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2018 --CLASSIFIER AlexNet > ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2018_AlexNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2018_AlexNet.err
# # echo "Ended training ISIC2018_AlexNet
# wait


# # echo "Started training ISIC2019_AlexNet
# nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2019 --CLASSIFIER AlexNet > ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2019_AlexNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2019_AlexNet.err
# # echo "Ended training ISIC2019_AlexNet
# wait


# # echo "Started training ISIC2020_AlexNet
# nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 --CLASSIFIER AlexNet > ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2020_AlexNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2020_AlexNet.err
# # echo "Ended training ISIC2020_AlexNet
# wait


# echo "Started training _7_point_criteria_AlexNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria --CLASSIFIER AlexNet > ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/_7_point_criteria_AlexNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/_7_point_criteria_AlexNet.err
# echo "Ended training _7_point_criteria_AlexNet
wait


# echo "Started training PAD_UFES_20_AlexNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PAD_UFES_20 --CLASSIFIER AlexNet > ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/PAD_UFES_20_AlexNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/PAD_UFES_20_AlexNet.err
# echo "Ended training PAD_UFES_20_AlexNet
wait


# echo "Started training MEDNODE_AlexNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB MEDNODE --CLASSIFIER AlexNet > ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/MEDNODE_AlexNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/MEDNODE_AlexNet.err
# echo "Ended training MEDNODE_AlexNet
wait


# echo "Started training KaggleMB_AlexNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB KaggleMB --CLASSIFIER AlexNet > ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/KaggleMB_AlexNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/KaggleMB_AlexNet.err
# echo "Ended training KaggleMB_AlexNet
wait


# echo "Started training ISIC2016+ISIC2017_AlexNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 --CLASSIFIER AlexNet > ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2017_AlexNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2017_AlexNet.err
# echo "Ended training ISIC2016+ISIC2017_AlexNet
wait


# echo "Started training ISIC2018+ISIC2019_AlexNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2018 ISIC2019 --CLASSIFIER AlexNet > ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2018+ISIC2019_AlexNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2018+ISIC2019_AlexNet.err
# echo "Ended training ISIC2018+ISIC2019_AlexNet
wait


# echo "Started training ISIC2020+PH2_AlexNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 PH2 --CLASSIFIER AlexNet > ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2020+PH2_AlexNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2020+PH2_AlexNet.err
# echo "Ended training ISIC2020+PH2_AlexNet
wait


# echo "Started training _7_point_criteria+PAD_UFES_20_AlexNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 --CLASSIFIER AlexNet > ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/_7_point_criteria+PAD_UFES_20_AlexNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/_7_point_criteria+PAD_UFES_20_AlexNet.err
# echo "Ended training _7_point_criteria+PAD_UFES_20_AlexNet
wait


# echo "Started training MEDNODE+KaggleMB_AlexNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB MEDNODE KaggleMB --CLASSIFIER AlexNet > ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/MEDNODE+KaggleMB_AlexNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/MEDNODE+KaggleMB_AlexNet.err
# echo "Ended training MEDNODE+KaggleMB_AlexNet
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018_AlexNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 --CLASSIFIER AlexNet > ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2017+ISIC2018_AlexNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2017+ISIC2018_AlexNet.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018_AlexNet
wait


# echo "Started training ISIC2019+ISIC2020+PH2_AlexNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2019 ISIC2020 PH2 --CLASSIFIER AlexNet > ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2019+ISIC2020+PH2_AlexNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2019+ISIC2020+PH2_AlexNet.err
# echo "Ended training ISIC2019+ISIC2020+PH2_AlexNet
wait


# echo "Started training _7_point_criteria+PAD_UFES_20+MEDNODE_AlexNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 MEDNODE --CLASSIFIER AlexNet > ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/_7_point_criteria+PAD_UFES_20+MEDNODE_AlexNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/_7_point_criteria+PAD_UFES_20+MEDNODE_AlexNet.err
# echo "Ended training _7_point_criteria+PAD_UFES_20+MEDNODE_AlexNet
wait


# echo "Started training PAD_UFES_20+MEDNODE+KaggleMB_AlexNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER AlexNet > ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/PAD_UFES_20+MEDNODE+KaggleMB_AlexNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/PAD_UFES_20+MEDNODE+KaggleMB_AlexNet.err
# echo "Ended training PAD_UFES_20+MEDNODE+KaggleMB_AlexNet
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019_AlexNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 --CLASSIFIER AlexNet > ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019_AlexNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019_AlexNet.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019_AlexNet
wait


# echo "Started training ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_AlexNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER AlexNet > ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_AlexNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_AlexNet.err
# echo "Ended training ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_AlexNet
wait


# echo "Started training _7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_AlexNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER AlexNet > ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_AlexNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_AlexNet.err
# echo "Ended training _7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_AlexNet
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_AlexNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER AlexNet > ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_AlexNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_AlexNet.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_AlexNet
wait


# echo "Started training PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_AlexNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER AlexNet > ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_AlexNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_AlexNet.err
# echo "Ended training PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_AlexNet
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_AlexNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 --CLASSIFIER AlexNet > ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_AlexNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_AlexNet.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_AlexNet
wait


# echo "Started training ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_AlexNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER AlexNet > ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_AlexNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_AlexNet.err
# echo "Ended training ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_AlexNet
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_AlexNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria --CLASSIFIER AlexNet > ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_AlexNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_AlexNet.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_AlexNet
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_AlexNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER AlexNet > ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_AlexNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_AlexNet.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_AlexNet
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_AlexNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER AlexNet > ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_AlexNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_AlexNet.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_AlexNet
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_AlexNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE --CLASSIFIER AlexNet > ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_AlexNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_AlexNet.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_AlexNet
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_AlexNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER AlexNet > ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_AlexNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/AlexNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_AlexNet.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_AlexNet
wait

