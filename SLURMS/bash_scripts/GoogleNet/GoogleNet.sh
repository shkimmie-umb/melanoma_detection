
export CUDA_VISIBLE_DEVICES=5


mkdir -p ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet
echo "Created SLURMS/LOGS/GoogleNet"


# echo "Started training ISIC2016_GoogleNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 --CLASSIFIER GoogleNet > ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016_GoogleNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016_GoogleNet.err
# echo "Ended training ISIC2016_GoogleNet
wait


# echo "Started training ISIC2017_GoogleNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2017 --CLASSIFIER GoogleNet > ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2017_GoogleNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2017_GoogleNet.err
# echo "Ended training ISIC2017_GoogleNet
wait


# echo "Started training ISIC2018_GoogleNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2018 --CLASSIFIER GoogleNet > ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2018_GoogleNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2018_GoogleNet.err
# echo "Ended training ISIC2018_GoogleNet
wait


# echo "Started training ISIC2019_GoogleNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2019 --CLASSIFIER GoogleNet > ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2019_GoogleNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2019_GoogleNet.err
# echo "Ended training ISIC2019_GoogleNet
wait


# echo "Started training ISIC2020_GoogleNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 --CLASSIFIER GoogleNet > ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2020_GoogleNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2020_GoogleNet.err
# echo "Ended training ISIC2020_GoogleNet
wait


# echo "Started training _7_point_criteria_GoogleNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria --CLASSIFIER GoogleNet > ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/_7_point_criteria_GoogleNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/_7_point_criteria_GoogleNet.err
# echo "Ended training _7_point_criteria_GoogleNet
wait


# echo "Started training PAD_UFES_20_GoogleNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PAD_UFES_20 --CLASSIFIER GoogleNet > ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/PAD_UFES_20_GoogleNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/PAD_UFES_20_GoogleNet.err
# echo "Ended training PAD_UFES_20_GoogleNet
wait


# echo "Started training MEDNODE_GoogleNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB MEDNODE --CLASSIFIER GoogleNet > ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/MEDNODE_GoogleNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/MEDNODE_GoogleNet.err
# echo "Ended training MEDNODE_GoogleNet
wait


# echo "Started training KaggleMB_GoogleNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB KaggleMB --CLASSIFIER GoogleNet > ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/KaggleMB_GoogleNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/KaggleMB_GoogleNet.err
# echo "Ended training KaggleMB_GoogleNet
wait


# echo "Started training ISIC2016+ISIC2017_GoogleNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 --CLASSIFIER GoogleNet > ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2017_GoogleNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2017_GoogleNet.err
# echo "Ended training ISIC2016+ISIC2017_GoogleNet
wait


# echo "Started training ISIC2018+ISIC2019_GoogleNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2018 ISIC2019 --CLASSIFIER GoogleNet > ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2018+ISIC2019_GoogleNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2018+ISIC2019_GoogleNet.err
# echo "Ended training ISIC2018+ISIC2019_GoogleNet
wait


# echo "Started training ISIC2020+PH2_GoogleNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 PH2 --CLASSIFIER GoogleNet > ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2020+PH2_GoogleNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2020+PH2_GoogleNet.err
# echo "Ended training ISIC2020+PH2_GoogleNet
wait


# echo "Started training _7_point_criteria+PAD_UFES_20_GoogleNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 --CLASSIFIER GoogleNet > ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/_7_point_criteria+PAD_UFES_20_GoogleNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/_7_point_criteria+PAD_UFES_20_GoogleNet.err
# echo "Ended training _7_point_criteria+PAD_UFES_20_GoogleNet
wait


# echo "Started training MEDNODE+KaggleMB_GoogleNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB MEDNODE KaggleMB --CLASSIFIER GoogleNet > ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/MEDNODE+KaggleMB_GoogleNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/MEDNODE+KaggleMB_GoogleNet.err
# echo "Ended training MEDNODE+KaggleMB_GoogleNet
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018_GoogleNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 --CLASSIFIER GoogleNet > ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2017+ISIC2018_GoogleNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2017+ISIC2018_GoogleNet.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018_GoogleNet
wait


# echo "Started training ISIC2019+ISIC2020+PH2_GoogleNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2019 ISIC2020 PH2 --CLASSIFIER GoogleNet > ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2019+ISIC2020+PH2_GoogleNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2019+ISIC2020+PH2_GoogleNet.err
# echo "Ended training ISIC2019+ISIC2020+PH2_GoogleNet
wait


# echo "Started training _7_point_criteria+PAD_UFES_20+MEDNODE_GoogleNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 MEDNODE --CLASSIFIER GoogleNet > ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/_7_point_criteria+PAD_UFES_20+MEDNODE_GoogleNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/_7_point_criteria+PAD_UFES_20+MEDNODE_GoogleNet.err
# echo "Ended training _7_point_criteria+PAD_UFES_20+MEDNODE_GoogleNet
wait


# echo "Started training PAD_UFES_20+MEDNODE+KaggleMB_GoogleNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER GoogleNet > ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/PAD_UFES_20+MEDNODE+KaggleMB_GoogleNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/PAD_UFES_20+MEDNODE+KaggleMB_GoogleNet.err
# echo "Ended training PAD_UFES_20+MEDNODE+KaggleMB_GoogleNet
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019_GoogleNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 --CLASSIFIER GoogleNet > ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019_GoogleNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019_GoogleNet.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019_GoogleNet
wait


# echo "Started training ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_GoogleNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER GoogleNet > ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_GoogleNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_GoogleNet.err
# echo "Ended training ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_GoogleNet
wait


# echo "Started training _7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_GoogleNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER GoogleNet > ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_GoogleNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_GoogleNet.err
# echo "Ended training _7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_GoogleNet
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_GoogleNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER GoogleNet > ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_GoogleNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_GoogleNet.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_GoogleNet
wait


# echo "Started training PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_GoogleNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER GoogleNet > ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_GoogleNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_GoogleNet.err
# echo "Ended training PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_GoogleNet
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_GoogleNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 --CLASSIFIER GoogleNet > ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_GoogleNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_GoogleNet.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_GoogleNet
wait


# echo "Started training ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_GoogleNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER GoogleNet > ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_GoogleNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_GoogleNet.err
# echo "Ended training ISIC2016+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_GoogleNet
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_GoogleNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria --CLASSIFIER GoogleNet > ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_GoogleNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_GoogleNet.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_GoogleNet
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_GoogleNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER GoogleNet > ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_GoogleNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_GoogleNet.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_GoogleNet
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_GoogleNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER GoogleNet > ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_GoogleNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_GoogleNet.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_GoogleNet
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_GoogleNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE --CLASSIFIER GoogleNet > ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_GoogleNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_GoogleNet.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE_GoogleNet
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_GoogleNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER GoogleNet > ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_GoogleNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_GoogleNet.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_GoogleNet
wait

