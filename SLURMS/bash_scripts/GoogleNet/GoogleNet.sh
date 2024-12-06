
export CUDA_VISIBLE_DEVICES=5


mkdir -p ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet
echo "Created SLURMS/LOGS/GoogleNet"


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


# echo "Started training ISIC2016+ISIC2020_GoogleNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2020 --CLASSIFIER GoogleNet > ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2020_GoogleNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2020_GoogleNet.err
# echo "Ended training ISIC2016+ISIC2020_GoogleNet
wait


# echo "Started training ISIC2016+ISIC2020+PH2_GoogleNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2020 PH2 --CLASSIFIER GoogleNet > ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2020+PH2_GoogleNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2020+PH2_GoogleNet.err
# echo "Ended training ISIC2016+ISIC2020+PH2_GoogleNet
wait


# echo "Started training ISIC2016+ISIC2018+ISIC2019+ISIC2020_GoogleNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER GoogleNet > ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2018+ISIC2019+ISIC2020_GoogleNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2018+ISIC2019+ISIC2020_GoogleNet.err
# echo "Ended training ISIC2016+ISIC2018+ISIC2019+ISIC2020_GoogleNet
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_GoogleNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 --CLASSIFIER GoogleNet > ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_GoogleNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_GoogleNet.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020_GoogleNet
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_GoogleNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 MEDNODE KaggleMB --CLASSIFIER GoogleNet > ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_GoogleNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_GoogleNet.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+MEDNODE+KaggleMB_GoogleNet
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_GoogleNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 PH2 _7_point_criteria --CLASSIFIER GoogleNet > ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_GoogleNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_GoogleNet.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+PH2+_7_point_criteria_GoogleNet
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_GoogleNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2020 PH2 --CLASSIFIER GoogleNet > ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_GoogleNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_GoogleNet.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2020+PH2_GoogleNet
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_GoogleNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE --CLASSIFIER GoogleNet > ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_GoogleNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_GoogleNet.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE_GoogleNet
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_GoogleNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 MEDNODE KaggleMB --CLASSIFIER GoogleNet > ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_GoogleNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_GoogleNet.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+MEDNODE+KaggleMB_GoogleNet
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_GoogleNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 --CLASSIFIER GoogleNet > ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_GoogleNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_GoogleNet.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2_GoogleNet
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_GoogleNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria --CLASSIFIER GoogleNet > ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_GoogleNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_GoogleNet.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria_GoogleNet
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_GoogleNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER GoogleNet > ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_GoogleNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_GoogleNet.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+PAD_UFES_20+MEDNODE+KaggleMB_GoogleNet
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_GoogleNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 _7_point_criteria PAD_UFES_20 --CLASSIFIER GoogleNet > ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_GoogleNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_GoogleNet.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+_7_point_criteria+PAD_UFES_20_GoogleNet
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_GoogleNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PAD_UFES_20 MEDNODE --CLASSIFIER GoogleNet > ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_GoogleNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_GoogleNet.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PAD_UFES_20+MEDNODE_GoogleNet
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_GoogleNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 PAD_UFES_20 MEDNODE --CLASSIFIER GoogleNet > ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_GoogleNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_GoogleNet.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+PAD_UFES_20+MEDNODE_GoogleNet
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_GoogleNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 MEDNODE KaggleMB --CLASSIFIER GoogleNet > ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_GoogleNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_GoogleNet.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+MEDNODE+KaggleMB_GoogleNet
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_GoogleNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 --CLASSIFIER GoogleNet > ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_GoogleNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_GoogleNet.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20_GoogleNet
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_GoogleNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria MEDNODE KaggleMB --CLASSIFIER GoogleNet > ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_GoogleNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_GoogleNet.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+MEDNODE+KaggleMB_GoogleNet
wait


# echo "Started training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_GoogleNet
nohup python ~/sansa/melanoma_detection/train_pytorch.py --DB ISIC2016 ISIC2017 ISIC2018 ISIC2019 ISIC2020 PH2 _7_point_criteria PAD_UFES_20 MEDNODE KaggleMB --CLASSIFIER GoogleNet > ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_GoogleNet.log 2> ~/sansa/melanoma_detection/SLURMS/LOGS/GoogleNet/ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_GoogleNet.err
# echo "Ended training ISIC2016+ISIC2017+ISIC2018+ISIC2019+ISIC2020+PH2+_7_point_criteria+PAD_UFES_20+MEDNODE+KaggleMB_GoogleNet
wait

